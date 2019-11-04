# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:28:15 2017

@author: tkoller
"""

import warnings
import casadi as cas
import numpy as np
from casadi import MX, mtimes, vertcat
from casadi import reshape as cas_reshape

from .gp_reachability_casadi import lin_ellipsoid_safety_distance
from .uncertainty_propagation_casadi import mean_equivalent_multistep, multi_step_taylor_symbolic

ATTR_NAMES_ENV = ['h_mat_safe', 'h_safe', 'lin_model', 'ctrl_bounds', 'h_mat_obs',
                  'h_obs']
DEFAULT_OPT_ENV = {'ctrl_bounds': None, 'safe_policy': None, 'lin_model': None,
                   'h_mat_obs': None, 'h_obs': None}


class CautiousMPC:
    """ Approximate implementation of the Cautious MPC algorithm

    Implementation similar to the "Cautious Model Predictive Control using Gaussian
    Process Regression" https://arxiv.org/abs/1705.10702
    paper with approximate uncertainty propagation under state-feedback control laws
    and chance constraints.

    Attributes
    ----------
    T: int
        The MPC horizon (number of timesteps in trajectroy planning)
    gp: SimpleGPModel
        The Gaussian Process statistical model
    env_options: dict
        Dictionary containing the environment setting (see ?? for details)
    beta_safety: float
        The safety coefficient that influences the cautiousness of the constraints
        (see discussion on constraints in https://arxiv.org/abs/1705.10702)
    rhc: Bool
        True, if we warmstart the NLP in receding horizon MPC style or just initialize
        with default values
    lin_model:
        The linear prior model
    lin_trafo_gp_input:
        A linear transformation of the GP input. Possible application: E.g. to turn off
        a particular dimension that is invariant (e.g. cart position in cart pole
        system)
    perf_trajectory: SX.Function
        The uncertainty propagation technique. E.g. Taylor approximation of the GP
        posterior under uncertaint inputs ( see https://arxiv.org/abs/1705.10702 for an
        overview)
    """

    def __init__(self, T, gp, env_options, beta_safety,
                 lin_trafo_gp_input=None, perf_trajectory="mean_equivalent",
                 k_fb=None):
        self.T = T
        self.gp = gp
        self.n_s = gp.n_s_out
        self.n_u = gp.n_u
        self.n_fail = T - 1
        self.beta_safety = beta_safety
        self._set_perf_trajectory(perf_trajectory)
        self.opt_x0 = False

        self.cost_func = None
        self.lin_prior = False

        self._set_attributes_from_dict(ATTR_NAMES_ENV, DEFAULT_OPT_ENV, env_options)

        self.lin_trafo_gp_input = lin_trafo_gp_input
        if self.lin_trafo_gp_input is None:
            self.lin_trafo_gp_input = MX.eye(self.n_s)

        if self.h_mat_obs is None:
            m_obs_mat = 0
        else:
            m_obs_mat, n_s_obs = np.shape(self.h_mat_obs)
            assert n_s_obs == self.n_s, " Wrong shape of obstacle matrix"
            assert np.shape(self.h_obs) == (m_obs_mat, 1), \
                " Shapes of obstacle linear inequality matrix/vector must match "
        self.m_obs = m_obs_mat

        self.has_ctrl_bounds = False

        if self.ctrl_bounds is not None:
            self.has_ctrl_bounds = True
            assert np.shape(self.ctrl_bounds) == (self.n_u, 2), """control bounds need
                to be of shape n_u x 2 with i,0 lower bound and i,1 upper bound per
                dimension"""

        m_safe_mat, n_s_safe = np.shape(self.h_mat_safe)
        assert n_s_safe == self.n_s, " Wrong shape of safety matrix"
        assert np.shape(self.h_safe) == (m_safe_mat, 1), \
            " Shapes of safety linear inequality matrix/vector must match."
        self.m_safe = m_safe_mat

        self.a = np.eye(self.n_s)
        self.b = np.zeros((self.n_s, self.n_u))

        if self.lin_model is not None:
            self.a, self.b = self.lin_model

        self.eval_prior = lambda x, u: np.dot(x, self.a.T) + np.dot(u, self.b.T)

        self.k_fb = k_fb

    def _set_attributes_from_dict(self, attrib_names, default_attribs={},
                                  custom_attrib={}):
        """ Set Attributes from a dictionary of attribute/value pairs"""

        for attr_name in attrib_names:
            if attr_name in custom_attrib:
                attr_val = custom_attrib[attr_name]
            elif attr_name in default_attribs:
                attr_val = default_attribs[attr_name]
            else:
                raise ValueError(
                    "Neither a custom nor a default value is given for the required "
                    "attribute {}".format(attr_name))

            setattr(self, attr_name, attr_val)

    def _set_perf_trajectory(self, name):
        """ Get the peformance trajectory function from identifier"""
        if name == 'mean_equivalent':
            self.perf_trajectory = mean_equivalent_multistep
        elif name == 'taylor':
            self.perf_trajectory = multi_step_taylor_symbolic
        else:
            raise NotImplementedError("Unknown uncertainty propagation method")

    def init_solver(self, cost_func=None, opt_x0=False):
        """ Initialize a casadi solver object with safety bounds information

        Parameters:
        -----------
        cost_func: Function
            A function which admits casadi.SX type inputs
            and returns a scalar
            If performance controls exist function has to be of the form:
                cost_func(p_all,k_ff_all,x_perf,u_perf)
            otherwise:
                cost_func(p_all,k_ff_all)

        """
        self.opt_x0 = opt_x0
        # Optimization variables
        k_ff = MX.sym("k_ff", (self.T, self.n_u))

        # Parameters
        k_fb = MX.sym("k_fb", (self.n_u, self.n_s))
        mu_0 = MX.sym("mu_0", (self.n_s, 1))

        k_fb_all = [k_fb] * (self.T - 1)

        mu_all, sigma_all, sigma_g = self.perf_trajectory(mu_0, self.gp, k_ff, k_fb_all,
                                                          None, self.a, self.b,
                                                          self.lin_trafo_gp_input)

        self.f_multistep_eval = cas.Function("f_multistep_eval", [mu_0, k_ff, k_fb],
                                             [mu_all, sigma_all, sigma_g])

        g, lbg, ubg, g_name = self.generate_safety_constraints(mu_all, sigma_all,
                                                               k_ff[0, :], k_fb_all,
                                                               k_ff[1:, :])

        if cost_func is None:
            cost_func = self.cost_func
        cost = cost_func(mu_0, k_ff[0, :], mu_all, sigma_all, k_ff[1:, :], k_fb, sigma_g)

        if opt_x0:
            opt_vars = vertcat(mu_0, k_ff)
            opt_params = vertcat(k_fb.reshape((-1, 1)))
        else:
            opt_vars = vertcat(k_ff)
            opt_params = vertcat(mu_0, k_fb.reshape((-1, 1)))

        prob = {'f': cost, 'x': opt_vars, 'p': opt_params, 'g': g}

        opt = {'error_on_fail': False,
               'ipopt': {'hessian_approximation': 'limited-memory', "max_iter": 120,
                         "expect_infeasible_problem": "no",
                         'acceptable_tol': 1e-4, "acceptable_constr_viol_tol": 1e-5,
                         "bound_frac": 0.5, "start_with_resto": "no",
                         "required_infeasibility_reduction": 0.85,
                         "acceptable_iter": 8}}

        solver = cas.nlpsol('solver', 'ipopt', prob, opt)

        self.solver = solver
        self.lbg = lbg
        self.ubg = ubg
        self.solver_initialized = True
        self.g = g
        self.g_name = g_name
        self.cost_func = cost_func

    def get_action(self, x0_mu, verbose=False):
        """ Wrapper around the solve Function

        Parameters
        ----------
        x0_mu: n_x x 0 np.array[float]
            The current state of the system

        Returns
        -------
        u_apply: n_ux0 np.array[float]
            The action to be applied to the system
        exit_code: int
            An exit code that indicates what kind of action is applied. The following
            values are possible:
                0: feasible solution found, optimization succeeded.
                1: Optimization failed to find feasible solution. Old solution applied.
                2: Optimization crashed.
                3: No old feasible solution found, apply k_fb

        """

        assert self.solver_initialized, "Need to initialize the solver first!"

        if self.opt_x0:
            k_ff_0 = np.random.randn(self.T, self.n_u)
            k_fb_0 = self.k_fb
            params = vertcat(cas_reshape(k_fb_0, (-1, 1)))
            opt_vars_init = vertcat(cas_reshape(x0_mu, (-1, 1)), cas_reshape(k_ff_0, (-1, 1)))
        else:
            k_ff_0, k_fb_0 = self._get_init_controls()
            params = vertcat(cas_reshape(x0_mu, (-1, 1)), cas_reshape(k_fb_0, (-1, 1)))
            opt_vars_init = vertcat(cas_reshape(k_ff_0, (-1, 1)))

        crash = False
        # sol = self.solver(x0=opt_vars_init,lbg=self.lbg,ubg=self.ubg,p=params)
        try:
            # pass
            sol = self.solver(x0=opt_vars_init, lbg=self.lbg, ubg=self.ubg, p=params)
        except:
            exit_code = 2
            crash = True
            warnings.warn("NLP solver crashed, solution infeasible")
            sol = None

        return self._get_solution(sol, crash, x0_mu, k_fb_0, verbose)

    def _get_solution(self, sol, crash, p_0, k_fb_0, verbose=False, feas_tol=1e-6):
        if crash:
            self.n_fail += 1
            exit_code = 2
            u_apply, _ = self._get_old_solution(p_0)
            return u_apply, exit_code

        x_opt = sol["x"]
        if self.opt_x0:
            p_0 = cas_reshape(x_opt[:self.n_s], (self.n_s, 1))
            k_ff = cas_reshape(x_opt[self.n_s:], (-1, self.n_u))
        else:
            k_ff = cas_reshape(x_opt, (-1, self.n_u))

        mu_all, sigma_all, sigma_g = self.f_multistep_eval(p_0, k_ff, k_fb_0)

        # Evaluate the constraints
        g_res = sol["g"]
        feasible = True
        if np.any(np.array(self.lbg) - feas_tol > g_res) or np.any(
                np.array(self.ubg) + feas_tol < g_res):
            feasible = False

        if feasible:
            self.k_ff_old = k_ff
            exit_code = 0
            self.n_fail = 0

            if verbose:
                return np.array(k_ff[0, :]).reshape(self.n_u, ), exit_code, np.vstack((p_0.T, mu_all)), sigma_all, k_ff, k_fb_0

            return np.array(k_ff[0, :]).reshape(self.n_u, ), exit_code
        else:
            self.n_fail += 1

            u_apply, exit_code = self._get_old_solution(p_0)

            if verbose:
                return u_apply, exit_code, None, None, None
            return u_apply, exit_code

    def _get_old_solution(self, x0_mu):
        """ Get previous solution in case of infeasibility

        In case of infeasibility, reuse shifted previous feasible solution or fall back
        to the feedback control without feed-forward

        Parameters
        ----------
        x0_mu: n_x x 0 np.array[float]
            The current state of the system

        Returns
        -------
        u_apply: n_ux0 np.array[float]
            The action to be applied to the system
        exit_code: int
            An exit code that indicates what kind of action is applied. The following
            values are possible in this mode:
                1: Optimization failed to find feasible solution. Old solution applied.
                3: No old feasible solution found, apply k_fb
        """
        if self.n_fail < self.T:
            u_apply = self.k_ff_old[self.n_fail, :]
            exit_code = 1
        else:
            u_apply = np.dot(self.k_fb, x0_mu)
            exit_code = 3

        return u_apply.reshape(self.n_u, ), exit_code

    def _get_init_controls(self):
        """ Initialize the NLP in RHC style or with zeros

        In case of infeasibility of the previous NLP, use
        a zero initialization, otherwise shift the previous controls
        in RHC style

        """
        if self.n_fail == 0:
            k_ff_old = np.copy(self.k_ff_old)
            k_ff_0 = np.vstack((k_ff_old[1:, :], k_ff_old[-1, :]))
        else:
            k_ff_0 = np.zeros((self.T, self.n_u))
        k_fb_0 = self.k_fb

        return k_ff_0, k_fb_0

    def generate_safety_constraints(self, p_all, q_all, u_0, k_fb, k_ff):
        """ Generate all safety constraints

        Parameters
        ----------
        p_all:
        q_all:
        k_fb_0:
        k_fb_ctrl:
        k_ff:
        ctrl_bounds:

        Returns
        -------
        g: list[casadi.SX]
        lbg: list[casadi.SX]
        ubg: list[casadi.SX]
        """
        g = []
        lbg = []
        ubg = []
        g_name = []

        H = np.shape(p_all)[0]
        # control constraints
        if self.has_ctrl_bounds:
            g_u_0, lbg_u_0, ubg_u_0 = self._generate_control_constraint(u_0)
            g = vertcat(g, g_u_0)
            lbg += lbg_u_0
            ubg += ubg_u_0
            g_name += ["u_0_ctrl_constraint"]

            for i in range(H - 1):
                p_i = p_all[i, :].T
                q_i = q_all[i, :].reshape((self.n_s, self.n_s))
                k_ff_i = k_ff[i, :].reshape((self.n_u, 1))
                k_fb_i = k_fb[i]

                g_u_i, lbg_u_i, ubg_u_i = self._generate_control_constraint(k_ff_i, q_i,
                                                                            k_fb_i)
                g = vertcat(g, g_u_i)
                lbg += lbg_u_i
                ubg += ubg_u_i
                g_name += ["ellipsoid_ctrl_constraint_{}".format(i)] * len(lbg_u_i)

        # intermediate state constraints
        if not self.h_mat_obs is None:
            for i in range(H):
                p_i = p_all[i, :].T
                q_i = q_all[i, :].reshape((self.n_s, self.n_s))
                g_state = lin_ellipsoid_safety_distance(p_i, q_i, self.h_mat_obs,
                                                        self.h_obs,
                                                        c_safety=self.beta_safety)
                g = vertcat(g, g_state)
                lbg += [-cas.inf] * self.m_obs
                ubg += [0] * self.m_obs
                g_name += ["obstacle_avoidance_constraint{}".format(i)] * self.m_obs

        return g, lbg, ubg, g_name

    def _generate_control_constraint(self, k_ff, q=None, k_fb=None, ctrl_bounds=None):
        """ Build control constraints from state ellipsoids and linear feedback controls

        k_ff: n_u x 1 ndarray[casadi.SX]
            The feed-forward control gain
        q: n_s x n_s ndarray[casadi.SX]
            The shape matrix of the state ellipsoid
        k_fb: n_u x n_s ndarray[casadi.SX]
            The feedback gain
        ctrl_bounds: n_u x 2 ndarray[float], optional

        Returns
        -------
        g: 2*n_u x 1 ndarray[casadi.SX]
            The control constraints (symbollicaly) evaluated at the current
            state/controls
        lbg: 2*n_u x 0 list[float]
            Lower bounds for the control constraints
        ubg: 2*n_u x 0 list[float]
            Upper bounds for the control constraints
        """
        if ctrl_bounds is None:
            if not self.has_ctrl_bounds:
                raise ValueError("""Either ctrl_bounds has to be specified or
                the objects' ctrl_bounds has to be specified """)
            ctrl_bounds = self.ctrl_bounds

        # no feedback term. Reduces to simple feed-forward control bounds

        n_u, _ = np.shape(ctrl_bounds)
        u_min = ctrl_bounds[:, 0]
        u_max = ctrl_bounds[:, 1]

        if k_fb is None:
            return k_ff, u_min.tolist(), u_max.tolist()

        h_vec = np.vstack((u_max[:, None], -u_min[:, None]))
        h_mat = np.vstack((np.eye(n_u), -np.eye(n_u)))

        p_u = k_ff
        q_u = mtimes(k_fb, mtimes(q, k_fb.T))

        g = lin_ellipsoid_safety_distance(p_u, q_u, h_mat, h_vec,
                                          c_safety=self.beta_safety)

        return g, [-cas.inf] * 2 * n_u, [0] * 2 * n_u

    def update_model(self, x, y, opt_hyp=False, replace_old=True,
                     reinitialize_solver=True):
        """ Update the model of the dynamics

        Parameters
        ----------
        x: n x (n_s+n_u) array[float]
            The raw training input (state,action) pairs
        y: n x (n_s) array[float]
            The raw training targets
        opt_hyp: Bool
            True, if the hyperparemeters should be re-optimized
        replace_old: Bool
            True, if old samples should be replaced. Basically, set to True
            if x,y is the whole dataset and to False if it is just additional samples
        reinitialize_solver:
            True, if you want to reinitialize the solver with the updated GP. This is
            necessary to incorporate the new GP into the optimizer. It rarely makes
            sense to set this to False
        """
        n_train = np.shape(x)[0]
        x_s = x[:, :self.n_s].reshape((n_train, self.n_s))
        x_u = x[:, self.n_s:].reshape((n_train, self.n_u))
        y_prior = self.eval_prior(x_s, x_u)
        x_trafo = np.dot(x_s, self.lin_trafo_gp_input.T)

        x = np.hstack((x_trafo, x_u))

        self.gp.update_model(x, y - y_prior, opt_hyp, replace_old)

        if reinitialize_solver:
            self.init_solver(self.cost_func)
        else:
            warnings.warn("""Updating gp without reinitializing the solver! \n
                This is potentially dangerous, since the new GP is not incorporated in
                the MPC""")
