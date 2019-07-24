# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:28:15 2017

@author: tkoller
"""
import warnings
import casadi as cas
import numpy as np
from casadi import MX, mtimes, vertcat, sum2, sqrt
from casadi import reshape as cas_reshape
from .gp_reachability_casadi import lin_ellipsoid_safety_distance
from .gp_reachability_casadi import multi_step_reachability as cas_multistep
from .uncertainty_propagation_casadi import mean_equivalent_multistep, \
    multi_step_taylor_symbolic
from .utils import dlqr, feedback_ctrl, array_of_vec_to_array_of_mat

ATTR_NAMES_PERF = ['type_perf_traj', 'n_perf', 'r', 'perf_has_fb']
DEFAULT_OPT_PERF = {'type_perf_traj': 'mean_equivalent', 'n_perf': 5, 'r': 1,
                    'perf_has_fb': True}

ATTR_NAMES_ENV = ['l_mu', 'l_sigma', 'h_mat_safe', 'h_safe', 'lin_model', 'ctrl_bounds',
                  'safe_policy', 'h_mat_obs', 'h_obs']
DEFAULT_OPT_ENV = {'ctrl_bounds': None, 'safe_policy': None, 'lin_model': None,
                   'h_mat_obs': None, 'h_obs': None}


class SimpleSafeMPC:
    """ Simplified implementation of the SafeMPC algorithm in Casadi

    """

    def __init__(self, n_safe, ssm, opt_env, wx_cost, wu_cost, beta_safety=2.5,
                 rhc=True,
                 safe_policy=None, opt_perf_trajectory={}, lin_trafo_gp_input=None, opts_solver=None, verbosity=0):
        """ Initialize the SafeMPC object with dynamic model information

        Parameters
        ----------
        n_safe: int
            Length of the safety trajectory (number of safe controls)
        ssm: StateSpaceModel
            The underlying statistical model
        opt_env: dict
            Dictionary of environment options. List of accepted attributes are given in ATTR_NAMES_ENV.
            All values that are NOT specified in DEFAULT_OPT_ENV are mandatory.
        wx_cost: n_x x n_x np.ndarray[float]
            State cost matrix for the LQR
        wu_cost: n_u x n_u np.ndarray[float]
            Control cost matrix for the LQR
        beta_safety: float, optional
            The safety coefficient for the confidence intervals (Denoted by \beta in paper)
        rhc: boolean, optional
            True, if we want to reinitialize the SafeMPC problem in a receding horizon fashion
        lin_model: Tuple, optional
            The linear prior model. Consists of tuple (a,b) that defines the linear system.
            Default is: x_{t+1} = x_t, i.e. (I,0)
        ctrl_bounds: n_u x 2 np.ndarray[float], optional
            Upper and lower bound per control input
        safe_policy: function, optional
            Custom safe policy u_t^safe = safe_policy(x_t)
            Default: LQR based on prior model and wx_cost,wu_cost
        opt_perf_trajectory: dict, optional
            Dictionary of environment options. List of accepted attributes are given in ATTR_NAMES_PERF.
            All values that are NOT specified in DEFAULT_OPT_PERF are mandatory.
        lin_trafo_gp_input: n_x_gp_in x n_x np.ndarray[float], optional
            Allows for a linear transformation of the gp input (e.g. removing an input)


        """
        self.rhc = rhc
        self.ssm = ssm
        self.ssm_forward = ssm.get_forward_model_casadi(True)
        self.n_safe = n_safe
        self.n_fail = self.n_safe  # initialize s.t. there is no backup strategy
        self.n_s = self.ssm.num_states
        self.n_u = self.ssm.num_actions
        self.has_openloop = False
        self.opts_solver = opts_solver

        self.safe_policy = safe_policy

        self.cost_func = None  # This is updated wheenver the solver is newly initialized (possibly again with None)

        self._set_attributes_from_dict(ATTR_NAMES_ENV, DEFAULT_OPT_ENV, opt_env)

        self.lin_trafo_gp_input = lin_trafo_gp_input
        if self.lin_trafo_gp_input is None:
            self.lin_trafo_gp_input = np.eye(self.n_s)

        if self.h_mat_obs is None:
            m_obs_mat = 0
        else:
            m_obs_mat, n_s_obs = np.shape(self.h_mat_obs)
            assert n_s_obs == self.n_s, " Wrong shape of obstacle matrix"
            assert np.shape(self.h_obs) == (m_obs_mat,
                                            1), " Shapes of obstacle linear inequality matrix/vector must match "
        self.m_obs = m_obs_mat

        m_safe_mat, n_s_safe = np.shape(self.h_mat_safe)
        assert n_s_safe == self.n_s, " Wrong shape of safety matrix"
        assert np.shape(self.h_safe) == (
            m_safe_mat,
            1), " Shapes of safety linear inequality matrix/vector must match "
        self.m_safe = m_safe_mat

        # init safety constraints evaluator
        p_cas = MX.sym('p', (self.n_s, self.n_u))
        q_cas = MX.sym('q', (self.n_s, self.n_s))
        g_val_term_cas = lin_ellipsoid_safety_distance(p_cas, q_cas, self.h_mat_safe,
                                                       self.h_safe)
        self.g_term_cas = cas.Function("g_term", [p_cas, q_cas], [g_val_term_cas])

        if not self.h_mat_obs is None:
            g_val_interm_cas = lin_ellipsoid_safety_distance(p_cas, q_cas,
                                                             self.h_mat_obs, self.h_obs)
            self.g_interm_cas = cas.Function("g_interm", [p_cas, q_cas],
                                             [g_val_term_cas])

        self.has_ctrl_bounds = False

        if not self.ctrl_bounds is None:
            self.has_ctrl_bounds = True
            assert np.shape(self.ctrl_bounds) == (self.n_u, 2), """control bounds need
            to be of shape n_u x 2 with i,0 lower bound and i,1 upper bound per dimension"""

        self.wx_cost = wx_cost
        self.wu_cost = wu_cost
        self.wx_feedback = wx_cost
        self.wu_feedback = 1 * wu_cost

        self.do_shift_solution = True
        self.solver_initialized = False

        self.beta_safety = beta_safety
        self.verbosity = verbosity

        # SET ALL ATTRIBUTES FOR THE ENVIRONMENT

        self.lin_prior = False
        self.a = np.eye(self.n_s)
        self.b = np.zeros((self.n_s, self.n_u))
        if not self.lin_model is None:
            self.a, self.b = self.lin_model
            self.lin_prior = True
            if self.safe_policy is None:
                # no safe policy specified? Use lqr as safe policy
                K = self.get_lqr_feedback()
                self.safe_policy = lambda x: np.dot(K, x)

        # if self.performance_trajectory is None:
        #    self.performance_trajectory = mean_equivalent
        self._set_attributes_from_dict(ATTR_NAMES_PERF, DEFAULT_OPT_PERF,
                                       opt_perf_trajectory)
        self._set_perf_trajectory(self.type_perf_traj)

        self.k_fb_all = None
        if self.safe_policy is None:
            warnings.warn("No SafePolicy!")

        # init safe

    def init_solver(self, cost_func=None, opt_x0=False, init_uncertainty=False):
        """ Initialize a casadi solver object corresponding to the SafeMPC optimization problem



        Parameters:
        -----------
        cost_func: Function
            A function which admits casadi.SX type inputs
            and returns a scalar function
            If performance controls exist function has to be of the form:
                cost_func(p_all,k_ff_all,x_perf,u_perf)
            otherwise:
                cost_func(p_all,k_ff_all)


        """
        self.cost_func = cost_func
        self.opt_x0 = opt_x0
        self.init_uncertainty = init_uncertainty

        u_0 = MX.sym("init_control", (self.n_u, 1))
        k_ff_all = MX.sym("feed-forward control", (self.n_safe - 1, self.n_u))
        g = []
        lbg = []
        ubg = []
        g_name = []

        p_0 = MX.sym("initial state", (self.n_s, 1))
        q_0 = None
        k_fb_0 = None
        if init_uncertainty:
            q_0 = MX.sym("init uncertainty", (self.n_s, self.n_s))
            k_fb_0 = MX.sym("init feddback control matrix", (self.n_u, self.n_s))

        k_fb_safe = MX.sym("feedback matrices",
                        (self.n_safe - 1, self.n_s * self.n_u))

        p_all, q_all, gp_sigma_pred_safe_all = cas_multistep(p_0, u_0, k_fb_safe, k_ff_all,
                                                             self.ssm_forward, self.l_mu,
                                                             self.l_sigma,
                                                             self.beta_safety, self.a,
                                                             self.b,
                                                             self.lin_trafo_gp_input, q_0, k_fb_0)

        # generate open_loop trajectory function [vertcat(x_0,u_0)],[f_x])

        if init_uncertainty:
            self._f_multistep_eval = cas.Function("safe_multistep",
                                                  [p_0, u_0, k_fb_safe, k_ff_all, q_0, k_fb_0],
                                                  [p_all, q_all, gp_sigma_pred_safe_all])
        else:
            self._f_multistep_eval = cas.Function("safe_multistep",
                                                  [p_0, u_0, k_fb_safe, k_ff_all],
                                                  [p_all, q_all, gp_sigma_pred_safe_all])

        g_safe, lbg_safe, ubg_safe, g_names_safe = self.generate_safety_constraints(
            p_all, q_all, u_0, k_fb_safe, k_ff_all, q_0, k_fb_0)
        g = vertcat(g, g_safe)
        lbg += lbg_safe
        ubg += ubg_safe
        g_name += g_names_safe

        # Generate performance trajectory
        if self.n_perf > 1:
            k_ff_perf, k_fb_perf, k_ff_perf_traj, k_fb_perf_traj, mu_perf, sigma_perf, gp_sigma_pred_perf_all, g_perf, lbg_perf, ubg_perf, g_names_perf = self._generate_perf_trajectory_casadi(
                p_0, u_0, k_ff_all, k_fb_safe, self.a, self.b, self.lin_trafo_gp_input)
            g = vertcat(g, g_perf)
            lbg += lbg_perf
            ubg += ubg_perf
            g_name += g_names_perf
        else:
            k_ff_perf = np.array([])
            k_fb_perf = np.array([])
            k_fb_perf_traj = np.array([])
            k_ff_perf_traj = np.array([])
            mu_perf = np.array([])
            sigma_perf = np.array([])
            gp_sigma_pred_perf_all = None

        cost = self.generate_cost_function(p_0, u_0, p_all, q_all, mu_perf, sigma_perf,
                                           k_ff_all, k_fb_safe, gp_sigma_pred_safe_all,
                                           k_fb_perf=k_fb_perf_traj,
                                           k_ff_perf=k_ff_perf_traj,
                                           gp_pred_sigma_perf=gp_sigma_pred_perf_all,
                                           custom_cost_func=cost_func)

        if self.opt_x0:
            opt_vars = vertcat(p_0, u_0, k_ff_perf, k_ff_all.reshape((-1, 1)))
            opt_params = vertcat(k_fb_safe.reshape((-1, 1)), k_fb_perf.reshape((-1, 1)))
        else:
            opt_vars = vertcat(u_0, k_ff_perf, k_ff_all.reshape((-1, 1)))
            opt_params = vertcat(p_0, k_fb_safe.reshape((-1, 1)), k_fb_perf.reshape((-1, 1)))

        if self.init_uncertainty:
            opt_params = vertcat(opt_params, q_0.reshape((-1, 1)), k_fb_0.reshape((-1, 1)))

        prob = {'f': cost, 'x': opt_vars, 'p': opt_params, 'g': g}

        opt = self.opts_solver
        if opt is None:
            opt = {'error_on_fail': False,
                   'ipopt': {'hessian_approximation': 'limited-memory', "max_iter": 100,
                             "expect_infeasible_problem": "no", \
                             'acceptable_tol': 1e-4, "acceptable_constr_viol_tol": 1e-5,
                             "bound_frac": 0.5, "start_with_resto": "no",
                             "required_infeasibility_reduction": 0.85,
                             "acceptable_iter": 8}}  # ipopt

            # opt = {'max_iter':120,'hessian_approximation':'limited-memory'}#,"c1":5e-4} #sqpmethod #,
        # opt = {'max_iter':120,'qpsol':'qpoases'}

        solver = cas.nlpsol('solver', 'ipopt', prob, opt)
        # solver = cas.nlpsol('solver','sqpmethod',prob,opt)
        # solver = cas.nlpsol('solver','blocksqp',prob,opt)

        self.solver = solver
        self.lbg = lbg
        self.ubg = ubg
        self.solver_initialized = True
        self.g = g
        self.g_name = g_name

    def generate_cost_function(self, p_0, u_0, p_all, q_all, mu_perf, sigma_perf,
                               k_ff_safe, k_fb_safe, sigma_safe, k_fb_perf=None,
                               k_ff_perf=None, gp_pred_sigma_perf=None,
                               custom_cost_func=None, eps_noise=0.0):
        # Generate cost function
        if custom_cost_func is None:
            cost = 0
            if self.n_perf > 1:

                n_cost_deviation = np.minimum(self.n_perf, self.n_safe)
                for i in range(1, n_cost_deviation):
                    cost += mtimes(mu_perf[i, :] - p_all[i, :],
                                   mtimes(.1 * self.wx_cost,
                                          (mu_perf[i, :] - p_all[i, :]).T))

                for i in range(self.n_perf):
                    cost -= sqrt(sum2(gp_pred_sigma_perf[i, :] + eps_noise))
            else:
                for i in range(self.n_safe):
                    cost -= sqrt(sum2(sigma_safe[i, :] + eps_noise))
        else:
            if self.n_perf > 1:
                cost = custom_cost_func(p_0, u_0, p_all, q_all, k_ff_safe, k_fb_safe,
                                        sigma_safe, mu_perf, sigma_perf,
                                        gp_pred_sigma_perf, k_fb_perf, k_ff_perf)
            else:
                cost = custom_cost_func(p_0, u_0, p_all, q_all, k_ff_safe, k_fb_safe,
                                        sigma_safe)

        return cost

    def generate_safety_constraints(self, p_all, q_all, u_0, k_fb, k_ff_all, q_0=None, k_fb_0=None):
        """ Generate all safety constraints

        Parameters
        ----------
        p_all: n_safe x n_s casadi.SX
            The centers of the safe trajctory ellipsoids
        q_all: n_safe x n_s x n_s ndarray[float]

        u_0 The initial
            The shape matrices of the safe trajectory ellipsoids
        k_fb: (n_safe-1) x (n_x * n_u) casadi.SX
            Feedback control matrices
        k_ff_all: (n_safe-1) x n_u casadi.SX
            Feed-forward controls

        Returns
        -------
        g: list[casadi.SX]
            The constraint functions
        lbg: list[casadi.SX]
            Lower bounds for the constraints
        ubg: list[casadi.SX]
            Upper bounds for the constraints
        """
        g = []
        lbg = []
        ubg = []
        g_name = []

        H = np.shape(p_all)[0]
        # control constraints
        if self.has_ctrl_bounds:
            g_u_0, lbg_u_0, ubg_u_0 = self._generate_control_constraint(u_0, q_0, k_fb_0)
            g = vertcat(g, g_u_0)
            lbg += lbg_u_0
            ubg += ubg_u_0
            g_name += ["u_0_ctrl_constraint"]

            for i in range(H - 1):
                p_i = p_all[i, :].T
                q_i = q_all[i, :].reshape((self.n_s, self.n_s))
                k_ff_i = k_ff_all[i, :].reshape((self.n_u, 1))
                k_fb_i = k_fb[i, :].reshape((self.n_u, self.n_s))

                g_u_i, lbg_u_i, ubg_u_i = self._generate_control_constraint(k_ff_i, q_i,
                                                                            k_fb_i)
                g = vertcat(g, g_u_i)
                lbg += lbg_u_i
                ubg += ubg_u_i
                g_name += ["ellipsoid_ctrl_constraint_{}".format(i)] * len(lbg_u_i)

        # intermediate state constraints
        if not self.h_mat_obs is None:
            for i in range(H - 1):
                p_i = p_all[i, :].T
                q_i = q_all[i, :].reshape((self.n_s, self.n_s))
                g_state = lin_ellipsoid_safety_distance(p_i, q_i, self.h_mat_obs,
                                                        self.h_obs)
                g = vertcat(g, g_state)
                lbg += [-cas.inf] * self.m_obs
                ubg += [0] * self.m_obs
                g_name += ["obstacle_avoidance_constraint{}".format(i)] * self.m_obs

        # terminal state constraint
        p_T = p_all[-1, :].T
        q_T = q_all[-1, :].reshape((self.n_s, self.n_s))

        g_terminal = lin_ellipsoid_safety_distance(p_T, q_T, self.h_mat_safe,
                                                   self.h_safe)
        g = vertcat(g, g_terminal)
        g_name += ["terminal constraint"] * self.m_safe
        lbg += [-cas.inf] * self.m_safe
        ubg += [0] * self.m_safe

        return g, lbg, ubg, g_name

    def _generate_perf_trajectory_casadi(self, mu_0, u_0, k_ff_ctrl, k_fb_safe, a=None,
                                         b=None, lin_trafo_gp_input=None,
                                         safety_constr=False):
        """ Generate the performance trajectory variables for the casadi solver

        Parameters:
        mu_0: n_x x 1 casadi.SX
            Initial state
        u_0: n_u x 1 casadi.SX
            Initial control
        k_ff_ctrl: (n_safe-1) x n_u casadi.SX
            Safe feed-forward controls
        k_fb_safe: (n_safe-1) x (n_x * n_u) casadi.SX
            Safe feedback control matrices
        a: n_x x n_x np.ndarray[float], optional
            The A-matrix of the prior linear model
        b: n_x x n_u np.ndarray[float], optional
            The B-matrix of the prior linear model
        lin_trafo_gp_input: n_x_gp_in x n_x np.ndarray[float], optional
            Allows for a linear transformation of the gp input (e.g. removing an input)
        safety_constr: boolean, optional
            True, if we want to put a constraint after (n_safe+1)th performance trajectory state to
            be inside the terminal safe set. Can potentially help with (recursive) feasibility
        """
        if self.r > 1:
            warnings.warn(
                "Coupling performance and safety trajectory for more than one step is UNTESTED")

        # we don't have a performance trajectory, so nothing to do here. Might wanna catch this even before
        if self.n_perf <= 1:
            return np.array([]), np.array([]), np.array([]), np.array(
                []), None, np.array([]), [], [], [], [], []
        else:
            k_ff_perf = MX.sym("k_ff_perf", (self.n_perf - self.r, self.n_u))
            k_ff_perf_traj = vertcat(k_ff_ctrl[:self.r - 1, :], k_ff_perf)

            k_fb_perf_traj = np.array([])
            for i in range(self.r - 1):
                k_fb_perf_traj = np.append(k_fb_perf_traj,
                                           [k_fb_safe[i, :].reshape((self.n_u, self.n_s))])
            if self.perf_has_fb and self.n_perf - self.r > 0:
                k_fb_perf = MX.sym("k_fb_perf", (self.n_u, self.n_s))
                for i in range(self.n_perf - self.r):
                    k_fb_perf_traj = np.append(k_fb_perf_traj, [k_fb_perf])

            mu_perf_all, sigma_perf_all, gp_sigma_pred_perf_all = self.perf_trajectory(
                mu_0, self.ssm_forward, vertcat(u_0, k_ff_perf_traj), k_fb_perf_traj, None, a, b,
                lin_trafo_gp_input)

            # evaluation trajectory (mainly for verbosity)
            mu_0_eval = MX.sym("mu_0", (self.n_s, 1))
            u_0_eval = MX.sym("u_0", (self.n_u, 1))
            k_fb_perf_all_eval = MX.sym("k_fb_perf",
                                        (self.n_perf - 1, self.n_u * self.n_s))
            k_ff_perf_all_eval = MX.sym("k_ff_perf", (self.n_perf - 1, self.n_u))

            list_kfb_perf = [cas_reshape(k_fb_perf_all_eval[i, :], (self.n_u, self.n_s))
                             for i in range(self.n_perf - 1)]
            mu_perf_eval_all, sigma_perf_eval_all, gp_sigma_pred_perf_all_eval = self.perf_trajectory(
                mu_0_eval, self.ssm_forward, vertcat(u_0_eval, k_ff_perf_all_eval),
                list_kfb_perf, None, a, b, lin_trafo_gp_input)
            self._f_multistep_perf_eval = cas.Function("f_multistep_perf_eval",
                                                       [mu_0_eval, u_0_eval,
                                                        k_fb_perf_all_eval,
                                                        k_ff_perf_all_eval],
                                                       [mu_perf_eval_all,
                                                        gp_sigma_pred_perf_all_eval])

        # generate (approxiamte) constraints for the performance trajectory
        g_name = []
        g = []
        lbg = []
        ubg = []
        if safety_constr and self.n_perf > self.n_safe:
            g_name += ["Terminal safety performance"]
            g_term = lin_ellipsoid_safety_distance(
                cas_reshape(mu_perf_all[self.n_safe + 1, :], (self.n_s, 1)),
                cas_reshape(sigma_perf_all[self.n_safe + 1, :], (self.n_s, self.n_s)),
                self.h_mat_safe, self.h_safe)
            g = vertcat(g, g_term)
            lbg += [-np.inf] * self.m_safe
            ubg += [0.] * self.m_safe

        if self.has_ctrl_bounds:
            for i in range(self.n_perf - self.r):
                g_u_i, lbu_i, ubu_i = self._generate_control_constraint(
                    k_ff_perf[i, :].T)
                g = vertcat(g, g_u_i)
                lbg += lbu_i
                ubg += ubu_i
                g_name += ["ctrl_constr_performance_{}".format(i)]

        return k_ff_perf, k_fb_perf, k_ff_perf_traj, k_fb_perf_traj, mu_perf_all, sigma_perf_all, gp_sigma_pred_perf_all, g, lbg, ubg, g_name

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

        g = lin_ellipsoid_safety_distance(p_u, q_u, h_mat, h_vec)

        return g, [-cas.inf] * 2 * n_u, [0] * 2 * n_u

    def _eval_prior_casadi(self, state, action):
        """ symbolically evaluate the prior

        Parameters
        ----------
        state: n x n_s array[casadi.SX]
            Symbolic array of states
        action: n x 1 array[casadi.SX]
            Symbolic array of actions

        Returns
        -------
        x_prior: n x n_s array[casadi.SX]
            The (state,action) pairs evaluated at the prior
        """

        return mtimes(self.a, state.T) + mtimes(self.b, action.T)

    def eval_prior(self, state, action):
        """ Evaluate the prior numerically

        Parameters
        ----------
        state: n x n_s array[float]
            Array of states
        action: n x n_u array[float]

        Returns
        -------
        x_prior: n x n_s array[float]
            The (state,action) pairs evaluated at the prior
        """

        return np.dot(state, self.a.T) + np.dot(action, self.b.T)

    def get_lqr_feedback(self, x_0=None, u_0=None):
        """ Get the initial feedback controller k_fb

        x_0: n_s x 1 ndarray[float], optional
            Current state of the system
        u_0: n_u x 1 ndarray[float], optional
            Initialization of the control input

        """
        q = self.wx_feedback
        r = self.wu_feedback

        if x_0 is None:
            x_0 = np.zeros((self.n_s, 1))
        if u_0 is None:
            u_0 = np.zeros((self.n_u, 1))

        if self.lin_prior:
            a = self.a
            b = self.b

            k_lqr, _, _ = dlqr(a, b, q, r)
            k_fb = -k_lqr
        else:

            raise NotImplementedError(
                "Cannot compute feed-back matrices without prior model")

        return k_fb.reshape((1, self.n_s * self.n_u))

    def get_safety_trajectory_openloop(self, x_0, u_0, k_fb=None, k_ff=None, q_0=None, k_fb_0=None):
        """ Compute a trajectory of ellipsoids based on an initial state and a set of controls

        Parameters
        ----------
        x_0: n_s x 1 2darray[float]
            The initial state
        u_0: n_u x 1 2darray[float]
            The initial action
        k_fb: (n_safe-1) x n_u x n_s  or n_u x n_s ndarray[float], optional
            The feedback controls. Uses the most recent solution to the
            MPC Problem (when calling solve()) if this parameter is not set
        k_ff: (n_safe-1) x n_u, optional
            The feed-forward controls. Uses the most recent solution to the
            MPC Problem (when calling solve()) if this parameter is not set
        get_controls: bool, optional
            Additionally returns the applied controls if this flag is set to TRUE
        Returns
        -------
        p_all: T x n_s ndarray[float]
            The centers of the trajctory ellipsoids
        q_all: T x n_s x n_s ndarray[float]
            The shape matrices of the trajectory ellipsoids
        """
        if not self.has_openloop:
            return None, None

        if k_fb is None:
            k_fb = np.array(self.k_fb_all)

        if k_ff is None:
            k_ff = np.array(self.k_ff_all)

        if self.init_uncertainty:
            p_all, q_all, gp_sigma_pred_safe_all = self._f_multistep_eval(x_0, u_0, k_fb, k_ff, q_0, k_fb_0)
        else:
            p_all, q_all, gp_sigma_pred_safe_all = self._f_multistep_eval(x_0, u_0, k_fb, k_ff)

        return p_all, q_all, gp_sigma_pred_safe_all

    def get_action(self, x0_mu, lqr_only=False, sol_verbose=False):
        """ Wrapper around the solve function

        Parameters
        ----------
        x0_mu: n_s x 0 1darray[float]
            The current state of the system

        Returns
        -------
        u_apply: n_u x 0 1darray[float]
            The action to be applied to the system
        success: bool
            The control was not successful if we are outside the safezone
            AND we have to revert to the safe controller.
        """
        safety_failure = False
        if lqr_only:
            u_apply = self.safe_policy(x0_mu)

            return u_apply, safety_failure

        if sol_verbose:
            _, u_apply, feasible, success, k_fb_apply, k_ff_all, p_all, q_all = self.solve(
                x0_mu[:, None], sol_verbose=True)
            return u_apply.reshape(
                self.n_u, ), feasible, success, k_fb_apply, k_ff_all, p_all, q_all

        else:
            _, u_apply, success = self.solve(x0_mu[:, None])

            return u_apply.reshape(self.n_u, ), success

    def solve(self, p_0, u_0=None, k_ff_all_0=None, k_fb_safe=None, u_perf_0=None,
              k_fb_perf_0=None, sol_verbose=False, q_0=None, k_fb_0=None):
        """ Solve the MPC problem for a given set of input parameters


        Parameters
        ----------
        p_0: n_s x 1 array[float]
            The initial (current) state
        k_ff_all_0: n_safe x n_u  array[float], optional
            The initialization of the feed-forward controls
        k_fb_all_0: n_safe x (n_s * n_u) array[float], optional
            The initialization of the feedback controls

        Returns
        -------
        k_fb_apply: n_u x n_s array[float]
            The feedback control term to be applied to the system
        k_ff_apply: n_u x 1 array[float]
            The feed-forward control term to be applied to the system
        k_fb_all: n_safe x n_u x n_s
            The feedback control terms for all time steps
        k_ff_all: n_safe x n_u x 1
        """
        assert self.solver_initialized, "Need to initialize the solver first!"

        u_0_init, k_ff_all_0_init, k_fb_safe_init, u_perf_0_init, k_fb_perf_0_init = self._get_init_controls()

        if u_0 is None:
            u_0 = u_0_init
        if k_ff_all_0 is None:
            k_ff_all_0 = k_ff_all_0_init
        if k_fb_safe is None:
            k_fb_safe = k_fb_safe_init
        if u_perf_0 is None:
            u_perf_0 = u_perf_0_init
        if k_fb_perf_0 is None:
            k_fb_perf_0 = k_fb_perf_0_init
        if q_0 is not None:
            if k_fb_0 is None:
                k_fb_0 = self.get_lqr_feedback()

        if self.opt_x0:
            params = np.vstack(
                (cas_reshape(k_fb_safe, (-1, 1)), cas_reshape(k_fb_perf_0, (-1, 1))))

            opt_vars_init = vertcat(cas_reshape(p_0, (-1, 1)), cas_reshape(u_0, (-1, 1)), u_perf_0, \
                               cas_reshape(k_ff_all_0, (-1, 1)))
        else:
            params = np.vstack(
                (p_0, cas_reshape(k_fb_safe, (-1, 1)), cas_reshape(k_fb_perf_0, (-1, 1))))

            opt_vars_init = vertcat(cas_reshape(u_0, (-1, 1)), u_perf_0, \
                             cas_reshape(k_ff_all_0, (-1, 1)))

        if self.init_uncertainty:
            params = vertcat(params, cas_reshape(q_0, (-1, 1)), cas_reshape(k_fb_0, (-1, 1)))

        crash = False
         
        sol = self.solver(x0=opt_vars_init, lbg=self.lbg, ubg=self.ubg, p=params)
        try:
            # pass
            sol = self.solver(x0=opt_vars_init, lbg=self.lbg, ubg=self.ubg, p=params)
        except:
            crash = True
            warnings.warn("NLP solver crashed, solution infeasible")
            sol = None

        return self._get_solution(p_0, sol, k_fb_safe, k_fb_perf_0, sol_verbose, crash, q_0=q_0, k_fb_0=k_fb_0)

    def _get_solution(self, x_0, sol, k_fb, k_fb_perf_0, sol_verbose=False,
                      crashed=False, feas_tol=1e-6, q_0=None, k_fb_0=None):
        """ Process the solution dict of the casadi solver

        Processes the solution dictionary of the casadi solver and
        (depending on the chosen mode) saves the solution for reuse in the next
        time step. Depending on the chosen verbosity level, it also prints
        some statistics.

        Parameters
        ----------
        sol: dict
            The solution dictionary returned by the casadi solver
        sol_verbose: boolean, optional
            Return additional solver results such as the constraint values

        Returns
        -------
        k_fb_apply: n_u x n_s array[float]
            The feedback control term to be applied to the system
        k_ff_apply: n_u x 1 array[float]
            The feed-forward control term to be applied to the system
        k_fb_all: n_safe x n_u x n_s
            The feedback control terms for all time steps
        k_ff_all: n_safe x n_u x 1

        h_values: (m_obs*n_safe + m_safe) x 0 array[float], optional
            The values of the constraint evaluation (distance to obstacle)
        """

        success = True
        feasible = True
        if crashed:
            feasible = False

            if self.verbosity > 1:
                print("Optimization crashed, infeasible soluion!")
        else:
            g_res = np.array(sol["g"]).squeeze()

            # This is not sufficient, since casadi gives out wrong feasibility values
            if np.any(np.array(self.lbg) - feas_tol > g_res) or np.any(
                    np.array(self.ubg) + feas_tol < g_res):
                feasible = False

            x_opt = sol["x"]
            self.has_openloop = True

            if self.opt_x0:
                x_0 = x_opt[:self.n_s]
                x_opt = x_opt[self.n_s:, :]

            # get indices of the respective variables
            n_u_0 = self.n_u
            n_u_perf = 0
            if self.n_perf > 1:
                n_u_perf = (self.n_perf - self.r) * self.n_u
            n_k_ff = (self.n_safe - 1) * self.n_u

            c = 0
            idx_u_0 = np.arange(n_u_0)
            c += n_u_0
            idx_u_perf = np.arange(c, c + n_u_perf)
            c += n_u_perf
            idx_k_ff = np.arange(c, c + n_k_ff)
            c += n_k_ff

            u_apply = np.array(cas_reshape(x_opt[idx_u_0], (1, self.n_u)))
            k_ff_perf = np.array(
                cas_reshape(x_opt[idx_u_perf], (self.n_perf - self.r, self.n_u)))

            k_ff_safe = np.array(
                cas_reshape(x_opt[idx_k_ff], (self.n_safe - 1, self.n_u)))
            k_ff_safe_all = np.vstack((u_apply, k_ff_safe))

            k_fb_safe_output = array_of_vec_to_array_of_mat(np.copy(k_fb), self.n_u,
                                                            self.n_s)

            p_safe, q_safe, gp_sigma_pred_safe_all = self.get_safety_trajectory_openloop(x_0, u_apply,
                                                                 np.copy(k_fb),
                                                                 k_ff_safe, q_0, k_fb_0)

            p_safe = np.array(p_safe)
            q_safe = np.array(q_safe)

            if self.verbosity > 1:
                print("=== Safe Trajectory: ===")
                print("Centers:")
                print(p_safe)
                print("Shape matrices:")
                print(q_safe)
                print("Safety controls:")
                print(u_apply)
                print(k_ff_safe)

            k_fb_perf_traj_eval = np.empty((0, self.n_s * self.n_u))
            k_ff_perf_traj_eval = np.empty((0, self.n_u))
            if self.n_safe > 1:
                k_fb_perf_traj_eval = np.vstack(
                    (k_fb_perf_traj_eval, k_fb[:self.r - 1, :]))
                k_ff_perf_traj_eval = np.vstack(
                    (k_ff_perf_traj_eval, k_ff_safe[:self.r - 1, :]))
            if self.n_perf > self.r:
                k_fb_perf_traj_eval = np.vstack((k_fb_perf_traj_eval,
                                                 np.matlib.repmat(k_fb_perf_0,
                                                                  self.n_perf - self.r,
                                                                  1)))
            k_ff_perf_traj_eval = np.vstack((k_ff_perf_traj_eval, k_ff_perf))

            if self.n_perf > 1:
                mu_perf, sigma_perf = self._f_multistep_perf_eval(x_0.squeeze(),
                                                                  u_apply,
                                                                  k_fb_perf_traj_eval,
                                                                  k_ff_perf_traj_eval)

                if self.verbosity > 1:
                    print("=== Performance Trajectory: ===")
                    print("Mu perf:")
                    print(mu_perf)
                    print("Peformance controls:")
                    print(k_ff_perf_traj_eval)

            feasible, _ = self.eval_safety_constraints(p_safe, q_safe)

            if self.rhc and feasible:
                self.k_ff_safe = k_ff_safe
                self.k_ff_perf = k_ff_perf
                self.p_safe = p_safe
                self.k_fb_safe_all = np.copy(k_fb)
                self.u_apply = u_apply
                self.k_fb_perf_0 = k_fb_perf_0

        if feasible:
            self.n_fail = 0

        if not feasible:
            self.n_fail += 1
            q_all = None
            k_fb_safe_output = None
            k_ff_all = None
            p_safe = None
            q_safe = None
            g_res = None

            if self.n_fail >= self.n_safe:
                # Too many infeasible solutions -> switch to safe controller
                if self.verbosity > 1:
                    print(
                        "Infeasible solution. Too many infeasible solutions, switching to safe controller")
                u_apply = self.safe_policy(x_0)
                k_ff_safe_all = u_apply
            else:
                # can apply previous solution
                if self.verbosity > 1:
                    print((
                        "Infeasible solution. Switching to previous solution, n_fail = {}, n_safe = {}".format(
                            self.n_fail, self.n_safe)))
                if sol_verbose:
                    u_apply, k_fb_safe_output, k_ff_safe_all, p_safe = self.get_old_solution(
                        x_0, get_ctrl_traj=True)
                else:
                    u_apply = self.get_old_solution(x_0)
                    k_ff_safe_all = u_apply

        if sol_verbose:
            return x_0, u_apply, feasible, success, k_fb_safe_output, k_ff_safe_all, p_safe, q_safe, sol, gp_sigma_pred_safe_all

        return x_0, u_apply, success

    def eval_safety_constraints(self, p_all, q_all, ubg_term=0., lbg_term=-np.inf,
                                ubg_interm=0., lbg_interm=-np.inf, terminal_only=False,
                                eps_constraints=1e-5):
        """ Evaluate the safety constraints """
        g_term_val = self.g_term_cas(p_all[-1, :, None],
                                     cas_reshape(q_all[-1, :], (self.n_s, self.n_s)))

        feasible_term = np.all(lbg_term - eps_constraints < g_term_val) and np.all(
            g_term_val < ubg_term + eps_constraints)

        feasible = feasible_term
        if terminal_only or self.h_mat_obs is None:

            if self.verbosity > 1:
                print((
                    "\n===== Evaluated terminal constraint values: FEASIBLE = {} =====".format(
                        feasible)))
                print(g_term_val)
                print("\n===== ===== ===== ===== ===== ===== ===== =====")

            return feasible, g_term_val

        g_interm_val = self.g_interm_cas(p_all[-1, :, None], cas_reshape(q_all[-1, :], (
            self.n_s, self.n_s)))

        feasible_interm = np.all(
            lbg_interm - eps_constraints < g_interm_val) and np.all(
            g_interm_val < ubg_interm + eps_constraints)

        feasible = feasible_term and feasible_interm

        return feasible, np.vstack((g_term_val, g_interm_val))

    def get_old_solution(self, x, k=None, get_ctrl_traj=False):
        """ Shift previously obtained solutions in time and return solution to be applied

        Prameters
        ---------
        k: int, optional
            The number of steps to shift back in time. This is number is
            already tracked by the algorithm, so a custom value should be used with caution
        get_ctrl_traj: bool, optional
            Return the safety trajectory state feedback ctrl laws in terms of k_fb,k_ff,p_ctrl

        Returns
        -------
        u_apply: n_s x 0 1darray[float]
            The controls to be applied at the current time step

        if get_ctrl_traj:
            k_fb_safe_traj:
                The feedback ctrls of the remaining safety trajectory
            k_ff_safe_traj:
                The current ff ctrl and the remaining ff ctrls of the safety trajectory
            p_ctrl_safe_traj:
                The



        """
        if self.n_fail > self.n_safe:
            warnings.warn(
                "There are no previous solution to be applied. Returning None")
            return None
        if k is None:
            k = self.n_fail

        if k < 1:
            warnings.warn("Have to shift at least one timestep back")
            return None

        k_fb_old = self.k_fb_safe_all[k - 1]
        k_ff = self.k_ff_safe[k - 1, :, None]
        p_safe = self.p_safe[k - 1, :, None]

        u_apply = feedback_ctrl(x, k_ff, k_fb_old, p_safe)
        if get_ctrl_traj:
            k_fb_safe_traj = None
            k_ff_safe_traj = u_apply
            p_ctrl_safe_traj = None

            if k < self.n_safe:
                k_fb_safe_traj = self.k_fb_safe_all[k:, :]
                # in accordance to the structure current ctrl u_apply is part of the k_ff ctrl trajectory
                k_ff_safe_traj = np.vstack((u_apply, self.k_ff_safe[k + 1:, :]))
                p_ctrl_safe_traj = self.p_safe[k:, :]

            return u_apply, k_fb_safe_traj, \
                   k_ff_safe_traj, p_ctrl_safe_traj

        return u_apply

    def _set_attributes_from_dict(self, attrib_names, default_attribs={},
                                  custom_attrib={}):
        """ Set class attributes from a list of keys,values """
        for attr_name in attrib_names:
            if attr_name in custom_attrib:
                attr_val = custom_attrib[attr_name]
            elif attr_name in default_attribs:
                attr_val = default_attribs[attr_name]
            else:
                raise ValueError(
                    "Neither a custom nor a default value is given for the requried attribute {}".format(
                        attr_name))

            setattr(self, attr_name, attr_val)

    def _set_perf_trajectory(self, name):
        """ Get the peformance trajectory function from identifier"""
        if name == 'mean_equivalent':
            self.perf_trajectory = mean_equivalent_multistep
        elif name == 'taylor':
            self.perf_trajectory = multi_step_taylor_symbolic
        else:
            raise NotImplementedError("Unknown uncertainty propagation method")

    def _get_init_controls(self):
        """ Initialize the controls for the MPC step


        Returns
        -------
        u_0: n_u x 0 np.array[float]
            Initialization of the first (shared) input
        k_ff_safe_new:  (n_safe-1) x n_u np.ndarray[float]
            Initialization of the safety feed-forward control inputs
        k_fb_new: n_u x n_x np.ndarray[float]
            Initialization of the safety feed-back control inputs
        k_ff_perf_new: (n_perf - r_1) x n_u np.ndarray[float]
            Initialization of the performance feed-forward control inputs.
            Not including the shared controls (if r > 1)
        k_fb_perf_0: n_u x n_x np.ndarray[float]
            Initialization of the safety feed-back control inputs

        """

        u_perf_0 = None
        k_fb_perf_0 = None
        k_fb_lqr = self.get_lqr_feedback()

        if self.do_shift_solution and self.n_fail == 0:
            if self.n_safe > 1:
                k_fb_safe = np.copy(self.k_fb_safe_all)

                # Shift the safe controls
                k_ff_safe = np.copy(self.k_ff_safe)

                u_0 = k_ff_safe[0, :]

                if self.n_safe > self.r and self.n_perf > self.n_safe:  # the first control after the shared controls
                    k_ff_perf = np.copy(self.k_ff_perf)
                    k_ff_r_last = (k_ff_perf[0, :] + k_ff_safe[self.r - 1,
                                                     :]) / 2  # mean of first perf ctrl and safe ctrl after shared
                else:
                    k_ff_r_last = k_ff_safe[-1, :]  # just the last safe control

                k_ff_safe_new = np.vstack((k_ff_safe[1:self.r, :], k_ff_r_last))

                if self.n_safe > self.r + 1:
                    k_ff_safe_new = np.vstack((k_ff_safe_new, k_ff_safe[self.r:, :]))
            else:
                u_0 = self.u_apply
                k_ff_safe_new = np.array([])

            if self.n_perf - self.r > 0:
                k_ff_perf = np.copy(self.k_ff_perf)
                k_ff_perf_new = np.vstack((k_ff_perf[1:, :], k_ff_perf[-1, :]))

                if self.perf_has_fb:
                    k_fb_perf_0 = np.copy(self.k_fb_perf_0)
                else:
                    k_fb_perf_0 = np.array([])
            else:
                k_ff_perf_new = np.array([])
                k_fb_perf_0 = np.array([])
        else:
            k_fb_safe = np.empty((self.n_safe - 1, self.n_s * self.n_u))
            for i in range(self.n_safe - 1):
                k_fb_safe[i] = cas_reshape(k_fb_lqr, (1, -1))

            k_ff_safe_new = np.zeros((self.n_safe - 1, self.n_u))
            u_0 = np.zeros((self.n_u, 1))

            k_ff_perf_new = np.array([])
            if self.n_perf > 1:
                k_ff_perf_new = np.zeros((self.n_perf - self.r, self.n_u))

                if self.perf_has_fb:
                    k_fb_perf_0 = k_fb_lqr
            else:
                k_fb_perf_0 = np.array([])

        if self.n_safe > 1:
            k_fb_safe_new = np.vstack((k_fb_safe[1:, :], k_fb_safe[-1, :]))

        else:
            k_fb_safe_new = np.array([])

        return u_0, k_ff_safe_new, k_fb_safe, k_ff_perf_new, k_fb_perf_0

    def update_model(self, x, y, opt_hyp=False, replace_old=True,
                     reinitialize_solver=True):
        """ Update the model of the dynamics

        Parameters
        ----------
        x: n x (n_s+n_u) array[float]
            The raw training input (state,action) pairs
        y: n x (n_s) array[float]
            The raw training targets (observations of next state)
        opt_hyp: bool
            True, if we want to re-optimize the GP hyperparameters
        replace_old: bool
            True, if we replace the current training set of the GP with x,y
        reinitialize_solver:
            True, if we re-initialize the solver (otherwise the MPC will not be updated with the new GP)

        """
        n_train = np.shape(x)[0]
        x_s = x[:, :self.n_s].reshape((n_train, self.n_s))
        x_u = x[:, self.n_s:].reshape((n_train, self.n_u))
        y_prior = self.eval_prior(x_s, x_u)
        x_trafo = mtimes(x_s, self.lin_trafo_gp_input.T)

        x = np.hstack((x_trafo, x_u))

        self.ssm.update_model(x, y - y_prior, opt_hyp, replace_old)
        self.ssm_forward = self.ssm.get_forward_model_casadi(True)

        if reinitialize_solver:
            self.init_solver(self.cost_func)
        else:
            warnings.warn("""Updating gp without reinitializing the solver! \n
                This is potentially dangerous, since the new GP is not incorporated in the MPC""")
