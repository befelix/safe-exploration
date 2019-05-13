# -*- coding: utf-8 -*-
"""
Implements different exploration oracles which provide informative samples
or exploration objectives to be used in a MPC setting.

Created on Tue Nov 14 10:08:45 2017

@author: tkoller
"""

import casadi as cas
import numpy as np
from casadi import reshape as cas_reshape
from casadi import sum1, sum2, MX, vertcat, mtimes

from .gp_reachability_casadi import lin_ellipsoid_safety_distance
from .gp_reachability_casadi import multi_step_reachability as cas_multistep


class StaticSafeMPCExploration:
    """ Oracle which finds informative samples
    similar with safety constraint and MPC setting.

    In this setting, the sample path doesn't have to be physically connected
    (i.e. we don't have a exploration trajectory but a set of distinct samples)

    Attributes
    ----------
    safempc: SafeMPC
        The underlying safempc which provides necessary information
        such as constraints, prior model and GP

    """

    def __init__(self, safempc, env, n_restarts_optimizer=1, sample_mean=None, sample_std=None, verbosity=1):
        """ Initialize with a pre-defined safempc object"""
        self.safempc = safempc
        self.env = env
        self.n_s = safempc.n_s
        self.n_u = safempc.n_u
        self.T = safempc.n_safe
        self.gp = safempc.ssm
        self.l_mu = safempc.l_mu
        self.l_sigma = safempc.l_sigma
        self.beta_safety = safempc.beta_safety
        self.a = safempc.a
        self.b = safempc.b
        self.lin_trafo_gp_input = safempc.lin_trafo_gp_input
        self.ctrl_bounds = safempc.ctrl_bounds
        self.has_ctrl_bounds = safempc.has_ctrl_bounds
        self.h_mat_safe = safempc.h_mat_safe
        self.h_mat_obs = safempc.h_mat_obs
        self.h_safe = safempc.h_safe
        self.h_obs = safempc.h_obs
        self.m_obs = safempc.m_obs
        self.m_safe = safempc.m_safe
        self.n_restarts_optimizer = n_restarts_optimizer
        self.sample_mean = sample_mean
        self.sample_std = sample_std
        self.verbosity = verbosity
        self.init_solver()

    def init_solver(self, cost_func=None):
        """ Generate the exloration NLP in casadi

        Parameters
        ----------
        T: int, optional
            the safempc horizon

        """

        u_0 = MX.sym("init_control", (self.n_u, 1))
        k_ff_all = MX.sym("feed-forward control", (self.T - 1, self.n_u))
        g = []
        lbg = []
        ubg = []
        g_name = []

        p_0 = MX.sym("initial state", (self.n_s, 1))

        k_fb_safe_ctrl = MX.sym("Feedback term", (self.n_u, self.n_s))
        p_all, q_all, gp_sigma_pred_safe_all = cas_multistep(p_0, u_0,
                                                             k_fb_safe_ctrl, k_ff_all,
                                                             self.gp.get_forward_model_casadi(True), self.l_mu,
                                                             self.l_sigma,
                                                             self.beta_safety, self.a,
                                                             self.b,
                                                             self.lin_trafo_gp_input)

        # generate open_loop trajectory function [vertcat(x_0,u_0)],[f_x])
        self.f_multistep_eval = cas.Function("safe_multistep",
                                             [p_0, u_0, k_fb_safe_ctrl, k_ff_all],
                                             [p_all, q_all])

        g_safe, lbg_safe, ubg_safe, g_names_safe = self.generate_safety_constraints(
            p_all, q_all, u_0, k_fb_safe_ctrl, k_ff_all)
        g = vertcat(g, g_safe)
        lbg += lbg_safe
        ubg += ubg_safe
        g_name += g_names_safe

        if cost_func is None:
            cost = -sum1(sum2(gp_sigma_pred_safe_all))
        else:
            cost = cost_func(p_all, q_all, gp_sigma_pred_safe_all, u_0, k_ff_all)

        opt_vars = vertcat(p_0, u_0, k_ff_all.reshape((-1, 1)))
        opt_params = vertcat(k_fb_safe_ctrl.reshape((-1, 1)))

        prob = {'f': cost, 'x': opt_vars, 'p': opt_params, 'g': g}

        opt = {'error_on_fail': False,
               'ipopt': {'hessian_approximation': 'exact', "max_iter": 120,
                         "expect_infeasible_problem": "no", \
                         'acceptable_tol': 1e-4, "acceptable_constr_viol_tol": 1e-5,
                         "bound_frac": 0.5, "start_with_resto": "no",
                         "required_infeasibility_reduction": 0.85,
                         "acceptable_iter": 8}}  # ipopt
        # opt = {'qpsol':'qpoases','max_iter':120,'hessian_approximation':'exact'}#,"c1":5e-4} #sqpmethod #,'hessian_approximation':'limited-memory'
        # opt = {'max_iter':120,'qpsol':'qpoases'}

        self.solver = cas.nlpsol('solver', 'ipopt', prob, opt)

        self.lbg = lbg
        self.ubg = ubg

    def generate_safety_constraints(self, p_all, q_all, u_0, k_fb_ctrl,
                                    k_ff_all):
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
                k_ff_i = k_ff_all[i, :].reshape((self.n_u, 1))
                k_fb_i = k_fb_ctrl

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

    def find_max_variance(self, x0,
                          sol_verbose=False):
        """ Find the most informative sample in the space constrained by the mpc structure

        Parameters
        ----------
        n_restarts: int, optional
            The number of random initializations of the optimization problem
        ilqr_init: bool, optional
            initialize the state feedback terms with the ilqr feedback law
        sample_mean: n_s x 1 np.ndarray[float], optional
            The mean of the gaussian initial state-action distribution
        sample_var: n_s x n_s np.ndarray[float], optional
            The variance of the gaussian initial state-action distribution

        Returns
        -------
        x_opt:
        u_opt:
        sigm_opt:

        """

        sigma_best = 0
        x_best = None
        u_best = None

        for i in range(self.n_restarts_optimizer):

            x_0 = self.env._sample_start_state(self.sample_mean, self.sample_std)[:,
                  None]  # sample initial state

            if self.T > 1:
                u_0 = self.env.random_action()[:, None]

                k_fb_lqr = self.safempc.get_lqr_feedback()
                k_ff_0 = np.zeros((self.T - 1, self.n_u))
                for j in range(self.T - 1):
                    k_ff_0[j, :] = self.env.random_action()

                params_0 = cas_reshape(k_fb_lqr, (-1, 1))
                vars_0 = np.vstack((x_0, u_0, cas_reshape(k_ff_0, (-1, 1))))
            else:
                u_0 = self.env.random_action()[:, None]
                params_0 = []
                vars_0 = np.vstack((x_0, u_0))

            sol = self.solver(x0=vars_0, p=params_0, lbg=self.lbg, ubg=self.ubg)

            f_opt = sol["f"]
            sigm_i = -float(f_opt)

            if sigm_i > sigma_best:  # check if solution would improve upon current best
                g_sol = np.array(sol["g"]).squeeze()

                if self._is_feasible(g_sol, np.array(self.lbg), np.array(
                        self.ubg)):  # check if solution is feasible
                    w_sol = sol["x"]
                    x_best = np.array(w_sol[:self.n_s])
                    u_best = np.array(w_sol[self.n_s:self.n_s + self.n_u])
                    sigma_best = sigm_i

                    z_i = np.vstack((x_best, u_best)).T
                    if self.verbosity > 0:
                        print(("New optimal sigma found at iteration {}".format(i)))
                        if self.verbosity > 1:
                            print((
                                "New feasible solution with sigma sum {} found".format(
                                    str(sigm_i))))

        return x_best, u_best

    def update_model(self, x, y, train=False, replace_old=False):
        """ Simple wrapper around the update_model function of SafeMPC"""
        self.safempc.update_model(x, y, train, replace_old)
        self.gp = self.safempc.ssm
        self.init_solver()

    def get_information_gain(self):
        return self.safempc.ssm.information_gain()

    def _is_feasible(self, g, lbg, ubg, feas_tol=1e-7):
        """ """
        return np.all(g > lbg - feas_tol) and np.all(g < ubg + feas_tol)


class DynamicSafeMPCExploration:
    """ """

    def __init__(self, safempc, env):
        """ Initialize with a pre-defined safempc object"""
        self.safempc = safempc
        self.env = env
        self.n_s = safempc.n_s
        self.n_u = safempc.n_u
        self.n_safe = safempc.n_safe
        self.n_perf = safempc.n_perf

        # 
        cost = None
        self.safempc.init_solver(cost)

    def find_max_variance(self, x_0, sol_verbose=False):
        if sol_verbose:
            u_apply, feasible, _, k_fb, k_ff, p_ctrl, q_all, _ = self.safempc.get_action(
                x_0, sol_verbose=True)

            return x_0[:,
                   None], u_apply, feasible, k_fb, k_ff, p_ctrl, q_all
        else:
            u_apply, _ = self.safempc.get_action(x_0)
            return x_0[:, None], u_apply[:, None]

    def update_model(self, x, y, train=False, replace_old=False):
        """ Simple wrapper around the update_model function of SafeMPC"""
        self.safempc.update_model(x, y, train, replace_old)

    def get_information_gain(self):
        return self.safempc.ssm.information_gain()
