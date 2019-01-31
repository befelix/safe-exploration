# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:03:39 2017

@author: tkoller
"""
import os.path

import numpy as np
import pytest
from casadi import reshape as cas_reshape
from numpy.testing import assert_allclose

from safe_exploration.ssm_gpy.gp_models_old import SimpleGPModel
from ..environments import CartPole
from ..gp_reachability import lin_ellipsoid_safety_distance
from ..safempc_simple import SimpleSafeMPC

a_tol = 1e-5
r_tol = 1e-4


@pytest.fixture(params=[("CartPole", True)])
def before_test_safempc(request):
    np.random.seed(12345)
    env, lin_model = request.param
    if env == "CartPole":
        env = CartPole()
        n_s = env.n_s
        n_u = env.n_u
        path = os.path.join(os.path.dirname(__file__), "data_cartpole.npz")
        c_safety = 0.5

    a = None
    b = None
    lin_model_param = None
    if lin_model:
        a, b = env.linearize_discretize()
        lin_model_param = (a, b)

    train_data = dict(list(np.load(path).items()))
    X = train_data["X"]
    X = X[:80, :]
    y = train_data["y"]
    y = y[:80, :]

    m = None
    gp = SimpleGPModel(n_s, n_s, n_u, X, y, m)
    gp.train(X, y, m, opt_hyp=False, choose_data=False)

    n_safe = 3
    n_perf = None

    l_mu = np.array([0.01] * n_s)
    l_sigma = np.array([0.01] * n_s)

    h_mat_safe = np.hstack((np.eye(n_s, 1), -np.eye(n_s, 1))).T
    h_safe = np.array([300, 300]).reshape((2, 1))
    h_mat_obs = np.copy(h_mat_safe)
    h_obs = np.array([300, 300]).reshape((2, 1))

    wx_cost = 10 * np.eye(n_s)
    wu_cost = 0.1

    dt = 0.1
    ctrl_bounds = np.hstack((np.reshape(-1, (-1, 1)), np.reshape(1, (-1, 1))))

    safe_policy = None
    env_opts_safempc = dict()
    env_opts_safempc["h_mat_safe"] = h_mat_safe
    env_opts_safempc["h_safe"] = h_safe
    env_opts_safempc["lin_model"] = lin_model
    env_opts_safempc["ctrl_bounds"] = ctrl_bounds
    env_opts_safempc["safe_policy"] = safe_policy
    env_opts_safempc["h_mat_obs"] = h_mat_obs
    env_opts_safempc["h_obs"] = h_obs
    env_opts_safempc["dt"] = dt
    env_opts_safempc["lin_trafo_gp_input"] = None
    env_opts_safempc["l_mu"] = l_mu
    env_opts_safempc["l_sigma"] = l_sigma
    env_opts_safempc["lin_model"] = lin_model_param

    safe_mpc = SimpleSafeMPC(n_safe, gp, env_opts_safempc, wx_cost, wu_cost,
                             beta_safety=c_safety)

    safe_mpc.init_solver()

    x_0 = 0.2 * np.random.randn(n_s, 1)

    _, _, _, k_fb_apply, k_ff_apply, p_all, q_all, sol = safe_mpc.solve(x_0,
                                                                        sol_verbose=True)

    return env, safe_mpc, None, None, k_fb_apply, k_ff_apply, p_all, q_all, sol


@pytest.mark.skip(reason="Not implemented yet")
def test_mpc_casadi_same_objective_value_values_as_numeric_eval(before_test_safempc):
    """ check if casadi mpc constr values are the same as from numpy reachability results

    TODO: A rather circuituous way to check if the internal function evaluations
    are the same as when applying the resulting controls to the reachability functions.
    Source of failure could be: bad reshaping operations resulting in different controls
    per time step. If the reachability functions itself are identical is tested
    in test_gp_reachability_casadi.py

    """
    env, safe_mpc, k_ff_perf_traj, k_fb_perf_traj, k_fb_apply, k_ff_apply, p_all, q_all, sol = before_test_safempc

    h_mat_safe = safe_mpc.h_mat_safe
    h_safe = safe_mpc.h_safe
    h_mat_obs = safe_mpc.h_mat_obs
    h_obs = safe_mpc.h_obs

    _, _, _, k_fb_apply, k_ff_all, p_all_planner, q_all_planner, sol = safe_mpc.solve(
        x_0, sol_verbose=True)


def test_mpc_casadi_same_constraint_values_as_numeric_eval(before_test_safempc):
    """check if the returned open loop (numerical) ellipsoids are the same as in internal planning"""

    env, safe_mpc, k_ff_perf_traj, k_fb_perf_traj, k_fb_apply, k_ff_apply, p_all, q_all, sol = before_test_safempc

    n_s = env.n_s
    n_u = env.n_u
    g_0 = lin_ellipsoid_safety_distance(p_all[0, :].reshape(n_s, 1),
                                        q_all[0, :].reshape(n_s, n_s),
                                        safe_mpc.h_mat_obs, safe_mpc.h_obs)
    g_1 = lin_ellipsoid_safety_distance(p_all[1, :].reshape(n_s, 1),
                                        cas_reshape(q_all[1, :], (n_s, n_s)),
                                        safe_mpc.h_mat_obs, safe_mpc.h_obs)
    g_safe = lin_ellipsoid_safety_distance(p_all[-1, :].reshape(n_s, 1),
                                           q_all[-1, :].reshape(n_s, n_s),
                                           safe_mpc.h_mat_safe, safe_mpc.h_safe)

    idx_state_constraints = env.n_u * safe_mpc.n_safe * 2 - 1

    constr_values = sol["g"]

    assert_allclose(g_0, constr_values[
                         idx_state_constraints:idx_state_constraints + safe_mpc.m_obs],
                    r_tol, a_tol)
    # Are the distances to the obstacle the same after two steps?
    assert_allclose(g_1, constr_values[
                         idx_state_constraints + safe_mpc.m_obs:idx_state_constraints + 2 * safe_mpc.m_obs],
                    r_tol, a_tol)
    # Are the distances to the safe set the same after the last step?
    assert_allclose(g_safe, constr_values[
                            idx_state_constraints + 2 * safe_mpc.m_obs:idx_state_constraints + 2 * safe_mpc.m_obs + safe_mpc.m_safe],
                    r_tol, a_tol)
