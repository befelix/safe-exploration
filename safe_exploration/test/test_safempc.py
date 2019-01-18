# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:03:39 2017

@author: tkoller
"""
import os.path

import pytest
from ..safempc_simple import SimpleSafeMPC
from ..environments import CartPole
from .. import gp_models
from ..gp_reachability import multistep_reachability, lin_ellipsoid_safety_distance,multistep_reachability_new
import deepdish as dd
import numpy as np
from ..utils import array_of_vec_to_array_of_mat

a_tol = 1e-5
r_tol = 1e-4

@pytest.fixture(params = [("CartPole",True)])
def before_test_safempc(request):
    np.random.seed(125)
    env,lin_model = request.param
    if env == "CartPole":
        env = CartPole()
        n_s = env.n_s
        n_u = env.n_u
        path = os.path.join(os.path.dirname(__file__), "data_cartpole.npz")
        c_safety = 0.5

    a = None
    b= None
    lin_model_param = None
    if lin_model:
        a,b = env.linearize_discretize()
        lin_model_param = (a,b)

    train_data = dict(np.load(path).items())
    X = train_data["X"]
    X = X[:80,:]
    y = train_data["y"]
    y = y[:80,:]


    m = None
    gp = gp_models.SimpleGPModel(n_s,n_s,n_u,X,y,m)
    gp.train(X,y,m,True,None,False)

    n_safe = 3
    n_perf = None

    l_mu = np.array([0.01]*n_s)
    l_sigma = np.array([0.01]*n_s)

    h_mat_safe = np.hstack((np.eye(n_s,1),-np.eye(n_s,1))).T
    h_safe = np.array([5,5]).reshape((2,1))
    h_mat_obs = np.copy(h_mat_safe)
    h_obs = np.array([5,5]).reshape((2,1))

    wx_cost = 10*np.eye(n_s)
    wu_cost = 0.1

    dt = 0.1
    ctrl_bounds = np.hstack((np.reshape(-1,(-1,1)),np.reshape(1,(-1,1))))

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


    safe_mpc_solver = SimpleSafeMPC(n_safe,gp,env_opts_safempc,wx_cost,wu_cost,beta_safety = c_safety)

    safe_mpc_solver.init_solver()

    x_0 = np.zeros((n_s,1))

    return env, safe_mpc_solver,x_0, c_safety, l_mu, l_sigma, gp,a,b,n_safe


@pytest.mark.xfail
def test_safempc_open_loop_trajectory_same_as_planned(before_test_safempc):
    """ check if casadi mpc constr values are the same as from numpy reachability results

    TODO: A rather circuituous way to check if the internal function evaluations
    are the same as when applying the resulting controls to the reachability functions.
    Source of failure could be: bad reshaping operations resulting in different controls
    per time step. If the reachability functions itself are identical is tested
    in test_gp_reachability_casadi.py

    """
    env, safe_mpc,x_0,c_safety,l_mu,l_sigma,gp,a,b,n_safe = before_test_safempc
    h_mat_safe = safe_mpc.h_mat_safe
    h_safe = safe_mpc.h_safe
    h_mat_obs = safe_mpc.h_mat_obs
    h_obs = safe_mpc.h_obs

    _,_,_, k_fb_apply, k_ff_all, p_all_planner,q_all_planner, constr_values = safe_mpc.solve(x_0,sol_verbose = True)

    if k_fb_apply.ndim == 2:
        k_fb_apply = array_of_vec_to_array_of_mat(k_fb_apply,env.n_u,env.n_s)
        #k_fb_tmp = np.copy(k_fb_apply)
        #n_u,n_s = np.shape(k_fb_tmp)
        #k_fb_apply = np.empty((n_safe-1,n_u,n_s))
        #for i in range(n_safe-1):
        #    k_fb_apply[i] = k_fb_tmp




    p_new,q_new,p_all_ms,q_all_ms = multistep_reachability(x_0,gp,k_fb_apply,k_ff_all,l_mu,l_sigma,None, c_safety,0,a,b)

    n_s = np.shape(p_all_ms)[1]

    assert np.allclose(p_all_ms[-1,:],p_all_planner[-1,:]), "Are the centers of the last ellipsoids the same?"
    assert np.allclose(p_all_ms[0,:],p_all_planner[0,:]), "Are the centers of the first ellipsoids the same?"
    assert np.allclose(q_all_ms[-1,:,:],q_all_planner[-1,:,:]), "Are the shape matrices of the last ellipsoids the same?"
    assert np.allclose(q_all_ms[0,:,:],q_all_planner[0,:,:]), "Are the shape matrices of the first ellipsoids the same?"


@pytest.mark.xfail
def test_mpc_casadi_same_constraint_values_as_numeric_eval(before_test_safempc):
    """check if the returned open loop (numerical) ellipsoids are the same as in internal planning"""

    env, safe_mpc,x_0,c_safety,l_mu,l_sigma,gp,a,b,n_safe = before_test_safempc

    _,_,_, k_fb_apply, k_ff_apply, p_all,q_all, constr_values= safe_mpc.solve(x_0,sol_verbose = True)


    p_all,q_all = safe_mpc.get_trajectory_openloop(x_0.squeeze(),k_fb_apply,k_ff_apply)

    h_mat_safe = safe_mpc.h_mat_safe
    h_safe = safe_mpc.h_safe
    h_mat_obs = safe_mpc.h_mat_obs
    h_obs = safe_mpc.h_obs
    m_obs,n_s = np.shape(h_mat_obs)
    m_safe, _ = np.shape(h_mat_safe)

    g_0 = lin_ellipsoid_safety_distance(p_all[0,:].reshape(n_s,1),q_all[0,:].reshape(n_s,n_s),h_mat_obs,h_obs)
    g_1 = lin_ellipsoid_safety_distance(p_all[1,:].reshape(n_s,1),q_all[1,:].reshape(n_s,n_s),h_mat_obs,h_obs)
    g_safe = lin_ellipsoid_safety_distance(p_all[1,:].reshape(n_s,1),q_all[1,:].reshape(n_s,n_s),h_mat_safe,h_safe)


    idx_state_constraints = env.n_u*n_safe*2-1

    assert np.allclose(g_1,constr_values[idx_state_constraints+m_obs:idx_state_constraints+2*m_obs],r_tol,a_tol), "Are the distances to the obstacle the same after two steps?"
    assert np.allclose(g_safe,constr_values[-m_safe:],r_tol,a_tol), "Are the distances to the obstacle the same after two steps?"
    assert np.allclose(g_0,constr_values[idx_state_constraints:idx_state_constraints+m_obs],r_tol,a_tol), "Are the distances to the obstacle the same after one step?"


