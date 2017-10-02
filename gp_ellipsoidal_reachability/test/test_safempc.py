# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:03:39 2017

@author: tkoller
"""
import pytest
from ..safempc import SafeMPC
from .. import gp_models
from ..gp_reachability import multistep_reachability, lin_ellipsoid_safety_distance
import deepdish as dd
import numpy as np

@pytest.fixture(params = ["CartPole"])
def before_test_safempc(request):
    
    env = request.param
    if env == "CartPole":
        n_s = 5
        n_u = 1
        path = "data_CartPole_N511.hd5"
        c_safety = 11.07
        
    train_data = dd.io.load(path)
    X = train_data["X"]
    y = train_data["y"]
    m = 100
    gp = gp_models.SimpleGPModel(X,y,m)
    
    n_safe = 2 
    n_perf = None
    
    l_mu = np.array([0.01]*n_s)
    l_sigma = np.array([0.01]*n_s)
    
    h_mat_safe = np.hstack((np.eye(n_s,1),-np.eye(n_s,1))).T
    h_safe = np.array([0.5,0.5]).reshape((2,1))
    h_mat_obs = np.copy(h_mat_safe)
    h_obs = np.array([0.5,0.5]).reshape((2,1))
    
    wx_cost = 10*np.eye(n_s)
    wu_cost = 0.1
    
    safe_mpc_solver = SafeMPC(n_safe,n_perf,gp,l_mu,l_sigma,h_mat_safe,h_safe,
                              h_mat_obs,h_obs,wx_cost,wu_cost)
                              
    safe_mpc_solver.init_solver()
    
    x_0 = np.zeros((n_s,1))
    x_target = np.eye(n_s,1)
    
    return safe_mpc_solver,x_0,x_target, c_safety, l_mu, l_sigma, gp
    
    
def test_mpc_casadi_same_constraint_values_as_numeric_eval(before_test_safempc):
    """ check if casadi mpc constr values are the same as from numpy reachability results 
    
    TODO: A rather circuituous way to check if the internal function evaluations
    are the same as when applying the resulting controls to the reachability functions.
    Source of failure could be: bad reshaping operations resulting in different controls
    per time step. If the reachability functions itself are identical are tested 
    in test_gp_reachability_casadi.py
    
    """
    safe_mpc,x_0,x_target,c_safety,l_mu,l_sigma,gp = before_test_safempc
    h_mat_safe = safe_mpc.h_mat_safe
    h_safe = safe_mpc.h_safe
    h_mat_obs = safe_mpc.h_mat_obs
    h_obs = safe_mpc.h_obs
    
    k_fb_apply, k_ff_apply, k_fb_all, k_ff_all, constr_values = safe_mpc.solve(x_0,x_target,sol_verbose = True)
    
    p_new,q_new,p_all,q_all = multistep_reachability(x_0,gp,k_fb_all,k_ff_all,l_mu,l_sigma,None, c_safety,0)
    
    m_obs,n_s = np.shape(h_mat_obs)
    
    g_0 = lin_ellipsoid_safety_distance(p_all[0,:].reshape(n_s,1),q_all[0,:].reshape(n_s,n_s),h_mat_obs,h_obs)
    g_1 = lin_ellipsoid_safety_distance(p_all[1,:].reshape(n_s,1),q_all[1,:].reshape(n_s,n_s),h_mat_obs,h_obs)
    g_safe = lin_ellipsoid_safety_distance(p_all[1,:].reshape(n_s,1),q_all[1,:].reshape(n_s,n_s),h_mat_safe,h_safe)

    assert np.allclose(g_0,constr_values[:m_obs]), "Are the distances to the obstacle the same after one step?"
    assert np.allclose(g_1,constr_values[m_obs:2*m_obs]), "Are the distances to the obstacle the same after two steps?"
    assert np.allclose(g_safe,constr_values[2*m_obs:]), "Are the distances to the obstacle the same after two steps?"
    