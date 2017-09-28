# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:12:30 2017

@author: tkoller
"""
import pytest
import numpy as np
import deepdish as dd

from .. import gp_models
from .. import gp_reachability_casadi as reach_cas
from .. import gp_reachability as reach_num
from casadi import SX, Function

@pytest.fixture(params = [("CartPole",True),("CartPole",False)])
def before_test_onestep_reachability(request):
    
    env, init_uncertainty = request.param
    if env == "CartPole":
        n_s = 5
        n_u = 1
        path = "data_CartPole_N511.hd5"
        c_safety = 11.07
        
    train_data = dd.io.load(path)
    X = train_data["X"]
    y = train_data["y"]
    m = 50
    gp = gp_models.SimpleGPModel(X,y,m)
    L_mu = np.array([0.01]*n_s)
    L_sigm = np.array([0.05]*n_s)
    k_fb = np.random.rand(n_u,n_s) # need to choose this appropriately later
    k_ff = np.random.rand(n_u,1)
    
    p = np.random.randn(n_s,1)
    if init_uncertainty:
        q = .1 * np.eye(n_s) # reachability based on previous uncertainty 
    else:
        q = None # no initial uncertainty
    
    return p,q,gp,k_fb,k_ff,L_mu,L_sigm,c_safety
    
    
def test_onestep_reachability(before_test_onestep_reachability):
    """ do we get the same results as the numpy equivalent?"""
    
    p,q,gp,k_fb,k_ff,L_mu,L_sigm,c_safety = before_test_onestep_reachability
    
    n_u,n_s = np.shape(k_fb)
    
    k_fb_cas = SX.sym("k_fb",(n_u,n_s))
    k_ff_cas = SX.sym("k_ff",(n_u,1))
    
    p_new_cas, q_new_cas = reach_cas.onestep_reachability(p,gp,k_fb_cas,k_ff_cas,L_mu,L_sigm,q,c_safety,0)
    f = Function("f",[k_fb_cas,k_ff_cas],[p_new_cas,q_new_cas])
    
    f_out_cas = f(k_fb,k_ff)
    
    f_out_num = reach_num.onestep_reachability(p,gp,k_fb,k_ff,L_mu,L_sigm,q,c_safety,0)
    
    assert np.allclose(f_out_cas[0],f_out_num[0]), "Are the centers of the next state the same?"
    assert np.allclose(f_out_cas[1],f_out_num[1]), "Are the sahpe matrices of the next state the same?"
    
    
    