# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:01:18 2017

@author: tkoller
"""

import pytest
import deepdish as dd
import numpy as np
from casadi import Function, SX
from ..gp_models import SimpleGPModel

np.random.seed(125)
a_tol = 1e-6
r_tol = 1e-4

@pytest.fixture(params = [("InvPend",["rbf","rbf"]),("InvPend",["lin_rbf","lin_rbf"]),("InvPend",["lin_mat52","lin_mat52"])])
def before_gp_predict_test(request):
    
    env, kern_types = request.param
    if env == "InvPend":
        path = "invpend_data.npz"
        n_s = 2
        n_u = 1

        
    train_data = np.load(path)
    X = train_data["X"]
    y = train_data["y"]
    m = None
    gp = SimpleGPModel(n_s,n_s,n_u,X,y,m,kern_types,train = True)
    
    return gp,n_s,n_u
    
def test_predict_casadi_symbolic(before_gp_predict_test):
    """ Does symbolic gp prediction yield the same results as numeric eval? """
    
    gp, n_s, n_u = before_gp_predict_test
    
    x_new = SX.sym("x_new",(1,n_s+n_u))
    
    mu_pred, sigm_pred = gp.predict_casadi_symbolic(x_new)
    f_nograd = Function("f_nograd",[x_new],[mu_pred,sigm_pred])
    
    test_input = np.random.randn(n_s+n_u,1)
    out_cas = f_nograd(test_input.T)
    out_numeric = gp.predict(test_input.T)
    
    assert np.allclose(out_cas[0],out_numeric[0],r_tol,a_tol), "Do the predictive means match?"
    assert np.allclose(out_cas[1],out_numeric[1],r_tol,a_tol), "Do the predictive vars match?"
    
    
    
    
    
    
    