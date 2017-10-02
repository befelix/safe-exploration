# -*- coding: utf-8 -*-
"""
Test the gp_models_utils_casadi.py module by comparing it to the
corresponding GPy implementation. Most tests will pass if they 
show the same behaviour as the gpy implementation 

@author: tkoller
"""

import pytest
import numpy as np

from casadi import Function, SX
from ..gp_models_utils_casadi import _unscaled_dist, _k_rbf, _k_lin, _k_prod_lin_rbf
from GPy.kern import RBF, Linear

@pytest.fixture(params = [(1,10,3),(5,1,3),(10,20,5),(10,0,5)])
def before_gp_utils_casadi_test_rbf(request):
    """ """
    n_x, n_y, n_dim = request.param
    x = np.random.rand(n_x,n_dim)
    y = None
    if n_y > 0:
        y = np.random.rand(n_y,n_dim)    

    return x,y,n_dim
    
    
def test_unscaled_dist(before_gp_utils_casadi_test_rbf):
    """ Does _unscaled_dist show the same behaviour as the GPy implementation"""
    tol = 1e-6
    x_inp, y_inp,n_dim = before_gp_utils_casadi_test_rbf
    
    ls = np.random.rand(n_dim,) + 1
    rbf_var = np.random.rand()+1
    kern_rbf = RBF(n_dim,rbf_var,ls,True)
    x = SX.sym("x",np.shape(x_inp))
    if y_inp is None:
        f = Function("f",[x],[_unscaled_dist(x,x)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y",np.shape(y_inp))    
        f = Function("f",[x,y],[_unscaled_dist(x,y)])
        f_out_casadi = f(x_inp,y_inp)
    f_out_gpy = kern_rbf._unscaled_dist(x_inp,y_inp)
    
    assert np.all(np.isclose(f_out_casadi,f_out_gpy))
    
    
def test_k_rbf(before_gp_utils_casadi_test_rbf):
    """ Does _k_rbf_ show the same behaviours as the GPy implementation?"""
    
    x_inp, y_inp,n_dim = before_gp_utils_casadi_test_rbf
    ls = np.random.rand(n_dim,) + 1
    rbf_var = np.random.rand()+1
    hyp = dict()
    hyp["rbf_lengthscales"] = ls
    hyp["rbf_variance"] = rbf_var
    kern_rbf = RBF(n_dim,rbf_var,ls,True)
    x = SX.sym("x",np.shape(x_inp))
    
    if y_inp is None:
        f = Function("f",[x],[_k_rbf(x,hyp)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y",np.shape(y_inp))    
        f = Function("f",[x,y],[_k_rbf(x,hyp,y)])
        f_out_casadi = f(x_inp,y_inp)
    
    f_out_gpy = kern_rbf.K(x_inp,y_inp)
    assert np.all(np.isclose(f_out_casadi,f_out_gpy))
    
def test_k_lin(before_gp_utils_casadi_test_rbf):
    """ Does _k_rbf_ show the same behaviours as the GPy implementation?"""
    
    x_inp, y_inp,n_dim = before_gp_utils_casadi_test_rbf
    ls = np.random.rand(n_dim,) + 1
    hyp = dict()
    hyp["lin_variances"] = ls
    kern_lin = Linear(n_dim,ls,True)
    x = SX.sym("x",np.shape(x_inp))
    
    if y_inp is None:
        f = Function("f",[x],[_k_lin(x,hyp)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y",np.shape(y_inp))    
        f = Function("f",[x,y],[_k_lin(x,hyp,y)])
        f_out_casadi = f(x_inp,y_inp)
    
    f_out_gpy = kern_lin.K(x_inp,y_inp)
    assert np.all(np.isclose(f_out_casadi,f_out_gpy))
    
def test_k_prod_lin_rbf(before_gp_utils_casadi_test_rbf):
    """ Does _k_rbf_ show the same behaviours as the GPy implementation?"""
    
    x_inp, y_inp,n_dim = before_gp_utils_casadi_test_rbf
    ls_lin = np.random.rand(n_dim,) + 1
    ls_rbf = np.random.rand(n_dim,) + 1
    rbf_var = np.random.rand()+1
    hyp = dict()
    hyp["lin_variances"] = ls_lin
    hyp["rbf_lengthscales"] = ls_rbf
    hyp["rbf_variance"] = rbf_var
    
    kern_lin = Linear(n_dim,ls_lin,True)*RBF(n_dim,rbf_var,ls_rbf,True)
    
    x = SX.sym("x",np.shape(x_inp))
    
    if y_inp is None:
        f = Function("f",[x],[_k_prod_lin_rbf(x,hyp)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y",np.shape(y_inp))    
        f = Function("f",[x,y],[_k_prod_lin_rbf(x,hyp,y)])
        f_out_casadi = f(x_inp,y_inp)
    
    f_out_gpy = kern_lin.K(x_inp,y_inp)
    assert np.all(np.isclose(f_out_casadi,f_out_gpy))
    
