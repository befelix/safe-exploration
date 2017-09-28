# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:21:10 2017

@author: tkoller
"""

from .. import utils_casadi as utils_cas
from .. import utils
import numpy as np
import pytest

from casadi import Function, SX


@pytest.fixture(params = ["t_1","t_2","t_3","t_4"])
def before_test_matrix_norm_2(request):
    if request.param == "t_1":
        n_s = 2
    elif request.param == "t_2":
        n_s = 3
    elif request.param == "t_3":
        n_s = 5   
    elif request.param == "t_4":
        n_s = 8   
    x_0 = np.random.rand(n_s,n_s)
    return np.dot(x_0,x_0.T), n_s
    
    
def test_matrix_norm_2(before_test_matrix_norm_2):
    """ Does the casadi norm2 return the same as numpy?"""
    
    #create symmetric matrix
    
    x_in, n_s = before_test_matrix_norm_2
    
    x_cas = SX.sym("x",(n_s,n_s))
    f = Function("f",[x_cas],[utils_cas.matrix_norm_2(x_cas)])
    
    f_out = f(x_in)
    print(f_out)
    assert np.allclose(f_out,np.max(np.linalg.eig(x_in)[0])), "Doesn't return the same as numpy eig"
    
    
@pytest.fixture(params = ["t_1","t_2","t_3","t_4"])
def before_test_vec_max(request):
    if request.param == "t_1":
        n_s = 2
    elif request.param == "t_2":
        n_s = 3
    elif request.param == "t_3":
        n_s = 5   
    elif request.param == "t_4":
        n_s = 8   
    return np.random.rand(n_s,), n_s
    
    
def test_vec_max(before_test_vec_max):
    """ Does casadi max return the same as numpy max? """
    x_in, n_s = before_test_vec_max
    x_cas = SX.sym("x",(n_s,1))
    f = Function("f",[x_cas],[utils_cas.vec_max(x_cas)])
    f_out = f(x_in)
    assert np.max(x_in) == f_out