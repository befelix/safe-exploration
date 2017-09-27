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
from ..gp_models_utils_casadi import _unscaled_dist
from GPy.kern import RBF

@pytest.fixture(params = ["t_1","t_2","t_3"])
def before_gp_utils_casadi_test_rbf(request):
    """ """
    
    if request.param == "t_1":
        x = np.random.rand(1,3)
        y = np.random.rand(10,3)
        kern_rbf = RBF(3)
    elif request.param == "t_2":
        x = np.random.rand(5,3)
        y = np.random.rand(1,3)
        kern_rbf = RBF(3)
    elif request.param == "t_3":
        x = np.random.rand(10,5)
        y = np.random.rand(20,5)
        kern_rbf = RBF(5)
        
    
    return x,y, kern_rbf
    
    
def test_unscaled_dist(before_gp_utils_casadi_test_rbf):
    """ Does _unscaled_dist show the same behaviour as the GPy implementation"""
    tol = 1e-6
    
    x_inp, y_inp, kern_rbf = before_gp_utils_casadi_test_rbf
    
    
    x = SX.sym("x",np.shape(x_inp))
    y = SX.sym("y",np.shape(y_inp))    
    f = Function("f",[x,y],[_unscaled_dist(x,y)])
    
    f_out_casadi = f(x_inp,y_inp)
    f_out_gpy = kern_rbf._unscaled_dist(x_inp,y_inp)
    
    assert np.all(np.isclose(f_out_casadi,f_out_gpy))
    
    