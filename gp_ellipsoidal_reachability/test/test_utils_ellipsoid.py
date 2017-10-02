# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:22:17 2017

@author: tkoller
"""
import pytest
from ..utils_ellipsoid import distance_to_center, ellipsoid_from_rectangle 
import numpy as np

@pytest.fixture(params = ["rectangle","cube"])
def before_ellipsoid_from_rectangle(request):
    
    if request.param == "rectangle":
        n_s = 3
        ub = [0.1,0.3,0.5]
        test_points = np.array([[-0.1,-0.3,0.5],[-0.1,0.3,-0.5],[0.1,0.3,0.5]])

    else:
        n_s = 3
        ub = [0.1]*n_s
        test_points = np.array([[-0.1,0.1,0.1],[-0.1,-0.1,-0.1],[0.1,0.1,0.1]])
    test_data = {"ub":ub,"n_s":n_s,"test_points":test_points}
    return test_data
    
def test_ellipsoid_from_rectangle_ub_below_zero_throws_exception():
    """ do we get an exception if lb > ub """
        
    with pytest.raises(Exception):
        ub = [0.6,-0.3]
        q_shape = ellipsoid_from_rectangle(ub)
    
def test_ellipsoid_from_from_rectangle_shape_matrix_spd(before_ellipsoid_from_rectangle):
    """ Is the resulting shape matrix symmetric postive definite?"""
    
    ub = before_ellipsoid_from_rectangle["ub"]
    n_s = before_ellipsoid_from_rectangle["n_s"]
    
    q_shape = ellipsoid_from_rectangle(ub)
    
    assert np.all(np.linalg.eigvals(q_shape) > 0)
    assert np.allclose(0.5*(q_shape+q_shape.T),q_shape)
    
def test_ellipsoid_from_from_rectangle_residuals_zero_(before_ellipsoid_from_rectangle):
    """ Are the residuals of the exact algebraic fit zero at the edges of the rectangle? """
    eps_tol = 1e-5    
    
    ub = before_ellipsoid_from_rectangle["ub"]
    n_s = before_ellipsoid_from_rectangle["n_s"]  
    
    q_shape = ellipsoid_from_rectangle(ub)

    p_center = np.zeros((n_s,1))
    
    test_points = before_ellipsoid_from_rectangle["test_points"]    
    
    
    d_test_points = distance_to_center(test_points,p_center,q_shape)
    
    assert np.all(np.abs(d_test_points-1) <= eps_tol)