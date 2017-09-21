# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:22:17 2017

@author: tkoller
"""
import pytest
from ..utils_ellipsoid import ellipsoid_from_box, distance_to_center 
import numpy as np

@pytest.fixture(params = ["full","diag_only","cube"])
def before_ellipsoid_from_box(request):
    
    if request.param == "full":
        n_s = 3
        lb = [-0.2,-0.3,0.4]
        ub = [0.1,0.3,0.5]
        test_points = np.array([[-0.2,-0.3,0.4],[-0.2,0.3,0.4],[-0.2,0.3,0.5]])
        diag_only = False
    elif request.param == "diag_only":
        n_s = 3
        lb = [-0.2,-0.3,0.4]
        ub = [0.1,0.3,0.5]
        test_points = np.array([[-0.2,-0.3,0.4],[-0.2,0.3,0.4],[-0.2,0.3,0.5]])
        diag_only = True
    else:
        n_s = 3
        lb = [-0.1]*n_s
        ub = [0.1]*n_s
        test_points = np.array([[-0.1,0.1,0.1],[-0.1,-0.1,-0.1],[0.1,0.1,0.1]])
        diag_only = True
        
    
    
    test_data = {"lb":lb,"ub":ub,"n_s":n_s,"test_points":test_points,"diag_only":diag_only}
    return test_data
    
def test_ellipsoid_from_box_ub_smaller_lb_throws_exception():
    """ do we get an exception if lb > ub """
        
    with pytest.raises(Exception):
        lb = [0.5,0.7]
        ub = [0.6,0.3]
        q_shape = ellipsoid_from_box(lb,ub)
    
def test_ellipsoid_from_box_shape_matrix_spd(before_ellipsoid_from_box):
    """ Is the resulting shape matrix symmetric postive definite?"""
    
    lb = before_ellipsoid_from_box["lb"]
    ub = before_ellipsoid_from_box["ub"]
    n_s = before_ellipsoid_from_box["n_s"]
    diag_only = before_ellipsoid_from_box["diag_only"]  
    
    q_shape = ellipsoid_from_box(lb,ub,diag_only=diag_only)
    
    assert np.all(np.linalg.eigvals(q_shape) > 0)
    assert np.allclose(0.5*(q_shape+q_shape.T),q_shape)
    
def test_ellipsoid_from_box_residuals_zero_(before_ellipsoid_from_box):
    """ Are the residuals of the exact algebraic fit zero at the edges of the rectangle? """
    eps_tol = 1e-5    
    
    lb = before_ellipsoid_from_box["lb"]
    ub = before_ellipsoid_from_box["ub"]
    n_s = before_ellipsoid_from_box["n_s"]  
    diag_only = before_ellipsoid_from_box["diag_only"]
    
    q_shape = ellipsoid_from_box(lb,ub,diag_only=diag_only)

    p_center = np.zeros((n_s,1))
    
    test_points = before_ellipsoid_from_box["test_points"]    
    
    
    d_test_points = distance_to_center(test_points,p_center,q_shape)
    
    assert np.all(np.abs(d_test_points-1) <= eps_tol)