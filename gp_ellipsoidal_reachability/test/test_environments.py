# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:39 2017

@author: tkoller
"""
import numpy as np
from ..environments import InvertedPendulum
from ..utils import sample_inside_polytope

def before_test_inv_pend():
    env = InvertedPendulum()
    
    return env

def test_normalization():
    """ """
    env = before_test_inv_pend()
    state = np.random.rand(2)
    action = np.random.rand(1)
    s_1,a_1 =  env.normalize(*env.unnormalize(state,action))
    s_2,a_2 =  env.unnormalize(*env.normalize(state,action))
    assert np.all(s_1==state) 
    assert np.all(a_1==action)
    assert np.all(s_2==state)
    assert np.all(a_2==action)
    
def test_linearization():
    """ """
    
def test_safety_bounds_normalization():
    """ """
    
    env = before_test_inv_pend()
    
    n_samples = 50
    x = np.random.randn(n_samples,env.n_s)
    h_mat_safe, h_safe,_,_ = env.get_safety_constraints(normalize = False)
    in_unnorm = sample_inside_polytope(x,h_mat_safe,h_safe)    
    
    x_norm ,_ = env.normalize(x)
    h_mat_safe_norm, h_safe_norm,_,_ = env.get_safety_constraints(normalize = True)
    in_norm = sample_inside_polytope(x_norm,h_mat_safe_norm,h_safe_norm)
    
    assert np.all(in_unnorm == in_norm), "do the normalized constraint correspond to the unnormalized?"