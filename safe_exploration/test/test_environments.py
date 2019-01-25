# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:39 2017

@author: tkoller
"""
import numpy as np
import pytest
from scipy.optimize import approx_fprime

from ..environments import InvertedPendulum, CartPole
from ..utils import sample_inside_polytope

np.random.seed(0)

@pytest.fixture(params=[InvertedPendulum(),CartPole()])
def before_test_inv_pend(request):
    env = request.param

    return env

def test_normalization(before_test_inv_pend):
    """ """
    env = before_test_inv_pend
    state = np.random.rand(env.n_s)
    action = np.random.rand(env.n_u)
    s_1,a_1 =  env.normalize(*env.unnormalize(state,action))
    s_2,a_2 =  env.unnormalize(*env.normalize(state,action))

    assert np.all(s_1==state)
    assert np.all(a_1==action)
    assert np.all(s_2==state)
    assert np.all(a_2==action)

def test_safety_bounds_normalization(before_test_inv_pend):
    """ """

    env = before_test_inv_pend

    n_samples = 50
    x = np.random.randn(n_samples,env.n_s)
    h_mat_safe, h_safe,_,_ = env.get_safety_constraints(normalize = False)
    in_unnorm = sample_inside_polytope(x,h_mat_safe,h_safe)

    x_norm ,_ = env.normalize(x)
    h_mat_safe_norm, h_safe_norm,_,_ = env.get_safety_constraints(normalize = True)
    in_norm = sample_inside_polytope(x_norm,h_mat_safe_norm,h_safe_norm)

    assert np.all(in_unnorm == in_norm), "do the normalized constraint correspond to the unnormalized?"



def test_gradients(before_test_inv_pend):

    env = before_test_inv_pend
    n_s = env.n_s
    n_u = env.n_u

    for i in range(n_s):
        f = lambda z: env._dynamics(0,z[:env.n_s],z[env.n_s:])[i]
        f_grad = env._jac_dynamics()[i,:]
        grad_finite_diff = approx_fprime(np.zeros((n_s+n_u,)),f,1e-8)

        #err = check_grad(f,f_grad,np.zeros((n_s+n_u,)))

        assert np.allclose(f_grad,grad_finite_diff), 'Is the gradient of the {}-th dynamics dimension correct?'.format(i)
