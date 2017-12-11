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
    
    assert np.allclose(f_out,np.max(np.linalg.eig(x_in)[0])), "Doesn't return the same as numpy eig"


@pytest.fixture(params = ["t_1","t_2","t_3","t_4"])
def before_test_matrix_norm_2_generalized(request):
    if request.param == "t_1":
        n_s = 2
    elif request.param == "t_2":
        n_s = 3
    elif request.param == "t_3":
        n_s = 5   
    elif request.param == "t_4":
        n_s = 8   
    
    a_0 = np.random.rand(n_s,n_s)
    a = np.dot(a_0,a_0.T)
    inv_b_0 = np.random.rand(n_s,n_s)
    inv_b = np.dot(inv_b_0,inv_b_0.T)
    
    return a, inv_b , n_s
    
    
def test_matrix_norm_2_generalized(before_test_matrix_norm_2_generalized):
    """ Do we get the same results as numpy """
    a,inv_b,n_s = before_test_matrix_norm_2_generalized
    
    a_cas = SX.sym("a",(n_s,n_s))
    inv_b_cas = SX.sym("inv_b",(n_s,n_s))
    
    f = Function("f",[a_cas,inv_b_cas],[utils_cas.matrix_norm_2_generalized(a_cas,inv_b_cas)])
    f_out = f(a,inv_b)
    
    assert np.allclose(f_out,np.max(np.linalg.eig(np.dot(inv_b,a))[0])), \
        """ Do we get the largest eigenvalue of the generalized eigenvalue problem
            a*x = \lambda*inv_b*x ?
        """
        
   
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
    
@pytest.fixture(params = ["t_1","t_2","t_3","t_4"])
def before_test_remainder_overapproximation(request):
    if request.param == "t_1":
        n_s = 2
        n_u = 1
    elif request.param == "t_2":
        n_s = 3
        n_u = 2
    elif request.param == "t_3":
        n_s = 5 
        n_u = 4
    elif request.param == "t_4":
        n_s = 8   
        n_u = 3
    x_0 = np.random.rand(n_s,n_s)
    q_num = np.dot(x_0,x_0.T) + 0.1*np.eye(n_s) # guarantees s.p.d.
    k_fb_num = np.random.randn(n_u,n_s)
    
    l_mu = np.array([.1]*n_s)
    l_sigma = np.array([.1]*n_s)
    return q_num, k_fb_num, n_s,n_u, l_mu, l_sigma
    
    
def test_remainder_overapproximation(before_test_remainder_overapproximation):
    """ Do we get the same results for python and casadi? """
    q_num, k_fb_num, n_s, n_u, l_mu, l_sigma = before_test_remainder_overapproximation
    
    q_cas = SX.sym("q",(n_s,n_s))
    ## test with numeric k_fb    
    u_mu_cas, u_sigma_cas = utils_cas.compute_remainder_overapproximations(q_cas,k_fb_num,l_mu,l_sigma)
    f = Function("f",[q_cas],[u_mu_cas,u_sigma_cas])
    f_out_cas = f(q_num)
    f_out_py = utils.compute_remainder_overapproximations(q_num,k_fb_num,l_mu,l_sigma)
    
    assert np.allclose(f_out_cas[0],f_out_py[0]), "are the overapproximations of mu the same"
    assert np.allclose(f_out_cas[1],f_out_py[1]), "are the overapproximations of sigma the same"
        
    ## test with symbolic k_fb
    k_fb_cas = SX.sym("k_fb",(n_u,n_s))
    
    u_mu_cas_sym_kfb, u_sigma_cas_sym_kfb = utils_cas.compute_remainder_overapproximations(q_cas,k_fb_cas,l_mu,l_sigma)
    f_sym_k = Function("f",[q_cas,k_fb_cas],[u_mu_cas_sym_kfb, u_sigma_cas_sym_kfb])
    
    f_out_cas_sym_k = f_sym_k(q_num,k_fb_num)
    
    assert np.allclose(f_out_cas_sym_k[0],f_out_py[0]), "are the overapproximations of mu the same"
    assert np.allclose(f_out_cas_sym_k[1],f_out_py[1]), "are the overapproximations of sigma the same"
     
    
    