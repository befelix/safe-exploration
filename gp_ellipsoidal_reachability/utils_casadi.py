# -*- coding: utf-8 -*-
""" Utility functions implemented with the casadi library

A set of utility functions required by various different modules / classes.
This module contains a subset of the functions implemented in the corresponding
utils.py file with the same functionality but is implemented in a way that
admits being use by a Casadi ( symbolic ) framework. 

@author: tkoller
"""

from casadi import mtimes, eig_symbolic, fmax, norm_2, horzcat,sqrt
import numpy as np


def compute_bounding_box_lagrangian(q,L,K,k,order = 2, verbose = 0):
    """ Compute box to overapproximate lagrangian remainder for Casadi
    
    Parameters
    ----------
    q: n_s x n_s array,
        The shape matrix of the input ellipsoid
    L: n_s x 0 1darray[float]
        Set of Lipschitz constants
    K: n_u x n_s array,
            The state feedback-matrix for the controls
    k: n_u x n-u array, 
            The additive term of the controls
    order: int, optional
        The order of the taylor remainder term
    verbose: int, optional:
        The verbosity level of the print output
    """
    SUPPORTED_TAYLOR_ORDER = [1,2]
    if not (order in SUPPORTED_TAYLOR_ORDER):
        raise ValueError("Cannot compute lagrangian remainder bounds for the given order")
    
    if order == 2:
        s_max = matrix_norm_2(q)

        qk = mtimes(K,mtimes(q,K.T))
        sk_max = matrix_norm_2(qk)

        l_max = s_max**2 + sk_max**2 
        box_lower = -L*l_max * 0.5
        box_upper =  L*l_max * 0.5
        
    if order == 1:
        s_max = matrix_norm_2(q)

        qk = mtimes(K,mtimes(q,K.T))
        sk_max = matrix_norm_2(qk)
        l_max = s_max + sk_max
        
        box_lower = -L*l_max 
        box_upper =  L*l_max 
        
    return box_lower, box_upper
    
def compute_remainder_overapproximations(q,k_fb,l_mu,l_sigma):
    """ Compute symbolically the (hyper-)rectangle over-approximating the lagrangians of mu and sigma 
    
    Parameters
    ----------
    q: n_s x n_s ndarray[casadi.SX.sym]
        The shape matrix of the current state ellipsoid
    k_fb: n_u x n_s ndarray[casadi.SX.sym]
        The linear feedback term
    l_mu: n x 0 numpy 1darray[float]
        The lipschitz constants for the gradients of the predictive mean
    l_sigma n x 0 numpy 1darray[float]
        The lipschitz constans on the predictive variance

    Returns
    -------
    u_mu: n_s x 0 numpy 1darray[casadi.SX.sym]
        The upper bound of the over-approximation of the mean lagrangian remainder
    u_sigma: n_s x 0 numpy 1darray[casadi.SX.sym]
        The upper bound of the over-approximation of the variance lagrangian remainder
    """
    n_u,n_s = np.shape(k_fb)
    s = horzcat(np.eye(n_s),k_fb.T)
    b = mtimes(s,s.T)

    qb = mtimes(q,b)
    evals = matrix_norm_2_generalized(b,q)
    r_sqr = vec_max(evals)
    
    u_mu = l_mu*r_sqr
    u_sigma = l_sigma*sqrt(r_sqr)
    
    return u_mu, u_sigma
def vec_max(x):
    """ Compute (symbolically) the maximum element in a vector
    
    Parameters
    ----------
    x : nx1 or array 
        The symbolic input array
    """
    n,_ = x.shape
    
    if n == 1: 
        return x[0]   
    c = fmax(x[0],x[1])
    if n > 2:
        for i in range(1,n-1):
            c = fmax(c,x[i+1])
    return c
    
def matrix_norm_2_generalized(a, b_inv, x = None, n_iter = None):
    """ Get largest generalized eigenvalue of the pair inv_a^{-1},b
    
    get the largest eigenvalue of the generalized eigenvalue problem 
        a x = \lambda b x 
    <=> b x = (1/\lambda) a x 
    
    Let \omega := 1/lambda
    
    We solve the problem 
        b x = \omega a x 
    using the inverse power iteration which converges to
    the smallest generalized eigenvalue \omega_\min
    
    Hence we can get  \lambda_\max = 1/\omega_\min,
        the largest eigenvalue of a x = \lambda b x
        
    """
    n,_ = a.shape
    if x is None:
        x = np.eye(n,1)
        x /= norm_2(x)
        
    if n_iter is None:
        n_iter = 2*n
        
    y = mtimes(b_inv,mtimes(a,x))
    for i in range(n_iter):
        x = y/norm_2(y)
        y = mtimes(b_inv,mtimes(a,x))
        
    return mtimes(y.T,x)
    
    
def matrix_norm_2(a_mat,x = None,n_iter = None):
    """ Compute an approximation to the maximum eigenvalue of the hermitian matrix x
       
    TODO: Can we impose a convergence constraint?
    TODO: Check for diagonal matrix 
    
    """        
    n,m = a_mat.shape  
    assert n == m, "Input matrix has to be symmetric"
    
    if x is None:
        x = np.eye(n,1)
        x /= norm_2(x)
        
    if n_iter is None:
        n_iter = 2*n
    
    y = mtimes(a_mat,x)
    
    for i in range(n_iter):
        x = y / norm_2(y)
        y = mtimes(a_mat,x)
    
    return mtimes(y.T,x)
        
    
    