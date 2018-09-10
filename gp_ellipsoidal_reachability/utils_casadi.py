# -*- coding: utf-8 -*-
""" Utility functions implemented with the casadi library

A set of utility functions required by various different modules / classes.
This module contains a subset of the functions implemented in the corresponding
utils.py file with the same functionality but is implemented in a way that
admits being use by a Casadi ( symbolic ) framework. 

@author: tkoller
"""

from casadi import mtimes, eig_symbolic, fmax, norm_2, horzcat,sqrt, exp
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
        
    
def trigProp(m, v, idx , a)
    """ Exact moment-matching for trig function with Gaussian input

    Compute E(a*sin(x)), E(a*cos(x)), V(a*sin(x)), V(a*cos(x)) and cross-covariances
    for Gaussian inputs x \sim N(m_idx,v_idx) as well as input-output covariances
    using exact moment-matching
    
    Parameters
    ----------
    m : dx1 ndarray[float | casadi.Sym]
        The mean of the input Gaussian
    v : dxd ndarray[float | casadi.Sym]
    idx: int
        The index to be trigonometrically augmented
    a: float
        A scalar coefficient 

    Returns
    -------
    m_out: 2x1 ndarray[float | casadi.Sym]
        The mean of the trigonometric output
    v_out: 2x2 ndarray[float | casadi.Sym] 
        The variance of the trigonometric output
    c_out: inv(v) times input-output-covariance

    """
    m_out = SX(2,1)
    v_out = SX(2,2)

    v_exp_0 = exp(-v[idx,idx]/2)

    m_out[0] = a*v_exp * sin(m[idx,0])
    m_out[1] = a*v_exp * cos(m[idx,0])

    
    v_exp_1 = exp(-v[idx,idx]*2)
    e_s_sq = (1 - v_exp_1*cos(2*m[idx,0]))/2
    e_c_sq = (1 + v_exp_1*cos(2*m[idx,0]))/2

    e_s_times_c = v_exp_1 * sin(2*m[idx,0])/2

    v_out[0,0] = e_s_sq - m_out[0]*2
    v_out[1,1] = e_c_sq - m_out[1]*2

    v_out[1,0] = e_s_times_c - m_out[0]*m_out[1]
    v_out[0,1] = v_out[1,0]

    v_out = a**2*v_out

    d = np.shape(m)[0]
    c = SX(d,2)
    c_out[idx,0] = m_out[0]
    c_out[idx,1] = -m_out[1]

    return m_out, v_out, c_out


