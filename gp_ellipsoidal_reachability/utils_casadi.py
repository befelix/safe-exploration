# -*- coding: utf-8 -*-
""" Utility functions implemented with the casadi library

A set of utility functions required by various different modules / classes.
This module contains a subset of the functions implemented in the corresponding
utils.py file with the same functionality but is implemented in a way that
admits being use by a Casadi ( symbolic ) framework. 

@author: tkoller
"""

from casadi import mtimes, eig_symbolic, fmax, norm_2
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
    
def matrix_norm_2(a_mat,x = None,n_iter = None):
    """ Compute an approximation to the maximum eigenvalue of the hermitian matrix x
       
    TODO: Can we impose a convergence constraint?
    TODO: Check for diagonal matrix 
    
    """        
    n,m = a_mat.shape  
    assert n == m, "Input matrix has to be symmetric"
    
    if x is None:
        x = np.random.rand(n,1)
        x /= norm_2(x)
        
    if n_iter is None:
        n_iter = 2*n
    
    y = mtimes(a_mat,x)
    
    for i in range(n_iter):
        x = y / norm_2(y)
        y = mtimes(a_mat,x)
    
    return mtimes(y.T,x)
        
    
    