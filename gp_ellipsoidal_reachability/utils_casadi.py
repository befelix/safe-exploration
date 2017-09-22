# -*- coding: utf-8 -*-
""" Utility functions implemented with the casadi library

A set of utility functions required by various different modules / classes.
This module contains a subset of the functions implemented in the corresponding
utils.py file with the same functionality but is implemented in a way that
admits being use by a Casadi ( symbolic ) framework. 

@author: tkoller
"""

from casadi import *
from casadi.tools import *

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
        s_max = norm2(q)
        qk = mtimes(K,mtimes(q,K.T))
        sk_max = norm2(qk)
        
        l_max = s_max**2 + sk_max**2 
        box_lower = -L*l_max * 0.5
        box_upper =  L*l_max * 0.5
        
    if order == 1:
        s_max = norm2(q)
        qk = mtimes(K,mtimes(q,K.T))
        sk_max = norm2(qk)
        
        l_max = s_max + sk_max
        
        box_lower = -L*l_max 
        box_upper =  L*l_max 
        
    if verbose > 0:
        print("\n=== bounding-box approximation of order {} ===".format(order))
        print("largest eigenvalue of Q: {} \nlargest eigenvalue of KQK^T: {}".format(s_max,sk_max))
        
    return box_lower, box_upper
    
def all_elements_equal(x):
    """ Check if all elements in a 1darray are equal 
    
    Parameters
    ----------
    x: numpy 1darray
        Input array
        
        
    Returns
    -------
    b: bool
        Returns true if all elements in array are equal
        
    """
    
    assert len(np.shape(x)) == 1, "needs to be a 1darray"
    
    n_s = np.shape(x)
    
    return 
    

    
    