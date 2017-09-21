# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:43:16 2017

@author: tkoller
"""

import numpy as np
import numpy.linalg as nLa

from numpy.linalg import solve,norm
from numpy import sqrt,trace,zeros,diag, eye

def compute_bounding_box_lagrangian(p,Q,L,K,k,order = 2, verbose = 0):
    """ Compute lagrangian remainder using lipschitz constants
        and ellipsoidal input set with affine control law
    
    """
    SUPPORTED_TAYLOR_ORDER = [1,2]
    if not (order in SUPPORTED_TAYLOR_ORDER):
        raise ValueError("Cannot compute lagrangian remainder bounds for the given order")
        
    if order == 2:
        s_max = norm(Q,ord =  2)
        QK = np.dot(K,np.dot(Q,K.T))
        sk_max = norm(QK,ord =  2)
        
        l_max = s_max**2 + sk_max**2
        
        box_lower = -L*l_max * (1./order)
        box_upper =  L*l_max * (1./order)
        
    if order == 1:
        s_max = norm(Q,ord =  2)
        QK = np.dot(K,np.dot(Q,K.T))
        sk_max = norm(QK,ord =  2)
        
        l_max = s_max + sk_max
        
        box_lower = -L*l_max 
        box_upper =  L*l_max 
        
    if verbose > 0:
        print("\n=== bounding-box approximation of order {} ===".format(order))
        print("largest eigenvalue of Q: {} \nlargest eigenvalue of KQK^T: {}".format(s_max,sk_max))
        
    return box_lower,box_upper
    
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
    return np.allclose(x,x[0])
    
def print_ellipsoid(p_center,q_shape,text = "ellipsoid",visualize = False):
    """
    
    """
    
    print("\n")
    print("===== {} =====".format(text))
    print("center:")
    print(p_center)
    print("==========")
    print("diagonal of shape matrix:")
    print(diag(q_shape))
    print("===============")
    
def _vec_to_mat(v,n,tril_vec = True):
    """ Reshape vector into square matrix
    
    Inputs:
        v: vector containing matrix entries (either of length n^2 or n*(n+1)/2)
        n: the dimensionality of the new matrix
        
    Optionals:        
        tril_vec:   If tril_vec is True we assume that the resulting matrix is symmetric
                    and that the 
    
    """
    n_vec = len(v)
    
    if tril_vec:
        A = np.empty((n,n))        
        c=0
        for i in range(n):
            for j in range(i,n):
                A[i,j] = v[c]
                A[j,i] = A[i,j]
                c += 1     
    else: 
        A = np.reshape(v,(n,n))
 
    return A    
    
    
def _get_edges_hyperrectangle(l_b,u_b,m = None):
    """ Generate set of points from box-bounds
    
    Given a set of lower and upper bounds l_b,u_b
    defining the Box 
    
        B = [l_b[0],u_b[0]] x ... x [l_b[-1],u_b[-1]] 
        
    generate a set of points P which represent the box
    and can be used to fit an ellipsoid
    
    Inputs:
        l_b:    list of lower bounds of intervals defining box (see above)
        u_b:    list of upper bounds of intervals defining box (see above)
    
    Optionals:  
        m:     Number of points to compute. (m < 2^n)
    
    Outputs:
        P:      Matrix (k-by-n) of points obtained from the bounds
        
    """
    
    assert(len(l_b) == len(u_b))
    
    n = len(l_b)
    L = [None]*n
    
    for i in range(n):
        L[i] = [l_b[i],u_b[i]]     
    result = list(itertools.product(*L))
    
    P = np.array(result)
    if not m is None:
        assert m  <= np.pow(2,n) ,"Cannot extract that many points"
        P = P[:m,:]
        
    return P
 
   
def state_to_operational_ellipsoid():
    """ transform the state safety ellipsoid to the operational space ellipsoid
    
    TODO: This is definietly the wrong place for this function. 
        Needs to be in the corresponding environment class.
        Needs to be analytic and differentiable
    """
    raise NotImplementedError("No clue yet")

    
def _prod_combinations_1darray(v):
    """ Product of all pairs in a vector
    
    Parameters
    ----------
        v: array_like, 1-dimensional 
            input vector          
        
    Outputs:
        v_combined: array_like, 1-dimensional
            vector containing the product of all pairs in v  
    """
    n = len(v)
    v_combined = np.empty((n*(n+1)/2,))
    c=0
    for i in range(n):
        for j in range(i,n):
            v_combined[c] = v[i]*v[j]
            c+=1
    return v_combined
    
    
def _solve_LLS(A,b,eps_mp = 0.0):
    """ Solve Linear Least Squares Problem
    
    solve problem of the form
        || Ax-b ||^2 -> min
        
    Parameters
    ----------   
        A: m x n array[float]
            The data-matrix
        b: n x 1 array [float]
            The data-vector 
        eps_mp: float, optional
            Moore-Penrose diagonal noise
            
    Returns
    -------
        x: m x 1 array[float]
            Solution to the above problem
    """
    m,n = np.shape(A)    
    
    A_tilde = np.dot(A.T,A)
    if eps_mp > 0.0:
        A_tilde += eps_mp*eye(n)
    b_tilde = np.dot(A.T,b)

    x = solve(A_tilde,b_tilde)
    
    return x