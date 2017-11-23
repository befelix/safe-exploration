# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:43:16 2017

@author: tkoller
"""

import numpy as np
import numpy.linalg as nLa
import scipy.linalg as sLa
import warnings

from numpy.linalg import solve,norm
from numpy import sqrt,trace,zeros,diag, eye


def dlqr(a,b,q,r):
    """ Get the feedback controls from linearized system at the current time step
    
    for a discrete time system Ax+Bu
    find the infinite horizon optimal feedback controller
    to steer the system to the origin
    with
    u = -K*x 
    """
    x = np.matrix(sLa.solve_discrete_are(a, b, q, r))
 
    k = np.matrix(sLa.inv(b.T*x*b+r)*(b.T*x*a))
 
    eigVals, eigVecs = sLa.eig(a-b*k)
    
    return np.asarray(k), np.asarray(x), eigVals
        
        
def compute_bounding_box_lagrangian(q,L,K,k,order = 2, verbose = 0):
    """ Compute lagrangian remainder using lipschitz constants
        and ellipsoidal input set with affine control law
    
    """
    warnings.warn("Function is deprecated")
    
    SUPPORTED_TAYLOR_ORDER = [1,2]
    if not (order in SUPPORTED_TAYLOR_ORDER):
        raise ValueError("Cannot compute lagrangian remainder bounds for the given order")
        
    if order == 2:
        s_max = norm(q,ord =  2)
        QK = np.dot(K,np.dot(q,K.T))
        sk_max = norm(QK,ord =  2)
        
        l_max = s_max**2 + sk_max**2
        
        box_lower = -L*l_max * (1./order)
        box_upper =  L*l_max * (1./order)
        
    if order == 1:
        s_max = norm(q,ord =  2)
        QK = np.dot(K,np.dot(q,K.T))
        sk_max = norm(QK,ord =  2)
        
        l_max = s_max + sk_max
        
        box_lower = -L*l_max 
        box_upper =  L*l_max 
        
    if verbose > 0:
        print("\n=== bounding-box approximation of order {} ===".format(order))
        print("largest eigenvalue of Q: {} \nlargest eigenvalue of KQK^T: {}".format(s_max,sk_max))
        
    return box_lower,box_upper
    
def compute_remainder_overapproximations(q,k_fb,l_mu,l_sigma):
    """ Compute the (hyper-)rectangle over-approximating the lagrangians of mu and sigma 
    
    Parameters
    ----------
    q: n_s x n_s ndarray[float]
        The shape matrix of the current state ellipsoid
    k_fb: n_u x n_s ndarray[float]
        The linear feedback term
    l_mu: n x 0 numpy 1darray[float]
        The lipschitz constants for the gradients of the predictive mean
    l_sigma n x 0 numpy 1darray[float]
        The lipschitz constans on the predictive variance

    Returns
    -------
    u_mu: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the mean lagrangian remainder
    u_sigma: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the variance lagrangian remainder
    """
    n_u,n_s = np.shape(k_fb)
    s = np.hstack((np.eye(n_s),k_fb.T))
    b = np.dot(s,s.T)
    qb = np.dot(q,b)
    evals,_ = sLa.eig(qb)
    r_sqr = np.max(evals)
    ## This is equivalent to:
    # q_inv = sLA.inv(q)
    # evals,_,_ = sLA.eig(b,q_inv)
    # however we prefer to avoid the inversion
    # and breaking the symmetry of b and q
    
    u_mu = l_mu*r_sqr
    u_sigma = l_sigma*np.sqrt(r_sqr)
    
    return u_mu, u_sigma
    
    
    
    
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
    
    
def solve_LLS(A,b,eps_mp = 0.0):
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