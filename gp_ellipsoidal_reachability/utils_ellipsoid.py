# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:34:16 2017

@author: tkoller
"""

import numpy as np
import scipy.linalg as sLa
import itertools

from utils import _prod_combinations_1darray,_solve_LLS,_vec_to_mat,all_elements_equal

from numpy import sqrt,trace,zeros,diag, eye

def sample_inside_ellipsoid(samples,p_center,q_shape,c = 1.):
    """ Check if a sample is inside a given ellipsoid
    
    Verify if a sample is inside the ellipsoid given by the shape matrix Q
    and center p. I.e. we check if
        (s-p).TQ^{-1}(s-p) <= c
    
    Args:
        samples (numpy.ndarray[float]): array of shape n_samples x n_s;
        p_center (numpy.ndarray[float]): array of shape n_s x 1;
        q_shape (numpy.ndarray[float]): array of shape n_s x n_s;   
        c (float, optional): The level set of the ellipsoid ( typically 1 makes sense)
    """
    
    d = distance_to_center(samples,p_center,q_shape)
    
    inside_ellipsoid_bool = d < c

    return inside_ellipsoid_bool     
    
    
def distance_to_center(samples,p_center,q_shape):
    """ Get the of a set of samples to the center of the ellipsoid
    
    
    Compute the distance:
        d = (s-p).TQ^{-1}(s-p)
    for a set of samples
    
    Args:
        samples (numpy.ndarray[float]): array of shape n_samples x n_s;
        p_center (numpy.ndarray[float]): array of shape n_s x 1;
        q_shape (numpy.ndarray[float]): array of shape n_s x n_s;
        
        
    Returns:
        distance (numpy.array[float]): 1darray of length n_samples containing 
                                    the distance to the center of the ellipsoid
                                    (see above)
    """
    
    p_centered = (samples - p_center.T).squeeze()
    d = np.sum(p_centered * sLa.solve(q_shape,p_centered.T).T, axis=1)
    
    return d
    
def sum_ellipsoids(b1,A1,b2,A2,p=None):
    """ 
    Ellipsoidal approximation to sum of two ellipsoids
    from
    "A Kurzhanski, I Valyi - Ellipsoidal Calculus for Estimation and Control"
    
    """
    
    ## choose p s.t. the trace of the new shape matrix is minimized
    if p is None:
        p = sqrt(trace(A1)/trace(A2))
        
    b_new = b1+b2
    A_new = (1+(1./p))*A1 + (1+p)*A2
    
    return b_new, A_new    
    
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
        
def ellipsoid_from_box(l_b,u_b,diag_only = False):
    """ Compute ellipsoid covering box
    
    Given a box defined by
     
        B = [l_b[0],u_b[0]] x ... x [l_b[-1],u_b[-1]] 
        
    we compute the minimum enclosing ellipsoid in closed-form
    as the solution to a linear least squares problem.
    
    NOTE: since B is a box centered around 0 the problem reduces to finding 
    a diagonal matrix D
    Method is described in:
        [1] :
        
    TODO:   Choice of point is terrible as of now, since it contains linearly dependent
            points which are not properly handled.
    
    Parameters
    ----------
        l_b: array_like, 1d    
            list of length n containing lower bounds of intervals defining box (see above)
        u_b: array_like, 1d    
            list of length n containing upper bounds of intervals defining box (see above)     
    Returns
    -------
        Q: np.ndarray[float, n_dim = 2], array of size n x n 
            Shape matrix of covering ellipsoid
        
    """        
    l_b = np.asarray(l_b)
    u_b = np.asarray(u_b)
    assert l_b.ndim == 1 and u_b.ndim == 1, "lb and ub need to be 1-dimensional (1darrays)!"
    assert l_b.size == u_b.size, "l_b and u_b need to have the same size!"
    assert np.all(l_b < u_b), "all elements of lb need to be smaller than ub!"
             
    n = len(l_b)    
    if all_elements_equal(l_b) and all_elements_equal(u_b): #check if hypercube
        d = np.sum(u_b**2)
        Q = np.diag(np.array([d]*n))
        return Q
        
    P = _get_edges_hyperrectangle(l_b,u_b) # create a set of points from l_b and u_b 
    
    ## transform the data using [1]    
    m,_ = np.shape(P)
    if diag_only:
        J = P**2
    else: 
        J = np.empty((m,n*(n+1)/2))   
        for i in range(m):
            J[i,:] = _prod_combinations_1darray(P[i,:])
        
    # avoid duplicates - this should be avoided in the first place
    J_unique = np.unique(J,axis=0)
    
    m_unique = np.shape(J_unique)[0]

    if m_unique < m:
        print("seem to have duplicates in the matrix. Size is reduced from {} to {}!".format(m,m_unique))
        
        print("rank of matrix: {}".format(np.linalg.matrix_rank(J_unique)))
    b = np.ones((m_unique,1))
    ## solve the resulting LLS
    params = _solve_LLS(J_unique,b,eps_mp = 1e-8)
    
    if diag_only:
        Q_inv = np.diag(params.squeeze())
    else:
        Q_inv = _vec_to_mat(params,n)
    
    # algebraic representation of matrices is always with Q^(-1)
    # hence we need to invert it to get the actual shape matrix
    Q = np.linalg.inv(Q_inv)
    
    print(Q)
    return np.round(Q, decimals = 8)