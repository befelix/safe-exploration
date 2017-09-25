# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:18:58 2017

@author: tkoller
"""

import numpy as np
import casadi as cas
import casadi.tools as castools

from casadi import SX, mtimes, vertcat, diag

def onestep_reachability(p_center,gp,K,k,L_mu,L_sigm,q_shape = None, c_safety = 1.,verbose = 1):
    """ Overapproximate the reachable set of states under affine control law
    
    given a system of the form:
        x_{t+1} = \mathcal{N}(\mu(x_t,u_t), \Sigma(x_t,u_t)),
    where x,\mu \in R^{n_s}, u \in R^{n_u} and \Sigma^{n_s \times n_s} are given bei the gp predictive mean and variance respectively
    we approximate the reachset of a set of inputs x_t \in \epsilon(p,Q)
    describing an ellipsoid with center p and shape matrix Q
    under the control low u_t = Kx_t + k 
    
    Parameters
    ----------
        p_center: n_s x 1 array[float]     
            Center of state ellipsoid        
        gp: SimpleGPModel     
            The gp representing the dynamics            
        K: n_u x n_s array[float]     
            The state feedback-matrix for the controls         
        k: n_u x 1 array[float]     
            The additive term of the controls
        L_mu: 1d_array of size n_s
            Set of Lipschitz constants on the Gradients of the mean function (per state dimension)
        L_sigm: 1d_array of size n_s
            Set of Lipschitz constants of the predictive variance (per state dimension)
        q_shape: np.ndarray[float], array of shape n_s x n_s, optional
            Shape matrix of state ellipsoid
        c_safety: float, optional
            The scaling of the semi-axes of the uncertainty matrix 
            corresponding to a level-set of the gaussian pdf.        
        verbose: int
            Verbosity level of the print output            
    Returns:
    -------
        p_new: n_s x 1 array[float]
            Center of the overapproximated next state ellipsoid
        Q_new: np.ndarray[float], array of shape n_s x n_s
            Shape matrix of the overapproximated next state ellipsoid  
    """         
    n_u, n_s = np.shape(K)
    
    if q_shape is None: # the state is a point
        print(type(K))
        print(np.shape(K))
        print(type(p_center))
        print(np.shape(p_center))
        u_p = mtimes(K,p_center) + k
        
        if verbose >0:
            print("\nApplying action:")
            print(u_p)
            
        z_bar = vertcat(p_center,u_p)
        p_new, q_new_unscaled = gp.predict_casadi(z_bar.T)
        
        print(warnings.warn("Need to verify this!"))
        q_1 = diag(q_new_unscaled.reshape((-1,)) * c_safety)
        
        p_1 = p_center + p_new.T
        
        if verbose >0:
            print_ellipsoid(p_1,q_1,text="uncertainty first state")
        
        return p_1, q_1
    else: # the state is a (ellipsoid) set
        if verbose > 0:
            print_ellipsoid(p_center,q_shape,text="initial uncertainty ellipsoid")
        ## compute the linearization centers
        x_bar = p_center   # center of the state ellipsoid
        u_bar = k   # u_bar = K*(u_bar-u_bar) + k = k
        z_bar = vertcat(x_bar,u_bar)
        
        if verbose >0:
            print("\nApplying action:")
            print(u_bar)
        ##compute the zero and first order matrices
        mu_0, sigm_0, Jac_mu = gp.predict_casadi(z_bar.T,compute_gradients = True)
        
        if verbose > 0:
            print_ellipsoid(mu_0,diag(sigm_0.squeeze()),text="predictive distribution")
            
        A_mu = Jac_mu[:,:n_s]
        B_mu = Jac_mu[:,n_s:]
         
        ## reach set of the affine terms
        H = A_mu + mtimes(B_mu,K)
        p_0 = mu_0.T + mtimes(B_mu,k-u_bar)
        
        Q_0 = mtimes(H,mtimes(q_shape,H.T))
        
        if verbose > 0:
            print_ellipsoid(p_0,Q_0,text="linear transformation uncertainty")
        ## computing the box approximate to the lagrange remainder
        lb_mean,ub_mean = compute_bounding_box_lagrangian(q_shape,L_mu,K,k,order = 2,verbose = verbose)
        lb_sigm,ub_sigm = compute_bounding_box_lagrangian(q_shape,L_sigm,K,k,order = 1,verbose = verbose)
        
        print(warnings.warn("Need to verify this!"))
        Q_lagrange_sigm = diag(c_safety*(ub_sigm+sqrt(sigm_0[0,:]))**2)   
        p_lagrange_sigm = SX.zeros((n_s,1))
        
        if verbose > 0:
            print_ellipsoid(p_lagrange_sigm,Q_lagrange_sigm,text="overapproximation lagrangian sigma")
    

        Q_lagrange_mu = ellipsoid_from_rectangle(ub_mean)
        p_lagrange_mu = SX.zeros((n_s,1))
        
        if verbose > 0:
            print_ellipsoid(p_lagrange_mu,Q_lagrange_mu,text="overapproximation lagrangian mu")
        
        p_sum_lagrange,Q_sum_lagrange = sum_two_ellipsoids(p_lagrange_sigm,Q_lagrange_sigm,p_lagrange_mu,Q_lagrange_mu)
        
        p_new , Q_new = sum_two_ellipsoids(p_sum_lagrange,Q_sum_lagrange,p_0,Q_0) 
        
        p_1, q_1 = sum_two_ellipsoids(p_new,Q_new,p_center,q_shape)
        if verbose > 0:
            print_ellipsoid(p_new,Q_new,text="accumulated uncertainty current step")
            
            q_comb = np.empty((4,n_s,n_s))
            q_comb[0] = q_shape
            q_comb[1] = Q_0
            q_comb[2] = Q_lagrange_mu
            q_comb[3] = Q_lagrange_sigm
            
            p_comb = np.zeros((4,n_s))
            
            p_test,q_test = sum_ellipsoids(p_comb,q_comb)
            print_ellipsoid(p_test,q_test,text="Test sum and old uncertainty combined")
        
            print_ellipsoid(p_1,q_1,text="sum old and new uncertainty")
            
            print("volume of ellipsoid summed individually")
            print(np.linalg.det(np.linalg.cholesky(q_1)))
            print("volume of combined sum:")
            print(np.linalg.det(np.linalg.cholesky(q_test)))
        
        
        return p_1,q_1
        

def lin_ellipsoid_safety_distance(p_center,q_shape,h_mat,h_vec,c_safety = 1.0):
    """ Distance between ELlipsoid and Polytope
    
    Evaluate the distance of an  ellipsoid E(p_center,q_shape), to a polytopic set
    of the form:
        h_mat * x <= h_vec.
        
    Parameters
    ----------
    p_center: n_s x 1 array
        The center of the state ellipsoid
    q_shape: n_s x n_s array
        The shape matrix of the state ellipsoid
    h_mat: m x n_s array:
        The shape matrix of the safe polytope (see above)
    h_vec: n_s x 1 array
        The additive vector of the safe polytope (see above)
        
    Returns
    -------
    d_safety: 1darray of length m
        The distance of the ellipsoid to the polytope. If d < 0 (elementwise),
        the ellipsoid is inside the poltyope (safe), otherwise safety is not guaranteed.
    """
    d_center = mtimes(h_mat,p_center)
    d_shape  = c_safety * sum2(mtimes(h_mat,q_shape)*h_mat)
    d_safety = d_center + d_shape - h_vec
    
    return d_safety.squeeze()
    
    
    