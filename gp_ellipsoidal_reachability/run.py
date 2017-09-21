# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:14 2017

@author: tkoller
"""

import numpy as np
import deepdish as dd

from gp_models import SimpleGPModel
from sampling_models import MonteCarloSafetyVerification
from utils_ellipsoid import sample_inside_ellipsoid, sum_ellipsoids, _get_edges_hyperrectangle, \
                            ellipsoid_from_box
from utils import compute_bounding_box_lagrangian, print_ellipsoid
from gp_reachability import onestep_reachability

def get_dummy_setting(path = "observations_CartPole_N511.hd5"):
    """ Cart Pole setting with previously obtained samples
    
    
    Inputs:
        path:   Path to the training data file
        
    Outputs:
        p:          Center of state ellipsoid
        Q:          Shape matrix of state ellipsoid
        gp:         The gp representing the dynamics
        K:          The state feedback-matrix for the controls 
        k:          The additive term of the controls
        L_mu:       Set of Lipschitz constants on the Gradients of the mean function (per state dimension)
        L_sigm:     Set of Lipschitz constants of the predictive variance (per state dimension)
        c_safety:   The scaling of the semi-axes of the uncertainty matrix 
                        corresponding to a level-set of the gaussian pdf.
    
    """
    
    n_s = 5
    n_u = 1
    L_mu = np.array([0.05]*n_s)
    L_sigm = np.array([0.01]*n_s)
    K = np.random.rand(n_u,n_s) # need to choose this appropriately later
    k = 2*np.random.rand(n_u,1)
    
    Q = np.diag([0.5,0.3,0.2,0.1,0.1])
    p = np.zeros((n_s,1))
    
    train_data = dd.io.load(path)
    X = train_data["X"]
    y = train_data["y"]
    m = 100
    
    gp = SimpleGPModel(X,y,m)
        
    c_safety =  	11.07
    #c_safety =  	1.
    
    
    return p,Q,gp,K,k,L_mu,L_sigm,c_safety
    
    
    
if __name__ == "__main__":
    p,Q,gp,K,k,L_mu,L_sigma,c_safety = get_dummy_setting()
    
    
    p_1,q_1 = onestep_reachability(p,gp,K,k,L_mu,L_sigma,c_safety = c_safety)
    

    p_2,q_2 = onestep_reachability(p_1,gp,K,k,L_mu,L_sigma,q_shape=q_1,c_safety = c_safety)

    mc_check = MonteCarloSafetyVerification(gp)
    
    n_u,n_s = np.shape(K)
    K_two_step = np.empty((2,n_u,n_s))
    K_two_step[0] = K
    K_two_step[1] = K
    k_two_step = np.vstack((k,k))
    
    S,S_all = mc_check.sample_n_step(p,K_two_step,k_two_step,n=2,n_samples=2000)

    print_ellipsoid(p_2,q_2,text="Final uncertainty ellipsoid")
    Ratio,R_bool = mc_check.inside_ellipsoid_ratio(S[None,:,:],q_2[None,:,:],p_2.T)

    print("safety percentage")
    print(Ratio)
    print(S[:10,:])