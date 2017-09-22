# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:14 2017

@author: tkoller
"""

import numpy as np
import deepdish as dd
import time
from gp_models import SimpleGPModel
from sampling_models import MonteCarloSafetyVerification
from utils_ellipsoid import sample_inside_ellipsoid, sum_ellipsoids, \
                            ellipsoid_from_rectangle
from utils import compute_bounding_box_lagrangian, print_ellipsoid
from gp_reachability import onestep_reachability, multistep_reachability

def get_dummy_setting(path = "observations_CartPole_N511.hd5"):
    """ Cart Pole setting with previously obtained samples
    
    
    Parameters
    ----------
        path: str  
            Path to the training data file
        
    Returns
    -------
        p: 3x1 array[float]
            Center of state ellipsoid
        Q: 3x1 array[float]         
            Shape matrix of state ellipsoid
        gp: gp_models.SimpleGPModel        
            The gp representing the dynamics
        K: 1x3 array[float]
            The state feedback-matrix for the controls 
        k: 1x1 array[float] 
            The additive term of the controls
        L_mu: 3x0 1darray[float]
            Set of Lipschitz constants on the Gradients of the mean 
            function (per state dimension).
        L_sigm: 3x0 1darray[float]
            Set of Lipschitz constants of the 
            predictive variance (per state dimension).
        c_safety: float
            The scaling of the semi-axes of the uncertainty matrix 
            corresponding to a level-set of the gaussian pdf.
    
    """
    
    n_s = 5
    n_u = 1
    L_mu = np.array([0.01]*n_s)
    L_sigm = np.array([0.1]*n_s)
    K = np.random.rand(n_u,n_s) # need to choose this appropriately later
    k = 2*np.random.rand(n_u,1)
    
    Q = None #np.diag([0.5,0.3,0.2,0.1,0.1])
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
    verbose = 1      
    p_0,q_0,gp,K,k,L_mu,L_sigma,c_safety = get_dummy_setting() #load dummy settings cartpole
    
    ## dummy two-step controls
    n_u,n_s = np.shape(K)
    K_two_step = np.empty((3,n_u,n_s))
    K_two_step[0] = K
    K_two_step[1] = K
    K_two_step[2] = K
    k_two_step = np.vstack((k,k,k))
    
    ##two-step reachability
    start = time.time()
    p_2,q_2,p_all,q_all = multistep_reachability(p_0,gp,K_two_step,k_two_step,L_mu,L_sigma,q_0,c_safety,verbose)
    stop = time.time()
    print("Runtime for two-step uncertainty propoagation: {} s".format(stop-start))
    ## sample over multiple steps with the same settings
    mc_check = MonteCarloSafetyVerification(gp)
    S,S_all = mc_check.sample_n_step(p_0,K_two_step,k_two_step,n=3,n_samples=2000)

    print_ellipsoid(p_2,q_2,text="Final uncertainty ellipsoid")
    Ratio,R_bool = mc_check.inside_ellipsoid_ratio(S[None,:,:],q_2[None,:,:],p_2.T)
    
    print("safety percentage")
    print(Ratio)
    print(S[:10,:])