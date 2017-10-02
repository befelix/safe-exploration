# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:14 2017

@author: tkoller
"""

import numpy as np
import deepdish as dd
import sys
import time
from gp_models import SimpleGPModel
from sampling_models import MonteCarloSafetyVerification
from utils import compute_bounding_box_lagrangian, print_ellipsoid
from gp_reachability import onestep_reachability, multistep_reachability
from cartpole_casadi import create_solver
from safempc import SafeMPC

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
    L_sigm = np.array([0.05]*n_s)
    K = np.random.rand(n_u,n_s) # need to choose this appropriately later
    k = np.random.rand(n_u,1)
    
    Q = None #np.diag([0.5,0.3,0.2,0.1,0.1])
    p = np.zeros((n_s,1))
    
    train_data = dd.io.load(path)
    X = train_data["X"]
    y = train_data["y"]
    m = 150
    kern_type = "prod_lin_rbf"
    gp = SimpleGPModel(X,y,m,kern_type)
        
    c_safety =  	11.07
    #c_safety =  	1.
    
    
    return p,Q,gp,K,k,L_mu,L_sigm,c_safety
        
def dummy_mpc_cartpole():
    """ Run the Cart Pole environment with safety mode in a dummy setting
    
    """
    
    p_0,q_0,gp,K,_,l_mu,l_sigma,c_safety = get_dummy_setting()
    
    n_u,n_s = np.shape(K)
    h_mat_safe = np.hstack((np.eye(n_s,1),-np.eye(n_s,1))).T
    h_safe = np.array([0.01,0.01])
    h_mat_obs = np.copy(h_mat_safe)
    h_obs = np.array([0.01,0.01])
    
    n_safe = 5
    n_perf = None
    
    wx_cost = 5*np.eye(n_s)
    wu_cost = 1
    solver,lbg,ubg = create_solver(gp,n_safe,n_perf,h_safe,h_mat_safe,h_obs,h_mat_obs,n_s,n_u,l_mu,l_sigma,wx_cost,wu_cost,c_safety)
    
    start_state = np.zeros((n_s,1))
    x_target = np.eye(n_s,1)
    params = np.vstack((start_state,x_target))
    u_0 = np.random.rand(n_s*n_u*n_safe+n_u*n_safe,1)  
    #sol = solver(x0=u_0,p=params) #
    sol = solver(x0=u_0,lbg=lbg,ubg=ubg,p=params)
    print(sol)
    
    return None
    
def dummy_cartpole_safempc():
    """ Run Cart Pole with SafeMPC class in a dummy setting """
    
    p_0,q_0,gp,K,_,l_mu,l_sigma,c_safety = get_dummy_setting()
    
    
    
    n_u,n_s = np.shape(K)
    x_target = np.eye(n_s,1)
    h_mat_safe = np.hstack((np.eye(n_s,1),-np.eye(n_s,1))).T
    h_safe = np.array([.5,.5]).reshape((-1,1))
    h_mat_obs = np.copy(h_mat_safe)
    h_obs = np.array([.5,.5]).reshape((-1,1))
    
    n_safe = 5
    n_perf = None
    
    wx_cost = 10*np.eye(n_s)
    wu_cost = 0.01
    
    safe_mpc_solver = SafeMPC(n_safe,n_perf,gp,l_mu,l_sigma,h_mat_safe,h_safe,
                              h_mat_obs,h_obs,wx_cost,wu_cost)
    safe_mpc_solver.init_solver()
    
    safe_mpc_solver.solve(p_0,x_target)    
    sys.exit(0)
    
if __name__ == "__main__":
    #dummy_cartpole_safempc()
    #dummy_mpc_cartpole()
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