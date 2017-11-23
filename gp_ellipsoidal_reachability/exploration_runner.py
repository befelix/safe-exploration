# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:37:59 2017

@author: tkoller
"""

from exploration_oracle import MPCExplorationOracle
from casadi import *

import numpy as np
import matplotlib.pyplot as plt
import warnings


def run_exploration(env, safempc,  
        static_exploration = True, 
        n_iterations = 50, 
        n_restarts_optimizer = 20,
        visualize = False,
        save_vis = False,
        save_path = None):          
    """ """
    exploration_module = MPCExplorationOracle(safempc,env)
    
    inf_gain = np.empty((n_iterations,env.n_s))
    sigm_sum = np.empty((n_iterations,1)) #the sum of the confidence intervals
    sigm = np.empty((n_iterations,env.n_s)) #the individual confidence intervals
    z_all = np.empty((n_iterations,env.n_s+env.n_u))
    x_next_obs = np.empty((n_iterations,env.n_s))
    x_next_pred = np.empty((n_iterations,env.n_s))
    x_next_prior = np.empty((n_iterations,env.n_s))
    
    for i in range(n_iterations):
        ## find the most informative sample
        exploration_module.init_solver()
        safempc = exploration_module.safempc
        x_i, u_i, sigm_i = exploration_module.find_max_variance(n_restarts = n_restarts_optimizer)
        
        ## Apply to system and observe next state 
        env.reset(x_i.squeeze(),0)
        _,_,x_plus_obs,_ = env.step(u_i.squeeze())
        
        x_next_obs[i,:] = x_plus_obs 
         
        z_i = np.vstack((x_i,u_i)).T         
        
        ## gather some information
        z_all[i] = z_i.squeeze()
        mu_next,s2_next = safempc.gp.predict(z_i)
        sigm[i] = np.sqrt(s2_next).squeeze() 
        x_next_prior[i,:] = safempc.eval_prior(x_i.T,u_i.T).squeeze()
        x_next_pred[i,:] = mu_next.squeeze() + safempc.eval_prior(x_i.T,u_i.T).squeeze()      
        sigm_sum[i] = sigm_i
        
        #update model and information gain
        exploration_module.update_model(z_i,x_plus_obs.reshape((1,env.n_s)),train = train_model)  
        inf_gain[i,:] = exploration_module.get_information_gain()

    if not save_path is None:
        results_dict = save_results(save_path,sigm_sum,sigm,inf_gain,z_all,x_next_obs,x_next_pred,x_next_prior)
        
        
    if save_vis:
        if save_path is None:
            warnings.warn("Cannot save the visualizations / figures without save_path being specified")
        save_plots(save_path,sigm_sum,sigm,inf_gain,z_all,env) 

        
def save_results(save_path,sigm_sum,sigm,inf_gain,z_all,x_next_obs,x_next_pred,x_next_prior):
    """ Create a dictionary from the results and save it """
    results_dict = dict()
    results_dict["sigm_sum"] = sigm_sum
    results_dict["sigm"] = sigm
    results_dict["inf_gain"] = inf_gain
    results_dict["z_all"] = z_all
    results_dict["x_next"] = x_next_obs
    results_dict["x_next_pred"] = x_next_pred
    results_dict["x_next_prior"] = x_next_prior
    save_data_path = "{}/res_data".format(save_path)
    np.save(save_data_path,results_dict)
    
    return results_dict
    
def save_plots(save_path,sigm_sum,sigm,inf_gain,z_all,env):
    """ """
    n_it = np.shape(inf_gain)[0]
    fig, ax = plt.subplots()
    ax.plot(np.arange(n_it),inf_gain.squeeze())
    ax.set_xlabel('iteration')
    ax.set_ylabel('information gain')
    
    path_inf_gain = "{}/information_gain.png".format(save_path)
    plt.savefig(path_inf_gain)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(n_it),sigm_sum.squeeze())
    ax.set_xlabel('iteration')
    ax.set_ylabel('sum of confidence intervals')
    
    path_sigm_sum = "{}/sigm_sum.png".format(save_path)
    plt.savefig(path_sigm_sum)
    
    fig, ax = env.plot_safety_bounds()
    ax.plot(z_all[:,0],z_all[:,1],"bx")
    
    path_sampleset = "{}/sample_set.png".format(save_path)
    plt.savefig(path_sampleset)
    
    ##information gain plot 
""" 
From here on visualization functions

"""

def get_information_gain_plot():
    """ """
    raise NotImplementedError("Still need to implement this")    
    