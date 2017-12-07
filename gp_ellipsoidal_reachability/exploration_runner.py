# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:37:59 2017

@author: tkoller
"""

from exploration_oracle import StaticMPCExplorationOracle, DynamicMPCExplorationOracle
from casadi import *
from gp_reachability import verify_trajectory_safety, trajectory_inside_ellipsoid

import numpy as np
import matplotlib.pyplot as plt
import warnings


def run_exploration(env, safempc,conf,  
        static_exploration = True, 
        n_iterations = 50, 
        n_restarts_optimizer = 20,
        visualize = False,
        save_vis = False,
        save_path = None,
        verify_safety = True):          
    """ """
    if static_exploration:
        exploration_module = StaticMPCExplorationOracle(safempc,env)
        
        if verify_safety:
            warnings.warn("Safety_verification not possible in static mode")
            verify_safety = False
    else:
        exploration_module = DynamicMPCExplorationOracle(safempc,env)
        
    inf_gain = np.empty((n_iterations,env.n_s))
    sigm_sum = np.empty((n_iterations,1)) #the sum of the confidence intervals
    sigm = np.empty((n_iterations,env.n_s)) #the individual confidence intervals
    z_all = np.empty((n_iterations,env.n_s+env.n_u))
    x_next_obs_all = np.empty((n_iterations,env.n_s))
    x_next_pred = np.empty((n_iterations,env.n_s))
    x_next_prior = np.empty((n_iterations,env.n_s))
    
    if visualize:
        fig, ax = env.plot_safety_bounds()
        ell = None
        
    safety_all = None
    inside_ellipsod= None    
    if verify_safety:
        safety_all = np.zeros((n_iterations,),dtype = np.bool)
        inside_ellipsoid = np.zeros((n_iterations,safempc.n_safe))
    if static_exploration:
        x_i = None
    else:
        x_i = env.reset(env.p_origin)
        
    for i in range(n_iterations):
        ## find the most informative sample
        safempc = exploration_module.safempc
        exploration_module.init_solver()
        
        
        if verify_safety:
            x_i, u_i, feasible, k_fb_all,k_ff_all,p_ctrl,q_all = exploration_module.find_max_variance(x_i,sol_verbose = True) 
            
            if feasible:
                h_m_safe_norm,h_safe_norm,h_m_obs_norm,h_obs_norm = env.get_safety_constraints(normalize = True) 
                safety_all[i],x_traj_safe = verify_trajectory_safety(env,x_i.squeeze(),k_fb_all, \
                                            k_ff_all,p_ctrl,h_m_safe_norm,h_safe_norm,h_m_obs_norm,h_obs_norm)
                inside_ellipsoid[i,:] = trajectory_inside_ellipsoid(env,x_i.squeeze(), p_ctrl,q_all,k_fb_all,k_ff_all)
                
                if visualize:
                    if not ell is None:
                        for j in range(len(ell)):
                            ell[j].remove()
                    ax, ell = env.plot_ellipsoid_trajectory(p_ctrl,q_all,ax = ax,color = "r")
                    fig.canvas.draw()
                    plt.show(block=False)
                    plt.pause(0.5)
        
        else:
            x_i, u_i = exploration_module.find_max_variance(x_i,n_restarts_optimizer)
        
        if visualize:
            ax = env.plot_state(ax,x = x_i,normalize = False)
            fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.25)
            
        ## Apply to system and observe next state 
        
        #only reset the system to a different state in static mode
        if static_exploration:
            x_next,x_next_obs = env.simulate_onestep(x_i.squeeze(),u_i.squeeze())
        else:
            _,x_next,x_next_obs,_ = env.step(u_i.squeeze())
            
        
        
        x_next_obs_all[i,:] = x_next_obs 
         
        ## gather some information
        z_i = np.vstack((x_i,u_i)).T
        z_all[i] = z_i.squeeze()
        mu_next,s2_next = safempc.gp.predict(z_i)
        pred_conf = np.sqrt(s2_next)
        sigm[i] = pred_conf.squeeze() 
        x_next_prior[i,:] = safempc.eval_prior(x_i.T,u_i.T).squeeze()
        x_next_pred[i,:] = mu_next.squeeze() + safempc.eval_prior(x_i.T,u_i.T).squeeze()      
        sigm_sum[i] = np.sum(pred_conf)
        
        #update model and information gain
        exploration_module.update_model(z_i,x_next_obs.reshape((1,env.n_s)),train = conf.retrain_gp)  
        inf_gain[i,:] = exploration_module.get_information_gain()
        
        x_i = x_next
        
    if not save_path is None:
        results_dict = save_results(save_path,sigm_sum,sigm,inf_gain,z_all, \
                                    x_next_obs_all,x_next_pred,x_next_prior, \
                                    safempc.gp,safety_all)
        
        
    if save_vis:
        if save_path is None:
            warnings.warn("Cannot save the visualizations / figures without save_path being specified")
        save_plots(save_path,sigm_sum,sigm,inf_gain,z_all,env) 

        
def save_results(save_path,sigm_sum,sigm,inf_gain,z_all,x_next_obs_all,x_next_pred,x_next_prior,gp,safety_all = None):
    """ Create a dictionary from the results and save it """
    results_dict = dict()
    results_dict["sigm_sum"] = sigm_sum
    results_dict["sigm"] = sigm
    results_dict["inf_gain"] = inf_gain
    results_dict["z_all"] = z_all
    results_dict["x_next"] = x_next_obs_all
    results_dict["x_next_pred"] = x_next_pred
    results_dict["x_next_prior"] = x_next_prior
    if not safety_all is None:
        results_dict["safety_all"] = safety_all
    
    save_data_path = "{}/res_data".format(save_path)
    np.save(save_data_path,results_dict)
    
    gp_dict = gp.to_dict()
    save_data_gp_path = "{}/res_gp".format(save_path)
    np.save(save_data_gp_path,gp_dict)
    
    
    return results_dict
    
""" 
From here on visualization functions

"""    
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


    