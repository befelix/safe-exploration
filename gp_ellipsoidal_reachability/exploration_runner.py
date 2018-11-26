# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:37:59 2017

@author: tkoller
"""

from exploration_oracle import StaticMPCExplorationOracle, DynamicMPCExplorationOracle
from casadi import *
from gp_reachability import verify_trajectory_safety, trajectory_inside_ellipsoid
from matplotlib.cm import viridis
from utils import sample_inside_polytope
from utils_config import create_env, create_solver

import numpy as np
import matplotlib.pyplot as plt
import warnings


def run_exploration(conf,  
        static_exploration = True, 
        n_iterations = 50, 
        n_restarts_optimizer = 20,
        visualize = False,
        save_path = None,
        verify_safety = True,
        n_experiments = 1):          
    """ """

    l_inf_gain =[]
    l_sigm_sum = []
    l_sigm = []
    l_z_all = []
    l_x_next_obs_all = []
    l_x_next_pred = []
    l_x_next_prior = []
    l_gp_dict = []

    for jj in range(n_experiments):
        env = create_env(conf.env_name,conf.env_options)
        safempc, safe_policy = create_solver(conf,env)
        X,y = generate_initial_samples(env,conf,conf.relative_dynamics,safempc,safe_policy)
        safempc.update_model(X,y,opt_hyp = conf.train_gp,reinitialize_solver = False)

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
        
        #initialize color code for plotting the states
        #d_blue = np.linspace(1.0,0.0,n_iterations)
        d_red = np.linspace(0.0,1.0,n_iterations)    
        c_sample = lambda it: (d_red[it],0.0,0.0)#d_blue[it]) #use color code which transitions from green to blue
        
        if visualize or save_vis:
            fig, ax = env.plot_safety_bounds(color ="b")
            
            ##plot the initial train set
            x_train_init = exploration_module.safempc.gp.x_train
            c_black = (0.,0.,0.)
            n_train, _ = np.shape(x_train_init)
            for i in range(n_train):
                ax = env.plot_state(ax,x_train_init[i,:env.n_s],color = c_black)
                
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
            
            if verify_safety:
                x_i, u_i, feasible,safe_ctrl_applied, k_fb_all,k_ff_all,p_ctrl,q_all = exploration_module.find_max_variance(x_i,sol_verbose = True) 
                
                if feasible:
                    h_m_safe_norm,h_safe_norm,h_m_obs_norm,h_obs_norm = env.get_safety_constraints(normalize = True) 
                    safety_all[i],x_traj_safe = verify_trajectory_safety(env,x_i.squeeze(),k_fb_all, \
                                                k_ff_all,p_ctrl,h_m_safe_norm,h_safe_norm,h_m_obs_norm,h_obs_norm)
                    inside_ellipsoid[i,:] = trajectory_inside_ellipsoid(env,x_i.squeeze(), p_ctrl,q_all,k_fb_all,k_ff_all)
                    
                    
                    if visualize or save_vis:
                        if not ell is None:
                            for j in range(len(ell)):
                                ell[j].remove()
                        ax, ell = env.plot_ellipsoid_trajectory(p_ctrl,q_all,ax = ax,color = "r")
                        fig.canvas.draw()

                        if visualize:
                            plt.show(block=False)
                            plt.pause(0.5)
            
            else:
                x_i, u_i = exploration_module.find_max_variance(x_i,n_restarts_optimizer)
            
            if visualize or save_vis:
                ax = env.plot_state(ax,x = x_i,color = c_sample(i),normalize = False)
                fig.canvas.draw()
                if visualize:
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
            mu_next,s2_next = exploration_module.safempc.gp.predict(z_i)
            pred_conf = np.sqrt(s2_next)
            sigm[i] = pred_conf.squeeze() 
            x_next_prior[i,:] = safempc.eval_prior(x_i.T,u_i.T).squeeze()
            x_next_pred[i,:] = mu_next.squeeze() + safempc.eval_prior(x_i.T,u_i.T).squeeze()      
            sigm_sum[i] = np.sum(pred_conf)
            
            #update model and information gain
            exploration_module.update_model(z_i,x_next_obs.reshape((1,env.n_s)),train = conf.retrain_gp)  
            inf_gain[i,:] = exploration_module.get_information_gain()
            
            x_i = x_next

        l_inf_gain += [inf_gain]
        l_sigm_sum += [sigm_sum]
        l_sigm += [sigm]
        l_z_all += [z_all]
        l_x_next_obs_all += [x_next_obs_all]
        l_x_next_pred += [x_next_pred]
        l_x_next_prior += [x_next_prior]
        l_gp_dict += [solver.gp.to_dict()]
            
        if not save_path is None:
            results_dict = save_results(save_path,l_sigm_sum,l_sigm,l_inf_gain,l_z_all, \
                                        l_x_next_obs_all,l_x_next_pred,x_next_prior, \
                                        safempc.gp,safety_all,x_train_init)


def generate_initial_samples(env,conf,relative_dynamics,solver,safe_policy):

    std = conf.init_std_initial_data 
    mean = conf.init_m_initial_data

    if conf.init_mode == "random_rollouts":
        

        X,y,_,_ = do_rollout(env, conf.n_steps_init,plot_trajectory=conf.plot_trajectory,render = conf.render,mean = mean, std = std)
        for i in range(1,conf.n_rollouts_init):
            xx,yy, _,_ = do_rollout(env, conf.n_steps_init,plot_trajectory=conf.plot_trajectory,render = conf.render)
            X = np.vstack((X,xx))
            y = np.vstack((y,yy))

    elif conf.init_mode == "safe_samples":

        n_samples = conf.n_safe_samples
        n_max = conf.c_max_probing_init*n_samples
        n_max_next_state = conf.c_max_probing_next_state *n_samples 

        states_probing = env._sample_start_state(n_samples = n_max,mean= mean,std= std).T


        h_mat_safe, h_safe,_,_ = env.get_safety_constraints(normalize = True)

        bool_mask_inside = np.argwhere(sample_inside_polytope(states_probing,solver.h_mat_safe,solver.h_safe))
        states_probing_inside = states_probing[bool_mask_inside,:]

        n_inside_first = np.shape(states_probing_inside)[0]

        i = 0
        cont = True

        X = np.zeros((1,env.n_s+env.n_u))
        y = np.zeros((1,env.n_s))

        n_success = 0
        while cont:
            state = states_probing_inside[i,:]
            action = safe_policy(state.T)
            next_state, next_observation = env.simulate_onestep(state.squeeze(),action)

            

            if sample_inside_polytope(next_state[None,:],h_mat_safe,h_safe):
                state_action = np.hstack((state.squeeze(),action.squeeze()))
                X = np.vstack((X,state_action))
                if relative_dynamics:
                    y = np.vstack((y,next_observation - state))
                    
                else:
                    y = np.vstack((y,next_observation))
                n_success += 1

            i += 1

            if i >= n_inside_first  or n_success >= n_samples:
                cont = False



        if conf.verbose > 1:
            print("==== Safety controller evaluation ====")
            print("Ratio sample / inside safe set: {} / {}".format(n_inside_first,n_max))
            print("Ratio next state inside safe set / intial state in safe set: {} / {}".format(n_success,i))

        X = X[1:,:]
        y = y[1:,:]

        return X,y

    else:
        raise NotImplementedError("Unknown option initialization mode: {}".format(conf.init_mode))

    return X,y


def save_results(save_path,sigm_sum,sigm,inf_gain,z_all,x_next_obs_all,x_next_pred,x_next_prior,gp,x_train_0,safety_all = None):
    """ Create a dictionary from the results and save it """
    results_dict = dict()
    results_dict["sigm_sum"] = sigm_sum
    results_dict["sigm"] = sigm
    results_dict["inf_gain"] = inf_gain
    results_dict["z_all"] = z_all
    results_dict["x_next"] = x_next_obs_all
    results_dict["x_next_pred"] = x_next_pred
    results_dict["x_next_prior"] = x_next_prior
    results_dict["x_train_0"] = x_train_0
    
    if not safety_all is None:
        results_dict["safety_all"] = safety_all
    
    save_data_path = "{}/res_data".format(save_path)
    np.save(save_data_path,results_dict)
    
    gp_dict = gp.to_dict()
    save_data_gp_path = "{}/res_gp".format(save_path)
    np.save(save_data_gp_path,gp_dict)
    
    
    return results_dict
    