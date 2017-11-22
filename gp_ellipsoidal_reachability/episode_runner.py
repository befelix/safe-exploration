# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:11:23 2017

@author: tkoller
"""
from environments import InvertedPendulum
from sampling_models import MonteCarloSafetyVerification
from safempc import SafeMPC
from gp_models import SimpleGPModel
from utils_config import create_solver, create_env

import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import sys
import copy
import utils_ellipsoid


DEFAULT_EPISODE_OPTIONS = {"n_ep": 20, "n_steps": 15, "n_scenarios": 1, "n_steps_init": 8, "n_rollouts_init": 10}
DEFAULT_SAVE_OPTIONS = {}
DEFAULT_ENV_OPTIONS= {"env_name": "InvertedPendulum"}

def run(env_options = None, episode_options = None, controller_options = None, 
        save_options = None, data_safepath= None):
    """ Run episode setting """
        
    n_ep, n_steps, n_scenarios, n_steps_init, n_rollouts_init = _process_episode_options(episode_options)
    env, visualize = create_env(env_options)

    if n_scenarios > 1:
        raise NotImplementedError("For now we don't support multiple experiments!")
        
    X,y, S,z  = do_rollout(env, n_steps_init, visualize = False,plot_trajectory=False)
    for i in range(1,n_rollouts_init):
        xx,yy, ss,zz  = do_rollout(env, n_steps_init, visualize = False,plot_trajectory=False)
        X = np.vstack((X,xx))
        y = np.vstack((y,yy))
        S = np.vstack((S,ss))
        z = np.vstack((z,zz))
        
    solver = create_solver(env, controller_options)
    for i in range(n_ep):
        
        solver.update_model(S,y)
        xx, yy, ss,zz = do_rollout(env, n_steps, solver = solver, visualize = visualize, sampling_verification = False)
        
        X = np.vstack((X,xx))
        y = np.vstack((y,yy))
        S = np.vstack((S,ss))
        z = np.vstack((z,zz))
        
    if not data_safepath is None:
        np.savez(data_safepath,X=X,y=y,S=S,z = z)
        
        
def _process_episode_options(episode_options = None):
    """ Return default options replaced by the specified input episode options 
    
    Merge the default episode options with 
    Parameters
    ----------
    episode_options: dict
        The episode_options chosen by the user    
    """
    
    if episode_options is None:
        opts = DEFAULT_EPISODE_OPTIONS
    else:
        raise NotImplementedError()
        
    n_ep = opts["n_ep"]
    n_steps = opts["n_steps"]
    n_scenarios = opts["n_scenarios"]
    n_steps_init = opts["n_steps_init"]
    n_rollouts_init = opts["n_rollouts_init"]
    
    return n_ep, n_steps, n_scenarios, n_steps_init, n_rollouts_init
    

        
def do_rollout(env, n_steps, solver = None, relative_dynamics = False, 
               visualize = False, plot_trajectory = True, 
               verbosity = 1,sampling_verification = False,
               plot_ellipsoids = True,
               check_system_safety = False, safedir_trajectory_plots = "trajectory_plots"): #safedir_trajectory_plots = None
    """ Perform a rollout on the system
    
    TODO: measurement noise, x0_sigm?
    """
    
    state = env.reset()
    old_observation = state #assume noiseless initial observation
    target = env.get_target()
    n_successful = 0
    xx = np.zeros((1,env.n_s+env.n_u))
    ss = np.zeros((1,env.n_s+env.n_u))
    zz = np.zeros((1,env.n_s))
    yy= np.zeros((1,env.n_s))
    obs = state
    
    
    if plot_trajectory:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        _,_,_,p_obs,width_obs,height_obs = env.get_safe_bounds()
        ax.set_xlim(p_obs[0],p_obs[0]+width_obs)
        ax.set_ylim(p_obs[1],p_obs[1]+height_obs)
        ell = None
        
    if sampling_verification:
        gp = solver.gp
        sampler = MonteCarloSafetyVerification(gp)
        
    if check_system_safety:
        n_inside = 0
        n_test_safety = 0
        
    for i in range(n_steps):
        p_traj = None
        q_traj = None
        k_fb = None
        k_ff = None
        
        if visualize:
            env.render()
            
        if solver is None:
            action = env.random_action()
        else:
            t_start_solver = time.time()
            p_safe = np.array([0.0,0.0])
            action, safety_fail = solver.get_action(state,target,p_safe)#,lqr_only = True)
            t_end_solver = time.time()
            t_solver = t_end_solver - t_start_solver
            
            if verbosity > 0:
                print("total time solver in ms: {}".format(t_solver))
        
        action,next_state,observation,done = env.step(action)
        
        ## Plot the trajectory planned by the MPC solver        
        if plot_trajectory:
            if not solver is None and plot_ellipsoids and solver.has_openloop:
                p_traj,q_traj, k_fb, k_ff = solver.get_trajectory_openloop(state, get_controls = True)
                if not ell is None:
                    for j in range(len(ell)):
                        ell[j].remove()
                ax, ell = env.plot_ellipsoid_trajectory(p_traj,q_traj,ax = ax,color = "r")
                fig.canvas.draw()
                #plt.draw()
                
                plt.show(block=False)
                plt.pause(0.5)
            ax = env.plot_state(ax)
            fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.2)               
            if not safedir_trajectory_plots is None:
                save_name = "img_step_{}.png".format(i)
                safe_path = "{}/{}".format(safedir_trajectory_plots,save_name)
                plt.savefig(safe_path)
                
        
        ## Verify whether the GP distribution is inside the ellipsoid over multiple steps via sampling
        if sampling_verification:
            if p_traj is None:
                p_traj,q_traj, k_fb, k_ff = solver.get_trajectory_openloop(state, get_controls = True)
                
            _,s_all = sampler.sample_n_step(state[:,None],k_fb,k_ff,p_traj,n_samples = 300)
            safety_ratio,_ = sampler.inside_ellipsoid_ratio(s_all,q_traj,p_traj)
            if verbosity > 0:                
                print("\n==== GP samples inside Safety Ellipsoids (time step {}) ====".format(i))
                print(safety_ratio)
                print("==========================\n")
                
        ## check if the true system is inside the one-step ellipsoid by checking if the next state is inside p,q ellipsoid
        if not solver is None:
            if check_system_safety:
                if p_traj is None:
                    p_traj,q_traj, k_fb, k_ff = solver.get_trajectory_openloop(state, get_controls = True)
                bool_inside = utils_ellipsoid.sample_inside_ellipsoid(next_state,p_traj[0,:,None],q_traj[0])
                n_test_safety += 1
                if bool_inside:
                    n_inside += 1
                if verbosity > 0:
                    print("\n==== Next state inside uncertainty ellipsoid:{} ====\n".format(bool_inside))
                    #print(q_traj[2].reshape((env.n_s,env.n_s)))
        #system failed
        if done:
            break

        if verbosity > 0:
            print("\n===== time step {} =====".format(i))
            print("= State =")
            print(state)
            print("= Action =")
            print(action)
            print("= Observation =")
            print(observation)
            print("==========================\n")
        state_action = np.hstack((old_observation,action))
        state_action_noiseless = np.hstack((state,action))
        xx = np.vstack((xx,state_action))
        ss = np.vstack((ss,state_action_noiseless))
        obs = np.vstack((obs,observation))
        if relative_dynamics:
            yy = np.vstack((yy,observation - old_observation))
            zz = np.vstack((zz,observation - state))
            
        else:
            yy = np.vstack((yy,observation))
            zz = np.vstack((zz,observation))
            
        n_successful += 1
        state = next_state
        old_observation = observation
        
    if n_successful == 0:
        warnings.warn("Agent survived 0 steps, cannot collect data")
        xx = []
        yy = []
        ss = []
        zz = []
    else:
        xx = xx[1:,:]
        yy = yy[1:,:]
        ss = ss[1:,:]
        zz = zz[1:,:]
        
    print("Agent survived {} steps".format(n_successful))
    if verbosity >0:
        print("========== State Trajectory ===========")
        print(obs)
        if check_system_safety and n_test_safety > 0:
            print("\n======= percentage system steps inside safety bounds =======")
            print(float(n_inside)/n_test_safety)
    return xx,yy,ss,zz
    

    
if __name__ == "__main__":
    run(data_safepath=None)#"inv_pend_no_rel_dyn")
    