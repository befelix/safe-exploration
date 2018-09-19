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
from collections import namedtuple
from utils import sample_inside_polytope 

import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import sys
import copy
import utils_ellipsoid


def run_episodic(conf,relative_dynamics):
    """ Run episode setting """

    warnings.warn("Need to check relative dynamics")
        
    env = create_env(conf.env_name,conf.env_options)

    X,y = generate_initial_samples(env,conf,relative_dynamics)
    
    for i in range(conf.n_ep):
        if i ==0:
            solver = create_solver(conf,env)
            solver.update_model(X,y)
            solver.init_solver(conf.cost)
        else:
            solver.update_model(X,y)

        xx, yy = do_rollout(env, conf.n_steps, solver = solver,plot_ellipsoids = conf.plot_ellipsoids,plot_trajectory=conf.plot_trajectory,render = conf.render)
        
        X = np.vstack((X,xx))
        y = np.vstack((y,yy))

        
    if not conf.data_savepath is None:
        savepath_data = "{}/{}".format(conf.save_path,conf.data_savepath)
        np.savez(savepath_data,X=X,y=y)
        
        
def do_rollout(env, n_steps, solver = None, relative_dynamics = False,
               plot_trajectory = True, 
               verbosity = 1,sampling_verification = False,
               plot_ellipsoids = False,render = False,
               check_system_safety = False, savedir_trajectory_plots = None): #safedir_trajectory_plots = None
    """ Perform a rollout on the system
    
    TODO: measurement noise, x0_sigm?
    """
    
    state = env.reset()
    
    xx = np.zeros((1,env.n_s+env.n_u))
    yy= np.zeros((1,env.n_s))

    obs = state
    
    n_successful = 0

    if plot_trajectory:
        fig, ax = env.plot_safety_bounds()
        
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
            
        if solver is None:
            action = .2 * env.random_action()
        else:
            t_start_solver = time.time()
            print(np.shape(state))
            action, safety_fail = solver.get_action(state,env.target_ilqr)#,lqr_only = True)
            t_end_solver = time.time()
            t_solver = t_end_solver - t_start_solver
            
            if verbosity > 0:
                print("total time solver in ms: {}".format(t_solver))
        
        action,next_state,observation,done = env.step(action)


        if verbosity > 0:
            print("\n==== Applied action at time step {} ====".format(i))
            print(action)
            print("\n==== Next state ====")
            print(next_state)
            print("==========================\n")

        if render:
            env.render()
        
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
            if not savedir_trajectory_plots is None:
                save_name = "img_step_{}.png".format(i)
                save_path = "{}/{}".format(savedir_trajectory_plots,save_name)
                plt.savefig(save_path)
                
        
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


        state_action = np.hstack((state,action))
        xx = np.vstack((xx,state_action))

        if relative_dynamics:
            yy = np.vstack((zz,observation - state))
            
        else:
            yy = np.vstack((yy,observation))

            
        n_successful += 1
        state = next_state
        
    if n_successful == 0:
        warnings.warn("Agent survived 0 steps, cannot collect data")
        xx = []
        yy = []
    else:
        xx = xx[1:,:]
        yy = yy[1:,:]

        
    print("Agent survived {} steps".format(n_successful))
    if verbosity >0:
        print("========== State Trajectory ===========")
        print(obs)
        if check_system_safety and n_test_safety > 0:
            print("\n======= percentage system steps inside safety bounds =======")
            print(float(n_inside)/n_test_safety)
    return xx,yy
    

def generate_initial_samples(env,conf,relative_dynamics):

    if conf.init_mode == "random_rollouts":
        X,y = do_rollout(env, conf.n_steps_init,plot_trajectory=conf.plot_trajectory,render = conf.render)
        for i in range(1,conf.n_rollouts_init):
            xx,yy  = do_rollout(env, conf.n_steps_init,plot_trajectory=conf.plot_trajectory,render = conf.render)
            X = np.vstack((X,xx))
            y = np.vstack((y,yy))

    elif conf.init_mode == "safe_samples":
        solver = create_solver(conf,env)

        n_samples = conf.n_safe_samples
        n_max = conf.c_max_probing_init*n_samples
        n_max_next_state = conf.c_max_probing_next_state *n_samples 

        states_probing = env._sample_start_state(n_samples = n_max).T


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
            action = solver.safe_policy(state.T)
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
            print("Ratio next state inside safe set / intial state in safe set: {} / {}".format(n_success,n_inside_first))


        X = X[1:,:]
        y = y[1:,:]

        return X,y

    else:
        raise NotImplementedError("Unknown option initialization mode: {}".format(conf.init_mode))

    return S,X,y,z
    
if __name__ == "__main__":
    env_options = namedtuple('conf',['env_name'])
    conf = env_options(env_name = "InvertedPendulum")
    print(conf.env_name)
    run(data_safepath="inv_pend_no_rel_dyn",env_options = conf)#)
    