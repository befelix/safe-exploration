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
from utils import generate_initial_samples

import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import sys
import copy
import utils_ellipsoid


def run_episodic(conf):
    """ Run episode setting """

    warnings.warn("Need to check relative dynamics")
    

    X_all = []
    y_all = [] 
    cc_all = []
    exit_codes_all = []
    safety_failure_all = []
    for k in range(conf.n_scenarios):
        
        env = create_env(conf.env_name,conf.env_options)
        solver, safe_policy = create_solver(conf,env)

        X,y = generate_initial_samples(env,conf,conf.relative_dynamics,solver,safe_policy)
        
        X_list = [X]
        y_list = [y]
        exit_codes_k = []
        safety_failure_k = []
        cc_k = []
        for i in range(conf.n_ep):

            solver.update_model(X,y,opt_hyp = conf.train_gp,reinitialize_solver = False)

            if i ==0:
                solver.init_solver(conf.cost)

            xx, yy, cc, exit_codes_i, safety_failure = do_rollout(env, conf.n_steps, cost = conf.rl_immediate_cost, 
                                            solver = solver,plot_ellipsoids = conf.plot_ellipsoids,
                                            plot_trajectory=conf.plot_trajectory,render = conf.render, obs_frequency = conf.obs_frequency)
            
            X = np.vstack((X,xx))
            y = np.vstack((y,yy))

            X_list += [xx]
            y_list += [yy]
            cc_k += [cc]
            exit_codes_k += [exit_codes_i]
            safety_failure_k += [safety_failure]

        exit_codes_all += [exit_codes_k]
        safety_failure_all += [safety_failure_k]
        cc_all += [cc_k]
        X_all += [X_list]
        y_all += [y_list]
      
    if not conf.data_savepath is None:
        savepath_data = "{}/{}".format(conf.save_path,conf.data_savepath)
        a,b = solver.lin_model
        np.savez(savepath_data,X=X,y=y,a = a,b=b, init_mode = conf.init_mode)

    if conf.save_results:
        save_name_results = conf.save_name_results
        if save_name_results is None:
            save_name_results = "results_episode"

        savepath_results  = conf.save_path +"/"+save_name_results

        results_dict = dict()
        results_dict["cc_all"] = cc_all
        results_dict["X_all"] = X_all
        results_dict["y_all"] = y_all
        results_dict["exit_codes"] = exit_codes_all
        results_dict["safety_failure_all"] = safety_failure_all
        print(savepath_results)
        
        np.save(savepath_results,results_dict)
        
        ## TO-DO: may wanna do this aswell
        #gp_dict = gp.to_dict()
        #save_data_gp_path = "{}/res_gp".format(save_path)
        #np.save(save_data_gp_path,gp_dict)

    
if __name__ == "__main__":
    env_options = namedtuple('conf',['env_name'])
    conf = env_options(env_name = "InvertedPendulum")
    print(conf.env_name)
    run(data_safepath="inv_pend_no_rel_dyn",env_options = conf)#)
    