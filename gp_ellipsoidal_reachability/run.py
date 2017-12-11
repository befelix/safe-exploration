# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:14 2017

@author: tkoller
"""

import argparse
from utils_config import loadConfig, create_env, create_solver, get_model_options_from_conf
from exploration_runner import run_exploration
from episode_runner import run_episodic
from uncertainty_propagation_runner import run_uncertainty_propagation

def create_parser():
    """ Create the argparser """
    
    parser = argparse.ArgumentParser(description=""" Library for MPC-base Safe Exploration
    of unknown Dynamic Systems using Gaussian Processes \n\n
    
    Default configurations for the following scenarios exist:
        
    
    """)
    parser.add_argument("--scenario_config",
                        default= "example_configs/static_mpc_exploration.py", type = str,
                        help= """ Create your own scenario by copying one of the scenarios
                        in the example_configs/ directory and changing the default options.\n
                        Default scenario is static_mpc_exploration.py (see above for explanation) """)
    return parser
    
def run_scenario(args):
    """ Run the specified scenario """    
    config_path = args.scenario_config
    conf = loadConfig(config_path)
    
    env = create_env(conf.env_name,conf.env_options)
    gp_model_options = get_model_options_from_conf(conf,env)
    safempc = create_solver(conf,env,gp_model_options)
    
    task = conf.task
    
    if task == "exploration":
        run_exploration(env,safempc,conf,conf.static_exploration,conf.n_iterations,
                        conf.n_restarts_optimizer,conf.visualize,conf.save_vis,conf.save_path)
    elif task == "episode_setting":
        run_episodic(conf)
        
    elif task == "uncertainty_propagation":
        run_uncertainty_propagation(env,safempc,conf)
        
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run_scenario(args)