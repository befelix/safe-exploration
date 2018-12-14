#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:14 2017

@author: tkoller
"""

import argparse

from safe_exploration.utils_config import loadConfig, create_env, create_solver, get_model_options_from_conf
from safe_exploration.exploration_runner import run_exploration
from safe_exploration.episode_runner import run_episodic
from safe_exploration.uncertainty_propagation_runner import run_uncertainty_propagation


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


def check_config_conflicts(conf):
    """ Check if there are conflicting options in the Config

    Parameters
    ----------
    conf: Config
        The config file

    Returns
    -------
    has_conflict: Bool
        True, if there is a conflict in the config
    conflict_str: String
        The error message

    """

    has_conflict = False
    conflict_str = ""
    if conf.task == "exploration" and not conf.solver_type == "safempc":
        return True, "Exploration task only allowed with safempc solver"
    elif conf.task == "uncertainty_propagation" and not conf.solver_type == "safempc":
        return True ,"Uncertainty propagation task only allowed with safempc solver"

    return has_conflict, conflict_str


def run_scenario(args):
    """ Run the specified scenario

    Parameters
    ----------
    args:
        The parsed arguments (see create_parser for details)
    """
    config_path = args.scenario_config
    conf = loadConfig(config_path)

    conflict, conflict_str = check_config_conflicts(conf)
    if conflict:
        raise ValueError("There are conflicting settings: {}".format(conflict_str))

    env = create_env(conf.env_name,conf.env_options)

    gp_model_options = get_model_options_from_conf(conf,env)
    solver, safe_policy = create_solver(conf,env,gp_model_options)

    task = conf.task
    if task == "exploration":
        run_exploration(conf)#,conf.static_exploration,conf.n_iterations,
                        #conf.n_restarts_optimizer,conf.visualize,conf.save_vis,conf.save_path,conf.verify_safety,conf.n_experiments)
    elif task == "episode_setting":
        run_episodic(conf)

    elif task == "uncertainty_propagation":
        run_uncertainty_propagation(env,solver,conf)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run_scenario(args)
