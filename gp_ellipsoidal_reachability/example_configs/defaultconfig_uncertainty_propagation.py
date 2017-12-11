# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""

import numpy as np
from default_config import DefaultConfig

class DefaultUncertaintyPropagation(DefaultConfig):
    """
    Options class for the exploration setting
    """
    ## task options
    task = "uncertainty_propagation" #don't change this 
    n_rollouts = 100 
    n_safe = 3
    n_restarts_optimizer = 20
    
    env_options = dict()
    init_std = np.array([1.2,.5])
    env_options["init_std"] = init_std # the std of the initial sample distribution
    
    ##GP
    gp_dict_path = None
    gp_data_path = None # None means no initial training data
    m = None #subset of data of size m for training
    kern_types = ["rbf","rbf"] #kernel type
    train_gp = True
    
    ## these parameters will be fixed during training
    gp_hyp = None
    ##environment
    env_name = "InvertedPendulum"

    
    ##safempc
    beta_safety = 3.
    n_perf = 0 #not required for this setting
    lqr_wx_cost = np.diag([1,1])
    lqr_wu_cost = .01*np.eye(1)
    
    lin_prior = True
    prior_model = dict()
    prior_m = .1
    prior_model["m"] = prior_m
    
    #general options
    save_results = True
    save_dir = None #the directory such that the overall save location is save_path_base/save_dir/
    save_path_base = "results_uncertainty_propagation" #the directory such that the overall save location is save_path_base/save_dir/
    data_savename = None
    
    def __init__(self,file_path):
        super(DefaultUncertaintyPropagation,self).create_savedirs(file_path)