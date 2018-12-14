# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
from default_config import DefaultConfig
import numpy as np

class DefaultConfigEpisode(DefaultConfig):
    """
    Options class for the exploration setting
    """
    ## task options
    task = "episode_setting" #don't change this 
    
    ##GP
    gp_data_path = None # None means no initial training data
    m = None #subset of data of size m for training
    kern_types = ["rbf","rbf"] #kernel type
    gp_dict_path = None
    gp_hyp = None
    train_gp = True



    ##environment
    env_name = "InvertedPendulum"


    ##safempc
    beta_safety = 3 
    n_safe = 2
    n_perf = 0
    lqr_wx_cost = np.diag([5.,1.,5.,1.])
    lqr_wu_cost = 50*np.eye(1)
    lin_prior = True
    prior_model = dict()
    cost_func = None
    # episode settings
    n_ep = 10
    n_steps = 15
    n_steps_init = 5
    n_rollouts_init = 15
    n_scenarios = 0
    
    #general options
    render = False
    visualize = True
    plot_ellipsoids = False
    plot_trajectory = False
    
    save_results = True
    save_vis = True
    save_dir = None #the directory such that the overall save location is save_path_base/save_dir/
    save_path_base = "results_episode_setting" #the directory such that the overall save location is save_path_base/save_dir/
    data_savepath = None
    
    def __init__(self,file_path):
        super(DefaultConfigEpisode,self).create_savedirs(file_path)
    