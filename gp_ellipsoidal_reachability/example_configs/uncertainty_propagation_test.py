# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
import warnings
import numpy as np
import datetime

from defaultconfig_uncertainty_propagation import DefaultUncertaintyPropagation

class Config(DefaultUncertaintyPropagation):
    """
    Options class for the exploration setting
    """
    ## task options
    task = "uncertainty_propagation" #don't change this 
    n_rollouts = 50 
    n_safe = 2
    
    ##GP
    #gp_dict_path = "exploration_gp_95.npy" # None means no initial training data
    gp_data_path = "results_episode_setting/res_episode_setting_random_data_23-11-17-14-45/random_rollouts_75.npz"    
    
    kern_dict_0 = dict()
    kern_dict_0["mul.Mat52.variance"] = 1.0
    kern_dict_0["mul.Mat52.lengthscale"] = 1.0  
    kern_dict_0["mul.linear.variances"] = 4.74667619e-05
    kern_dict_0["linear.variances"] = np.array([  4.74667619e-05,   1.11359543e-05,   4.67080600e-01])
    
    kern_dict_1 = dict()
    kern_dict_1["mul.Mat52.variance"] = 1.0
    kern_dict_1["mul.Mat52.lengthscale"] = 1.0  
    kern_dict_1["mul.linear.variances"] = 2.88698464e-08
    kern_dict_1["linear.variances"] = np.array([  2.88698464e-08,   3.05621919e-09,   2.86362642e-04])
    
    gp_hyp = [kern_dict_0, kern_dict_1]
    gp_hyp = None
    #safempc
    beta_safety = 2.5
    
    def __init__(self):
        """ """
        super(Config,self).__init__(__file__)