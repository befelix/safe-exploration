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
    n_rollouts = 100
    n_safe = 1
    
    ##GP
    #gp_dict_path = "exploration_gp_95.npy" # None means no initial training data
    gp_data_path = "data.npz"    
    
    """
    kern_dict_0 = dict()
    kern_dict_0["mul.Mat52.variance"] = 1.0
    kern_dict_0["mul.Mat52.lengthscale"] = 1.0  
    kern_dict_0["mul.linear.variances"] = np.array([1.,1., 0.0])
    kern_dict_0["linear.variances"] = np.array([.1,.1,.1])
    
    kern_dict_1 = dict()
    kern_dict_1["mul.Mat52.variance"] = 1.0
    kern_dict_1["mul.Mat52.lengthscale"] = 1.0  
    kern_dict_1["mul.linear.variances"] = np.array([1.,1., 0.0])
    kern_dict_1["linear.variances"] = np.array([ .1,.1,.1])
    
    gp_hyp = [kern_dict_0, kern_dict_1]
    """
    train_gp = True
    ilqr_init = True
    #gp_hyp = None
    #safempc
    beta_safety = 3.
    
    def __init__(self):
        """ """
        super(Config,self).__init__(__file__)