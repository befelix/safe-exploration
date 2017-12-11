# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
import warnings
import numpy as np
import datetime

from defaultconfig_exploration import DefaultConfigExploration
from os.path import basename, splitext,dirname
from os import makedirs, getcwd
from shutil import copy


class Config(DefaultConfigExploration):
    """
    Options class for the exploration setting
    """
    
    ##GP
    gp_data_path = "results_episode_setting/res_episode_setting_random_data_23-11-17-14-45/random_rollouts_75.npz"
    m = None #subset of data of size m for training
    kern_type = "prod_lin_rbf" #kernel type
    
    ##safempc
    n_safe = 1
    
    # exploration
    n_iterations = 150
    n_restarts_optimizer = 20
    lin_prior = True

    def __init__(self):
        """ """
        super(Config,self).__init__(__file__)
       
            
            