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
    verbose = 2
    static_exploration = False 

    ##safempc
    beta_safety = 2.0
    n_safe = 1
    n_perf = 0
    r = 1
    
    ##GP
    gp_data_path = "random_rollouts_25.npz"
    m = 25 #subset of data of size m for training
    #kern_types = ["rbf","rbf"]

    def __init__(self):
        """ """
        super(Config,self).__init__(__file__)
        
            
            
