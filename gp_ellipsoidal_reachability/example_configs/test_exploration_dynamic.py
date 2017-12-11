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
    static_exploration = False    
    
    ##GP
    gp_data_path = "random_rollouts_25.npz"
    m = None #subset of data of size m for training
    
    ##safempc
    n_safe = 3
    
    # exploration
    n_iterations = 150
    lin_prior = True
    init_ilqr = True
    
    visualize = True

    def __init__(self):
        """ """
        super(Config,self).__init__(__file__)
       
            
            