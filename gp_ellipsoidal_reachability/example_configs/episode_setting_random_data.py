# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
import warnings
import numpy as np
import datetime

from defaultconfig_episode import DefaultConfigEpisode
from os.path import basename, splitext,dirname
from os import makedirs, getcwd
from shutil import copy


class Config(DefaultConfigEpisode):
    """
    Options class for the exploration setting
    """
    
    # episode settings
    n_ep = 0
    n_steps_init = 0
    init_std = np.array([0.4,.3])
    n_steps_init = 4
    n_rollouts_init = 50
    visualize = False
    plot_trajectory = False
    
    data_savename = "random_rollouts_{}".format(n_steps_init*n_rollouts_init)
    def __init__(self):
        super(Config,self).__init__(__file__)
            
            