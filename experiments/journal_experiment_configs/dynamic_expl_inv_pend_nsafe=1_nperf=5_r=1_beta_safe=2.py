# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
from .defaultconfig_exploration import DefaultConfigExploration


class Config(DefaultConfigExploration):
    """
    Options class for the exploration setting
    """
    verbose = 2
    static_exploration = False 

    # safempc
    beta_safety = 2.0
    n_safe = 1
    n_perf = 5
    r = 1


    def __init__(self):
        """ """
        super(Config, self).__init__(__file__)
        
            
            
