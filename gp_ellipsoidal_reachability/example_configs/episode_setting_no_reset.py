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


class Config(DefaultConfigEpisode):
    """
    Options class for the exploration setting
    """
    

    

    def __init__(self):
        """ """
        
        conf_name = splitext(basename(__file__))[0]
        
        if self.save_results and self.save_dir is None:
            time_string = datetime.datetime.now().strftime("%d-%m-%y-%H-%M")
            res_path = "{}/res_{}_{}".format(self.save_path_base,conf_name,time_string)
            try:
                makedirs(res_path)
            except Exception, e:
                warnings.warn(e)
            self.save_path = res_path
            
            #copy config file into results folder
            dirname_conf = dirname(__file__)
            copy("{}/{}.py".format(dirname_conf,conf_name),"{}/".format(res_path))
            
            