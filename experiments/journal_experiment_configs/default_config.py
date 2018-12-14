# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""

import numpy as np
import warnings
import datetime

from os.path import basename, splitext,dirname
from os import makedirs, getcwd
from shutil import copy

class DefaultConfig(object):
    """
    Options class for the exploration setting
    """
    #the verbosity level 
    verbosity = 2
    ilqr_init = True

    type_perf_traj = 'mean_equivalent'
    r = 1
    perf_has_fb = True
    
    env_options=dict()
    
    
    def create_savedirs(self,file_path):
        """ """
        conf_name = splitext(basename(file_path))[0]
        
        if self.save_results and self.save_dir is None:
            time_string = datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")
            res_path = "{}/res_{}_{}".format(self.save_path_base,conf_name,time_string)
            try:
                makedirs(res_path)
            except Exception, e:
                warnings.warn(e)
            self.save_path = res_path
            
            #copy config file into results folder
            dirname_conf = dirname(file_path)
            copy("{}/{}.py".format(dirname_conf,conf_name),"{}/".format(res_path))
            