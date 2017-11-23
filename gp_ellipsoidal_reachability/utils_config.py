# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:44:58 2017

@author: tkoller
"""

from gp_models import SimpleGPModel
from safempc import SafeMPC
from environments import InvertedPendulum
from os.path import abspath, join, exists, dirname, basename, realpath
from importlib import import_module
from os import mkdir
from utils import dlqr

import numpy as np 

def create_solver(conf, env, model_options = None):
    """ Create a solver from a set of options and environment information"""
    

    

    n_safe = conf.n_safe
    n_perf = conf.n_perf
    l_mu = env.l_mu
    l_sigm = env.l_sigm
    h_mat_safe = env.h_mat_safe
    h_safe = env.h_safe
    h_mat_obs = env.h_mat_obs
    h_obs = env.h_obs
    wx_cost = conf.lqr_wx_cost
    wu_cost = conf.lqr_wu_cost
    lin_model = None
    
    a,b = env.linearize_discretize()
    
    safe_policy = None
    if conf.lin_prior:
        lin_model = (a,b)
    else:
        #by default takes identity as prior. Need to define safe_policy
        q= wx_cost
        r= wu_cost
        k_lqr,_,_ = dlqr(a,b,q,r)
        k_fb = -k_lqr
        safe_policy = lambda x: np.dot(K,x) 
        
    if model_options is None:
        gp = SimpleGPModel(env.n_s,env.n_u,kern_type = conf.kern_type)
    else:
        gp = SimpleGPModel.from_dict(model_options)
    
    dt = env.dt
    ctrl_bounds = np.hstack((np.reshape(env.u_min,(-1,1)),np.reshape(env.u_max,(-1,1))))
    mpc_control = SafeMPC(n_safe,n_perf,gp,l_mu,l_sigm,h_mat_safe,h_safe,
                          wx_cost,wu_cost,dt,h_mat_obs = h_mat_obs,h_obs = h_obs,
                          lin_model = lin_model,ctrl_bounds = ctrl_bounds,safe_policy=safe_policy)
    #mpc_control.init_solver()
    
    return mpc_control
    
    
def create_env(conf):
    """ Given a set of options, create an environment """
    print(conf.env_name)
    if conf.env_name == "InvertedPendulum":
        return InvertedPendulum(init_std = conf.init_std)
    else:
        raise NotImplementedError("Unknown environment: {}".format(conf.env_name))
        
        
def get_model_options_from_conf(conf,env):
    """ Utility function to create a gp options dict from the config class"""
    
    if conf.gp_data_path is None:
        return None
        
    gp_dict = dict()
    if conf.lin_prior:
        a,b = env.linearize_discretize()     
    else:
        a,b = (np.eye(env.n_s),np.zeros((env.n_s,env.n_u)))
        
    ab = np.hstack((a,b))
    prior_model = lambda z: np.dot(z,ab.T)
    gp_dict["prior_model"] = prior_model
    
    gp_dict["data_path"] = conf.gp_data_path
    gp_dict["m"] = conf.m
    gp_dict["kern_types"] = conf.kern_types
    gp_dict["hyp"] = conf.gp_hyp
    gp_dict["train"] = conf.train_gp
    
    return gp_dict
    
def loadConfig(conf_path):
    conf_path = abspath(conf_path)
    if not exists(conf_path):
        raise(ValueError("The specified configuration does not exist!"))
    else:
        filename = basename(conf_path)
        module_name = filename.split('.')[0]
        module_name = 'example_configs.' + module_name
        configuration = import_module(module_name)
        conf = configuration.Config()
        #conf_dir = dirname(conf_path)
        #model_path = join(conf_dir, '..', 'models', conf.model_name)
        return conf