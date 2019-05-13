# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""

import numpy as np
from .default_config import DefaultConfig

class DefaultConfigExploration(DefaultConfig):
    """
    Options class for the exploration setting
    """
    # task options
    task = "exploration" #don't change this
    static_exploration = True

    # environment
    env_name = "InvertedPendulum"
    env_options = dict()
    init_std = np.array([.05,.05])
    env_options["init_std"] = init_std
    solver_type = "safempc"

    init_mode = "safe_samples"
    n_safe_samples = 25
    c_max_probing_init = 3
    c_max_probing_next_state = 2

    init_std_initial_data = np.array([.1, .1])
    init_m_initial_data = [0., 0.]

    # safempc
    beta_safety = 2.0
    n_safe = 1

    lqr_wx_cost = np.diag([1.,2.])
    lqr_wu_cost = 25*np.eye(1)
    init_ilqr = True
    str_cost_func = None

    # perf_traj_options
    n_perf = 0
    type_perf_traj = 'taylor'
    r = 1
    perf_has_fb = True

    # prior model: The prior model for the safempc approach
    # can be different from the true model (!)
    lin_prior = True
    prior_model = dict()
    prior_m = .1
    prior_b = 0.0
    prior_model["m"] = prior_m
    prior_model["b"] = prior_b


    # GP
    gp_dict_path = None
    gp_data_path = None # None means no initial training data
    m = 25 #subset of data of size m for training
    kern_types = ["lin_mat52", "lin_mat52"] #kernel type
    train_gp = False # train the gp initially?
    retrain_gp = False # retrain the gp after every sample?
    gp_hyp = None
    Z = None
    lin_trafo_gp_input = None
    gp_ns_in = 2
    gp_ns_out = 2
    relative_dynamics = False ## This should be False
    # gp_ns_in, gp_ns_out = np.shape(lin_trafo_gp_input)

    # Similar setting as in befelix/safe_learning
    #with variances = [(a_true-a.b_true-b)**2]
    kern_dict_0 = dict()
    kern_dict_0["mul.Mat52.variance"] = 1.0
    kern_dict_0["mul.Mat52.lengthscale"] = 1.0
    kern_dict_0["mul.linear.variances"] = 5e-2
    kern_dict_0["linear.variances"] = np.array([  4.74667619e-05,   1.11359543e-05,   4.67080600e-01])

    kern_dict_1 = dict()
    kern_dict_1["mul.Mat52.variance"] = 1.0
    kern_dict_1["mul.Mat52.lengthscale"] = 1.0
    kern_dict_1["mul.linear.variances"] = 1e-3
    kern_dict_1["linear.variances"] = np.array([  2.88698464e-08,   3.05621919e-09,   2.86362642e-04])

    gp_hyp = [kern_dict_0, kern_dict_1]
    #gp_hyp = None
    # exploration
    n_experiments = 1
    n_iterations = 50
    n_restarts_optimizer = 20

    #general options
    verify_safety = False
    visualize = True
    save_results = True
    save_vis = True
    save_dir = None #the directory such that the overall save location is save_path_base/save_dir/
    save_path_base = "results_exploration" #the directory such that the overall save location is save_path_base/save_dir/
    data_savename = None

    def __init__(self, file_path):
        super(DefaultConfigExploration,self).create_savedirs(file_path)