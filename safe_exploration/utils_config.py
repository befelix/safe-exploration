# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:44:58 2017

@author: tkoller
"""
import warnings
from importlib import import_module
from os.path import abspath, exists, split

import numpy as np

from .cautious_mpc import CautiousMPC
from .environments import InvertedPendulum, CartPole
from .gp_models import SimpleGPModel
from .safempc_simple import SimpleSafeMPC
from .utils import dlqr


def create_solver(conf, env, model_options=None):
    """ Create a solver from a set of options and environment information"""

    lin_model = None
    lin_trafo_gp_input = conf.lin_trafo_gp_input

    h_mat_safe, h_safe, h_mat_obs, h_obs = env.get_safety_constraints(normalize=True)

    warnings.warn("Normalization of constraints may be wrong!")

    a_true, b_true = env.linearize_discretize()

    safe_policy = None
    if conf.lin_prior:
        a_prior, b_prior = get_prior_model_from_conf(conf, env)
        lin_model = (a_prior, b_prior)
        # by default takes identity as prior. Need to define safe_policy
    wx_cost = conf.lqr_wx_cost
    wu_cost = conf.lqr_wu_cost

    q = wx_cost
    r = wu_cost
    k_lqr, _, _ = dlqr(a_true, b_true, q, r)
    k_fb = -k_lqr
    safe_policy = lambda x: np.dot(k_fb, x)

    if model_options is None:
        gp = SimpleGPModel(conf.gp_ns_out, conf.gp_ns_in, env.n_u, m=conf.m,
                           kern_types=conf.kern_types, Z=conf.Z)
    else:
        gp = SimpleGPModel.from_dict(model_options)

    dt = env.dt
    ctrl_bounds = np.hstack(
        (np.reshape(env.u_min, (-1, 1)), np.reshape(env.u_max, (-1, 1))))

    env_opts_safempc = dict()

    env_opts_safempc["h_mat_safe"] = h_mat_safe
    env_opts_safempc["h_safe"] = h_safe
    env_opts_safempc["lin_model"] = lin_model
    env_opts_safempc["ctrl_bounds"] = ctrl_bounds
    env_opts_safempc["safe_policy"] = safe_policy
    env_opts_safempc["h_mat_obs"] = h_mat_obs
    env_opts_safempc["h_obs"] = h_obs
    env_opts_safempc["dt"] = env.dt
    env_opts_safempc["lin_trafo_gp_input"] = lin_trafo_gp_input

    if conf.solver_type == "safempc":
        # the environment options needed for the safempc algorithm
        n_safe = conf.n_safe
        n_perf = conf.n_perf
        l_mu = env.l_mu
        l_sigm = env.l_sigm

        env_opts_safempc["l_mu"] = env.l_mu
        env_opts_safempc["l_sigma"] = env.l_sigm

        perf_opts_safempc = dict()
        perf_opts_safempc["type_perf_traj"] = conf.type_perf_traj
        perf_opts_safempc["n_perf"] = conf.n_perf
        perf_opts_safempc["r"] = conf.r
        perf_opts_safempc["perf_has_fb"] = conf.perf_has_fb

        solver = SimpleSafeMPC(n_safe, gp, env_opts_safempc, wx_cost, wu_cost,
                               beta_safety=conf.beta_safety, lin_model=lin_model,
                               safe_policy=safe_policy,
                               opt_perf_trajectory=perf_opts_safempc,
                               lin_trafo_gp_input=lin_trafo_gp_input)
    elif conf.solver_type == "cautious_mpc":
        T = conf.T

        solver = CautiousMPC(T, gp, env_opts_safempc, conf.beta_safety,
                             lin_trafo_gp_input=lin_trafo_gp_input, k_fb=k_fb)
    else:
        raise ValueError("Unknown solver type: {}".format(conf.solver_type))

    # mpc_control = SafeMPC(n_safe, gp, env_opts_safempc, wx_cost, wu_cost,beta_safety = conf.beta_safety,
    #             ilqr_init = conf.ilqr_init, lin_model = lin_model, ctrl_bounds = ctrl_bounds,
    #             safe_policy = safe_policy, opt_perf_trajectory = perf_opts_safempc,lin_trafo_gp_input = lin_trafo_gp_input)

    return solver, safe_policy


def create_env(env_name, env_options_dict=None):
    """ Given a set of options, create an environment """
    if env_options_dict is None:
        env_options_dict = dict()
    if env_name == "InvertedPendulum":
        return InvertedPendulum(**env_options_dict)
    elif env_name == "CartPole":
        return CartPole(**env_options_dict)
    else:
        raise NotImplementedError("Unknown environment: {}".format(conf.env_name))


def get_prior_model_from_conf(conf, env_true):
    """ Get prior model from config"""
    if conf.lin_prior:
        # unless specified otherwise, use the same normalization as for the true model
        if not "norm_x" in conf.prior_model:
            conf.prior_model["norm_x"] = env_true.norm[0]
        if not "norm_u" in conf.prior_model:
            conf.prior_model["norm_u"] = env_true.norm[1]

        env_prior = create_env(conf.env_name, conf.prior_model)

        a_prior, b_prior = env_prior.linearize_discretize()

    else:
        a_prior, b_prior = (np.eye(env_true.n_s), np.zeros((env_true.n_s, env.n_u)))

    return a_prior, b_prior


def get_model_options_from_conf(conf, env):
    """ Utility function to create a gp options dict from the config class"""

    # There already is a gp_dict ready to use
    if not conf.gp_dict_path is None:
        return np.load(conf.gp_dict_path)

    # neither a gp_dict_path nor a gp_data_path exists -> return None
    gp_dict = dict()

    a_prior, b_prior = get_prior_model_from_conf(conf, env)

    ab_prior = np.hstack((a_prior, b_prior))
    prior_model = lambda z: np.dot(z, ab_prior.T)
    gp_dict["prior_model"] = prior_model

    gp_dict["data_path"] = conf.gp_data_path
    gp_dict["m"] = conf.m
    gp_dict["kern_types"] = conf.kern_types
    gp_dict["hyp"] = conf.gp_hyp
    gp_dict["train"] = conf.train_gp
    gp_dict["n_s_in"] = conf.gp_ns_in
    gp_dict["n_s_out"] = conf.gp_ns_out
    gp_dict["n_u"] = conf.gp_nu
    gp_dict["Z"] = conf.Z

    return gp_dict


def loadConfig(conf_path):
    if not exists(abspath(conf_path)):
        raise ValueError
    else:

        file_path, filename = split(conf_path)
        module_name = filename.split('.')[0]
        module_name_path = file_path + '.' + module_name
        configuration = import_module(module_name_path)
        conf = configuration.Config()

        return conf
