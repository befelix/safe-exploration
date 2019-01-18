# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:42:40 2017

@author: tkoller
"""
from gp_reachability import trajectory_inside_ellipsoid,multistep_reachability
from utils_config import get_prior_model_from_conf
import numpy as np
from casadi import mtimes

def run_uncertainty_propagation(env,safempc,conf):
    """ """
    random_actions = True
    n_rollouts = conf.n_rollouts
    x_all = np.empty((n_rollouts,conf.n_safe+1,env.n_s))
    inside_ellipsoid_per_step = np.empty((n_rollouts,conf.n_safe))
    inside_ellipsoid_all = np.empty((n_rollouts,))
    p_all = np.empty((n_rollouts,conf.n_safe,env.n_s))
    q_all = np.empty((n_rollouts,conf.n_safe,env.n_s,env.n_s))
    #choose constant cost function, since we only care about feasibility
    c_optimize = lambda p_safe,u_safe,p_all,k_ff_all: mtimes(p_all[-1].T,p_all[-1])

    safempc.init_solver(c_optimize)

    for i in range(n_rollouts):

        feasible = False
        if random_actions:
            k_fb = .1*np.random.randn(safempc.n_safe-1,env.n_u,env.n_s)
            k_ff = .1*np.random.randn(safempc.n_safe,env.n_u)
            x_0 = env.reset()
            a_prior, b_prior = get_prior_model_from_conf(conf,env)
            _,_,p_traj_i,q_traj_i = multistep_reachability(x_0[:,None],k_ff[0,:,None],k_fb,k_ff[1:,:],safempc.gp,env.l_mu,env.l_sigm,c_safety = conf.beta_safety,a = a_prior,b=b_prior)
            q_traj_i = q_traj_i.reshape((-1,env.n_s,env.n_s))
        else:
            for j in range(conf.n_restarts_optimizer):
                x_0 = env.reset()

                u_0, feasible, _, k_fb, k_ff, p_traj_i, q_traj_i = safempc.solve(x_0[:,None],env.p_origin[:,None],env.p_origin[:,None],sol_verbose = True)

                if feasible:
                    break

            if not feasible:
                raise ValueError("Couldn't find a feasible solution during {} attempts".format(conf.n_restarts_optimizer))

        p_all[i] = p_traj_i
        q_all[i] = q_traj_i
        x_all[i,0,:] = x_0

        inside_ellipsoid_i = trajectory_inside_ellipsoid(env,x_0,p_traj_i,q_traj_i,k_fb,k_ff)

        inside_ellipsoid_per_step[i,:] = inside_ellipsoid_i
        inside_ellipsoid_all[i] = np.all(inside_ellipsoid_i)

    if conf.save_results:
        save_results(conf.save_path,inside_ellipsoid_per_step,inside_ellipsoid_all,
                     x_all,p_all,q_all)

    print_results(inside_ellipsoid_per_step,inside_ellipsoid_all,conf)

def save_results(save_path,inside_ellipsoid_per_step,
                 inside_ellipsoid_all,x_all,p_all,q_all):
    """ Create a dictionary from the results and save it

    TODO: this is stupid - define functon in utils.py which can take a
    set of k,v from which we can build a dict.
    """

    results_dict = dict()
    results_dict["inside_ellipsoid_per_step"] = inside_ellipsoid_per_step
    results_dict["inside_ellipsoid_all"] = inside_ellipsoid_all
    results_dict["x_all"] = x_all
    results_dict["p_all"] = p_all
    results_dict["q_all"] = q_all
    save_data_path = "{}/res_data".format(save_path)
    np.save(save_data_path,results_dict)

    return results_dict

def print_results(inside_ellipsoid_per_step,inside_ellipsoid_all,conf):
    """ Print results depending on verbosity level """
    if conf.verbosity > 0:
        per_step_percentage = np.sum(inside_ellipsoid_per_step,axis = 0) / float(conf.n_rollouts)
        print("""\n=====Uncertainty propagation===== """)
        print("""Per step percentage of system states inside ellipsoid:""")
        step_str = "Step              "
        perc_str = "Inside Percentage "
        for i in range(conf.n_safe):
            step_str += "| {}  ".format(i)
            perc_str += "| {} ".format(per_step_percentage[i])
        print(step_str)
        print(perc_str)
        print(""" Trajectory safety percentage: {}""".format(float(sum(inside_ellipsoid_all))/conf.n_rollouts))
