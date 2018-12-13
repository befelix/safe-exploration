# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
from default_config import DefaultConfig
from utils_casadi import trig_aug, generic_cost, loss_sat, loss_quadratic,cost_dev_safe_perf 
import numpy as np

class DefaultConfigEpisode(DefaultConfigEpisode):
    """
    Options class for the exploration setting
    """
    verbose = 2

    ## task options
    task = "episode_setting" #don't change this 
    solver_type = "safempc"
    relative_dynamics = False
    
    ##GP
    gp_data_path = None#"random_rollouts_cp.npz"
    gp_ns_in = 3
    gp_ns_out = 4
    gp_nu = 1
    z_sampling_mean = np.array([0.,0.,0.,0.])[None,:]
    z_sampling_std = np.array([1.5,0.3,3.,6])[None,:]

    m = 150 #subset of data of size m for training / number of inducing points for sparse gp
    #z_m = np.matlib.repmat(z_sampling_mean,m,1)
    #z_std = np.matlib.repmat(z_sampling_std,m,1)
    Z = None#np.random.randn(m,gp_ns_in+gp_nu)*z_std + z_m

    kern_types = ["lin_mat52","lin_mat52","lin_mat52","lin_mat52"] #kernel type
    #kern_types = ["rbf","rbf","rbf","rbf"]
    gp_dict_path = None
    gp_hyp = None
    train_gp = True
    lin_trafo_gp_input = np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])


    ##environment
    init_std_initial_data = [1.,1.5,0.2,0.8]
    init_m_initial_data = [-1.0,0.,0.0,0.]
    rl_immediate_cost = None
    env_name = "InvertedPendulum"
    env_name = "CartPole"
    env_options = dict()
    init_std = np.array([.1,.1,0.1,0.1])
    init_m = init_m_initial_data
    env_options["init_std"] = init_std
    env_options["init_m"] = init_m
    env_options["M"] = 0.5
    env_options["verbosity"] = 2
    env_options["norm_x"] = np.array([1.,1.,1.,1.])
    env_options["norm_u"] = np.array([1.])



    ##safempc
    beta_safety = 2.0
    n_safe = 1
    n_perf = 0
    lqr_wx_cost = np.diag([4,8,12,2])# # old working version however with problems in staying in x direction: 
    lqr_wu_cost = 40*np.eye(1) # old working version: 80*np.eye(1)
    lin_prior = True

    prior_model = dict()
    prior_model["m"] = 0.55
    prior_model["M"] = .55
    prior_model["b"] = 0.0
    prior_model["norm_x"] = np.array([1.,1.,1.,1.])
    prior_model["norm_u"] = np.array([1.])
    
    """ Setting that results in  "results_rl_init" Nov. 5th
    prior_model = dict()
    prior_model["m"] = 0.4
    prior_model["b"] = 0.0
    """
    cost_func = None

    # episode settings
    
    init_mode = "safe_samples" #random_rollouts , safe_samples
    n_safe_samples = 25
    c_max_probing_init = 4
    c_max_probing_next_state = 2

    n_ep = 5
    n_steps = 60
    n_steps_init = 8
    n_rollouts_init = 5#5
    n_scenarios = 10
    obs_frequency = 2 # Only take an observation every k-th time step (k = obs_frequency)
    
    #general options
    render = False
    visualize = False
    plot_ellipsoids = False
    plot_trajectory = False
    
    save_results = True
    save_vis = True
    save_dir = None #the directory such that the overall save location is save_path_base/save_dir/
    save_path_base = "results_journal/results_rl" #the directory such that the overall save location is save_path_base/save_dir/
    data_savepath = "gp_data"
    save_name_results = None
    
    def __init__(self,file_path):
        self._generate_save_dir_string()
        super(DefaultConfigEpisode,self).create_savedirs(file_path)

    def _generate_cost(self):
 
        idx_angle = 2
        l = 0.5
        self.env_options["l"] = l
        x_target = 2.5
        z_target = np.array([x_target,0.0,0.0,0.0,l])[:,None] #cart-pos,cart-vel,angle-vel,x_angle,y_angle
        W_target = np.diag([20.0,0.0,0.0,0.0,5.0])/100.
        

        trigon_augm = lambda m,v: trig_aug(m,v,idx_angle)

        ## Saturating cost functions (PILCO)


        cost_stage = lambda m,v,u: loss_sat(m,v,z_target,W_target)
        terminal_cost = lambda m,v: loss_sat(m,v,z_target,W_target)

        ## Quadratic cost functions (standard)
        #cost_stage = lambda m,v,u: loss_quadratic(m,z_target,v,W_target)
        #terminal_cost = lambda m,v: loss_quadratic(m,z_target,v,W_target)


        #cost = lambda p_0,u_0,p_all,q_all,mu_perf,sigma_perf,k_ff_all,k_fb_ctrl,k_fb_perf,k_ff_perf:  \
        #                 generic_cost(mu_perf,sigma_perf,k_ff_perf,cost_stage,terminal_cost,state_trafo = trigon_augm)

        w_rl_cost = 100.0
        self.rl_immediate_cost = lambda state: w_rl_cost*(state[0]- x_target)**2 
        
        if self.n_perf > 0:
        
            cost = lambda p_0,u_0,p_all,q_all,mu_perf,sigma_perf,k_ff_all,k_fb_ctrl,k_fb_perf,k_ff_perf:  \
                         generic_cost(mu_perf,sigma_perf,k_ff_perf,cost_stage,terminal_cost,state_trafo = trigon_augm) #+ \
                         #cost_dev_safe_perf(p_all,mu_perf)
        else:
            cost_stage = lambda m,v,u: loss_quadratic(m,z_target,v,W_target)
            terminal_cost = lambda m,v: loss_quadratic(m,z_target,v,W_target)

            cost = lambda p_0,u_0,p_all,q_all,k_ff_all: \
                        generic_cost(p_all,q_all,k_ff_all,cost_stage,terminal_cost,state_trafo = trigon_augm)



        return cost

    def _generate_save_dir_string(self):
        if self.save_dir is None:
            if self.solver_type == "safempc":
                self.save_dir = self.solver_type+"_"+self.env_name+"_nsafe="+str(self.n_safe)+"_nperf="+str(self.n_perf)+"_r="+str(self.r)+"_beta_safety="+str(self.beta_safety).replace(".","_")
                print(self.save_dir)
            else:
                self.save_dir = self.solver_type+"_"+self.env_name+"_T="+str(self.T)+"_beta_safety="+str(self.beta_safety).replace(".","_")
        
        
