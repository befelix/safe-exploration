# -*- coding: utf-8 -*-
"""
Implements different exploration oracles which provide informative samples
or exploration objectives to be used in a MPC setting.

Created on Tue Nov 14 10:08:45 2017

@author: tkoller
"""

from casadi import *
from casadi.tools import *
from casadi import reshape as cas_reshape
from gp_reachability_casadi import multi_step_reachability, onestep_reachability

class MPCExplorationOracle:
    """ Oracle which finds informative samples
    similar to the GP-UCB setting but with safety constraint and MPC setting
    
    Attributes
    ----------
    safempc: SafeMPC
        The underlying safempc which provides necessary information
        such as constraints, prior model and GP
    
    """
    
    
    def __init__(self,safempc, env):
        """ Initialize with a pre-defined safempc object"""
        self.safempc = safempc
        self.env = env
        self.n_s = safempc.n_s 
        self.n_u = safempc.n_u
        self.T = safempc.n_safe
        
    def init_solver(self,T = None):
        """ Generate the exloration NLP in casadi
        
        Parameters
        ----------
        T: int, optional
            the safempc horizon
        
        """
        if T is None:
            T = self.T
        else:
            self.T = T
        
        x_0 = SX.sym("x_0",(self.n_s,1))
        u_0 = SX.sym("u_0",(self.n_u,1))
        gp = self.safempc.gp
        l_mu = self.safempc.l_mu
        l_sigma = self.safempc.l_sigma
        beta_safety = self.safempc.beta_safety
        a = self.safempc.a
        b = self.safempc.b
        
        if T > 1:
            k_fb_0 = SX.sym("k_fb_0",(T-1,self.n_s*self.n_u))        
            k_ff = SX.sym("k_ff",(T-1,self.n_u))
            k_fb_ctrl = SX.sym("k_fb_ctrl",(self.n_u,self.n_s))
            
            p_all, q_all = multi_step_reachability(x_0,u_0,k_fb_0,k_fb_ctrl,
                                                   k_ff,gp,l_mu,l_sigma,beta_safety,a,b)
            
            ## generate constraints
            g_safe, lbg_safe, ubg_safe, _ = self.safempc.generate_safety_constraints(p_all,q_all,u_0,
                                                                                k_fb_0,k_fb_ctrl,k_ff)
                                                                                
            #stack variables and parameters
            opt_vars = vertcat(x_0,u_0,k_ff.reshape((-1,1)),k_fb_ctrl.reshape((-1,1)))
            opt_params = vertcat(k_fb_0.reshape((-1,1)))
        else:
            p_new ,q_new = onestep_reachability(x_0,gp,u_0,l_mu,l_sigma,a=a,b=b,c_safety = beta_safety)
            g_safe, lbg_safe, ubg_safe, _ = self.safempc.generate_safety_constraints(p_new.T,q_new.reshape((1,self.n_s*self.n_s)),u_0,
                                                                                None,None,None)
            opt_vars = vertcat(x_0,u_0)
            opt_params = []
        g = g_safe
        lbg = lbg_safe
        ubg = ubg_safe
        
        ## generate the exploration objective function
        _,sigm = gp.predict_casadi_symbolic(vertcat(x_0,u_0).T)
        c = -sum2(sqrt(sigm)) 
        
        
        
        
        prob = {'f':c,'x': opt_vars,'p':opt_params,'g':g}
        opt = {'ipopt':{'hessian_approximation':'limited-memory',"max_iter":60,"expect_infeasible_problem":"yes"}} #ipopt 
       
        solver = nlpsol("solver","ipopt",prob,opt)
        
        self.solver = solver
        self.lbg = lbg
        self.ubg = ubg
        self.T = T
        
    def find_max_variance(self, n_restarts = 1,ilqr_init = False, 
                          sample_mean = None, 
                          sample_var = None, 
                          verbosity = 1, beta_safety = None):
        """ Find the most informative sample in the space constrained by the mpc structure
        
        Parameters
        ----------
        n_restarts: int, optional
            The number of random initializations of the optimization problem
        ilqr_init: bool, optional
            initialize the state feedback terms with the ilqr feedback law
        sample_mean: n_s x 1 np.ndarray[float], optional
            The mean of the gaussian initial state-action distribution
        sample_var: n_s x n_s np.ndarray[float], optional
            The variance of the gaussian initial state-action distribution
            
        Returns
        -------
        x_opt:
        u_opt:
        sigm_opt:
                
        """
        
        if ilqr_init:
            self.safempc.init_ilqr_initializer()
        
        sigma_best = 0
        x_best = None
        u_best = None
        
        for i in range(n_restarts):
            
            x_0 = self.env._sample_start_state(sample_mean,sample_var)[:,None] # sample initial state
            
            if self.T > 1:
                if ilqr_init:
                    u_0, k_fb_0, k_ff_0 = self.safempc.init_ilqr(x_0,self.env.p_origin,self.env.p_origin)
                    k_fb_ctrl_0 = np.zeros((self.n_s*self.n_u,1))
                    k_fb_0 = cas_reshape(k_fb_0,(-1,1))
                else:
                    u_0 = self.env.random_action()[:,None]
                    k_fb_0 = np.zeros(((self.T-1)*self.n_s*self.n_u,1))
                    k_ff_0 = np.zeros(((self.T-1)*self.n_u,1))
                    k_fb_ctrl_0 = np.random.rand(self.n_u*self.n_s,1)
                params_0 = k_fb_0
                vars_0 = np.vstack((x_0,u_0,k_ff_0,k_fb_ctrl_0)) 
            else:
                u_0 = self.env.random_action()[:,None]
                params_0 = []
                vars_0 = np.vstack((x_0,u_0))
            
            sol = self.solver(x0 = vars_0,p=params_0,lbg = self.lbg, ubg = self.ubg)
            
            f_opt = sol["f"]

            sigm_i = -float(f_opt)
            if sigm_i > sigma_best: # check if solution would improve upon current best
                g_sol = np.array(sol["g"]).squeeze()
                if self._is_feasible(g_sol, np.array(self.lbg), np.array(self.ubg)): #check if solution is feasible
                    w_sol = sol["x"]
                    x_best = np.array(w_sol[:self.n_s])
                    u_best = np.array(w_sol[self.n_s:self.n_s+self.n_u])
                    sigma_best = sigm_i

                    z_i = np.vstack((x_best,u_best)).T
                    if verbosity > 0:
                        print("new optimal sigma found at iteration {}".format(i))
                        
        return x_best, u_best, sigma_best
        
    def update_model(self,x,y,train = False,replace_old = False):
        """ Simple wrapper around the update_model function of SafeMPC"""
        self.safempc.update_model(x,y,train,replace_old)
        
    def get_information_gain(self):
        return self.safempc.gp.information_gain()
        
    def _is_feasible(self,g, lbg, ubg, feas_tol = 1e-7):
        """ """
        return np.all(g > lbg - feas_tol  ) and np.all(g < ubg + feas_tol )
            