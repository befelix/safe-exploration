# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:28:15 2017

@author: tkoller
"""
import numpy as np
import casadi as cas
import warnings
import sys

from casadi import SX, mtimes, vertcat
from casadi import reshape as cas_reshape

from gp_reachability_casadi import onestep_reachability, lin_ellipsoid_safety_distance, objective

class SafeMPC:
    """ Gaussian Process MPC with safety bounds
    
    
    
    """
    
    def __init__(self, n_safe, n_perf, gp, l_mu, l_sigm, h_mat_safe, h_safe, h_mat_obs, h_obs, wx_cost, wu_cost, p_safety = .95, rhc = True):
        """ Initialize the SafeMPC object with dynamic model information
        
        
        """
        self.rhc = rhc
        self.p_safety = p_safety
         
        self.gp = gp
        
        self.n_safe = n_safe
        self.n_perf = n_perf
        self.n_s = self.gp.n_s
        self.n_u = self.gp.n_u
        self.l_mu = l_mu
        self.l_sigm = l_sigm
        
        m_obs_mat, n_s_obs = np.shape(h_mat_obs)
        m_safe_mat, n_s_safe = np.shape(h_mat_safe)
        assert n_s_obs == self.n_s, " Wrong shape of obstacle matrix"
        assert n_s_safe == self.n_s, " Wrong shape of safety matrix"
        assert np.shape(h_safe) == (m_safe_mat,1), " Shapes of safety linear inequality matrix/vector must match "
        assert np.shape(h_obs) == (m_obs_mat,1), " Shapes of obstacle linear inequality matrix/vector must match "
        
        self.m_obs = m_obs_mat
        self.m_safe = m_safe_mat
        self.h_mat_safe = h_mat_safe
        self.h_safe = h_safe
        self.h_mat_obs = h_mat_obs
        self.h_obs = h_obs
        
        self.wx_cost = wx_cost
        self.wu_cost = wu_cost
        
        self.do_shift_solution = False
        self.solver_initialized = False
        
        self.c_safety = self._c_safety_from_p_safety(None,None,None)
        self.verbosity = 1
        
    def init_solver(self):
        """ Initialize a casadi solver object with safety bounds information
        
        
        """     
        
        k_fb_all = SX.sym("feedback controls", (self.n_safe,self.n_s*self.n_u))
        k_ff_all = SX.sym("feed-forward control",(self.n_safe,self.n_u))
        lbg = []
        ubg = []
        
        p_0 = SX.sym("initial state",(self.n_s,1))
        x_target = SX.sym("target state",(self.n_s,1)) 
        k_fb_0 = k_fb_all[0,:].reshape((self.n_u,self.n_s))
        k_ff_0 = k_ff_all[0,:].reshape((self.n_u,1))
        
        p_new, q_new = onestep_reachability(p_0,self.gp,k_fb_0,k_ff_0,self.l_mu,self.l_sigm,c_safety = self.c_safety)
        g = lin_ellipsoid_safety_distance(p_new,q_new,self.h_mat_obs,self.h_obs)
        
        p_all = p_new.T
        q_all = q_new.reshape((1,self.n_s*self.n_s))
        lbg += [-cas.inf]*self.m_obs
        ubg += [0]*self.m_obs
        
        for i in range(1,self.n_safe):
            k_fb_i = k_fb_all[i,:].reshape((self.n_u,self.n_s))
            k_ff_i = k_ff_all[i,:].reshape((self.n_u,1))
            
            p_new, q_new = onestep_reachability(p_new,self.gp,k_fb_i,k_ff_i,self.l_mu,self.l_sigm,q_new,c_safety = self.c_safety) 
            g_i = lin_ellipsoid_safety_distance(p_new,q_new,self.h_mat_obs,self.h_obs)
            
            g = vertcat(g,g_i)
            lbg += [-cas.inf]*self.m_obs
            ubg += [0]*self.m_obs
            p_all = vertcat(p_all,p_new.T)
            q_all = vertcat(q_all,q_new.reshape((1,self.n_s*self.n_s)))
        
        g_terminal = lin_ellipsoid_safety_distance(p_new,q_new,self.h_mat_safe,self.h_safe)
        g = vertcat(g,g_terminal)
        lbg += [-cas.inf]*self.m_safe
        ubg += [0]*self.m_safe
        
        cost = objective(p_all,q_all,x_target,k_ff_all,self.wx_cost,self.wu_cost)
        opt_vars = vertcat(k_fb_all.reshape((-1,1)),k_ff_all.reshape((-1,1)))
        opt_params = vertcat(p_0,x_target)
        
        prob = {'f':cost,'x': opt_vars,'p':opt_params,'g':g}
        opt = {'ipopt':{'hessian_approximation':'limited-memory'}}
        solver = cas.nlpsol('solver','ipopt',prob,opt)
        
        self.solver = solver
        self.lbg = lbg
        self.ubg = ubg
        self.solver_initialized = True
        
    def solve(self,p_0, p_target, k_ff_all_0 = None, k_fb_all_0 = None, sol_verbose = False):
        """ Solve the MPC problem for a given set of input parameters
        
        
        Parameters
        ----------
        p_0: n_s x 1 array[float]
            The initial (current) state
        p_target n_s x 1 array[float]
            The target state
        k_ff_all_0: n_safe x n_u  array[float], optional
            The initialization of the feed-forward controls
        k_fb_all_0: n_safe x (n_s * n_u) array[float], optional
            The initialization of the feedback controls
            
        Returns
        -------
        k_fb_apply: n_u x n_s array[float]
            The feedback control term to be applied to the system
        k_ff_apply: n_u x 1 array[float]
            The feed-forward control term to be applied to the system
        k_fb_all: n_safe x n_u x n_s
            The feedback control terms for all time steps
        k_ff_all: n_safe x n_u x 1
        """
        assert self.solver_initialized, "Need to initialize the solver first!"
        
        k_ff_all_0, k_fb_all_0 = self._get_init_controls(k_ff_all_0,k_fb_all_0)
        params = np.vstack((p_0,p_target))
        
        u_0 = np.vstack((k_fb_all_0.reshape((self.n_s*self.n_u*self.n_safe,1)), \
                        k_ff_all_0.reshape((self.n_u*self.n_safe,1))))
                        
        sol = self.solver(x0=u_0,lbg=self.lbg,ubg=self.ubg,p=params)
        
        return self._get_solution(sol,sol_verbose)
        
    def _c_safety_from_p_safety(self,p_safety, n_s, n_steps):
        """ Convert a desired safety probability the corresponding safety coefficient """
        warnings.warn("Still need to implement this! Currently returning c_safety = 11.07")
        
        return 11.07
        
    def _get_solution(self,sol, sol_verbose = False):
        """ Process the solution dict of the casadi solver 
        
        Processes the solution dictionary of the casadi solver and 
        (depending on the chosen mode) saves the solution for reuse in the next
        time step. Depending on the chosen verbosity level, it also prints
        some statistics.
        
        Parameters
        ----------
        sol: dict
            The solution dictionary returned by the casadi solver
        sol_verbose: boolean, optional
            Return additional solver results such as the constraint values
            
        Returns
        -------
        k_fb_apply: n_u x n_s array[float]
            The feedback control term to be applied to the system
        k_ff_apply: n_u x 1 array[float]
            The feed-forward control term to be applied to the system
        k_fb_all: n_safe x n_u x n_s
            The feedback control terms for all time steps
        k_ff_all: n_safe x n_u x 1
        
        h_values: (m_obs*n_safe + m_safe) x 0 array[float], optional
            The values of the constraint evaluation (distance to obstacle)
        """
        
        x_opt = sol["x"]

        n_fb = self.n_s*self.n_u*self.n_safe
        x_fb = x_opt[:n_fb]
        x_ff = x_opt[n_fb:]
        
        k_fb_all = cas_reshape(x_fb,(self.n_safe,self.n_s*self.n_u))
        k_ff_all = cas_reshape(x_ff,(self.n_safe,self.n_u))
        
        k_fb_apply = cas_reshape(k_fb_all[0,:],(self.n_u,self.n_s))
        k_ff_apply = cas_reshape(k_ff_all[0,:],(self.n_u,1))
        
        if self.verbosity > 0:
            print("Optimized feed-forward controls:")
            print(k_ff_all)
            print("Optimized feedback controls:")
            print(k_fb_all)
            
        if self.rhc:
            self.k_fb_all = k_fb_all
            self.k_ff_all = k_ff_all
            self.do_shift_solution = True
            
        
        fb_apply_out = np.array(k_fb_apply)
        fb_all_out = np.array(k_fb_all).reshape(self.n_safe,self.n_u,self.n_s)
        ff_apply_out = np.array(k_ff_apply)
        ff_all_out = np.array(k_ff_all)
        if sol_verbose:
            constr_values = np.array(sol["g"])
            return fb_apply_out, ff_apply_out, fb_all_out, ff_all_out, constr_values
        
        return fb_apply_out, ff_apply_out, fb_all_out, ff_all_out
        
    def _get_init_controls(self, k_ff_all_0 = None, k_fb_all_0 = None):
        """ Initialize the controls for the MPC step
        
        """
        if k_ff_all_0 is None:
            if self.do_shift_solution:
                k_ff_old = np.copy(self.k_ff_sol) 
                k_ff_all_0 = np.vstack((k_ff_old[1:,:]),np.zeros((1,self.n_s*self.n_u)))  
            else:
                k_ff_all_0 = .1*np.random.rand(self.n_safe,self.n_u)
            
        if k_fb_all_0 is None:
            if self.do_shift_solution:
                k_fb_old = np.copy(self.k_fb_sol) 
                k_fb_all_0 = np.vstack((k_fb_old[1:,:]),np.zeros((1,self.n_s*self.n_u))) 
            else:
                k_fb_all_0 = .1*np.random.rand(self.n_safe,self.n_s*self.n_u)
                
        return k_ff_all_0, k_fb_all_0 
                
    def update_model(self):
        """ Update the model of the dynamics """
        raise NotImplementedError("Need to implement this")