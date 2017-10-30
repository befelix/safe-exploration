# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:28:15 2017

@author: tkoller
"""
import numpy as np
import casadi as cas
import warnings
import sys
import scipy

from casadi import SX, mtimes, vertcat
from casadi import reshape as cas_reshape

from gp_reachability_casadi import onestep_reachability, lin_ellipsoid_safety_distance, objective
from gp_reachability import multistep_reachability
class SafeMPC:
    """ Gaussian Process MPC with safety bounds
    
    
    
    """
    
    def __init__(self, n_safe, n_perf, gp, l_mu, l_sigm, h_mat_safe, h_safe,
                 h_mat_obs, h_obs, wx_cost, wu_cost, p_safety = .95, 
                 rhc = True, lqr_feedback = True, lin_model = None, ctrl_bounds = None):
        """ Initialize the SafeMPC object with dynamic model information
        
        
        """
        self.rhc = rhc
        self.p_safety = p_safety 
        self.gp = gp
        self.n_safe = n_safe
        self.n_perf = n_perf
        self.n_fail = self.n_safe + 1 #initialize s.t. there is no backup strategy
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
        
        self.has_ctrl_bounds = False        
        if not ctrl_bounds is None:
            self.has_ctrl_bounds = True
            assert np.shape(ctrl_bounds) == (self.n_u,2), """control bounds need 
            to be of shape n_u x 2 with i,0 lower bound and i,1 upper bound per dimension"""
            self.ctrl_bounds = ctrl_bounds
            
        self.m_obs = m_obs_mat
        self.m_safe = m_safe_mat
        self.h_mat_safe = h_mat_safe
        self.h_safe = h_safe
        self.h_mat_obs = h_mat_obs
        self.h_obs = h_obs
        
        self.wx_cost = wx_cost
        self.wu_cost = wu_cost
        self.wx_feedback = wx_cost/3
        self.wu_feedback = 15*wu_cost
        
        self.do_shift_solution = False
        self.solver_initialized = False
        
        self.c_safety = self._c_safety_from_p_safety(None,None,None)
        self.verbosity = 2
        self.lqr_feedback = lqr_feedback
        self.lin_prior = False
        self.a = None
        self.b = None
        if not lin_model is None:
            self.a,self.b = lin_model
            self.lin_prior = True
            
        if self.gp.gp_trained:
            self.init_solver()
            
    def init_solver(self):
        """ Initialize a casadi solver object with safety bounds information
        
        TODO: First feedback control unnecessary - remove it asap
        
        """     
        if self.lqr_feedback:
            k_fb_all = SX.sym("feedback controls", (1,self.n_s*self.n_u))
        else:
            k_fb_all = SX.sym("feedback controls", (self.n_safe-1,self.n_s*self.n_u))
            
        k_ff_all = SX.sym("feed-forward control",(self.n_safe,self.n_u))
        lbg = []
        ubg = []
        
        p_0 = SX.sym("initial state",(self.n_s,1))
        x_target = SX.sym("target state",(self.n_s,1)) 
        
        k_ff_0 = k_ff_all[0,:].reshape((self.n_u,1))
        
                
        p_new, q_new = onestep_reachability(p_0,self.gp,k_ff_0,self.l_mu,self.l_sigm,c_safety = self.c_safety,a=self.a,b=self.b)
        g = lin_ellipsoid_safety_distance(p_new,q_new,self.h_mat_obs,self.h_obs)
        lbg += [-cas.inf]*self.m_obs
        ubg += [0]*self.m_obs
        
        if self.has_ctrl_bounds:
                g_u_i, lbu_i, ubu_i = self._generate_control_constraint(k_ff_0)
                g = vertcat(g,g_u_i)
                lbg += lbu_i
                ubg += ubu_i
                
        p_all = p_new.T
        q_all = q_new.reshape((1,self.n_s*self.n_s))
        
        
        for i in range(1,self.n_safe):
            p_old = p_new
            q_old = q_new
                
            if self.lqr_feedback:
                k_fb_i = k_fb_all.reshape((self.n_u,self.n_s))
            else:
                k_fb_i = k_fb_all[i-1,:].reshape((self.n_u,self.n_s))
                
            k_ff_i = k_ff_all[i,:].reshape((self.n_u,1))
            if self.has_ctrl_bounds:
                g_u_i, lbu_i, ubu_i = self._generate_control_constraint(k_ff_i,q_old,k_fb_i)
                g = vertcat(g,g_u_i)
                lbg += lbu_i
                ubg += ubu_i
                 
            p_new, q_new = onestep_reachability(p_old,self.gp,k_ff_i,self.l_mu,self.l_sigm,q_old,k_fb_i,c_safety = self.c_safety,a=self.a,b=self.b) 
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
        
        
        if self.lqr_feedback:
            opt_vars = vertcat(k_ff_all.reshape((-1,1)))
            opt_params = vertcat(p_0,x_target,k_fb_all.T)
        else:
            opt_vars = vertcat(k_fb_all.reshape((-1,1)),k_ff_all.reshape((-1,1)))
            opt_params = vertcat(p_0,x_target)
        
        prob = {'f':cost,'x': opt_vars,'p':opt_params,'g':g}
        #opt = {'ipopt':{'hessian_approximation':'limited-memory',"max_iter":30}}
        opt = {'qpsol','hessian_approximation':'limited-memory',"max_iter":30}
        solver = cas.nlpsol('solver','sqpmethod',prob,opt)
        
        self.solver = solver
        self.lbg = lbg
        self.ubg = ubg
        self.solver_initialized = True
    
    def _generate_control_constraint(self,k_ff,q = None,k_fb = None,ctrl_bounds = None):
        """ Build control constraints from state ellipsoids and linear feedback controls
        
        k_ff: n_u x 1 ndarray[casadi.SX]
            The feed-forward control gain
        q: n_s x n_s ndarray[casadi.SX]
            The shape matrix of the state ellipsoid
        k_fb: n_u x n_s ndarray[casadi.SX]
            The feedback gain
        ctrl_bounds: n_u x 2 ndarray[float], optional
        
        Returns
        -------
        g: 2*n_u x 1 ndarray[casadi.SX]
            The control constraints (symbollicaly) evaluated at the current
            state/controls
        lbg: 2*n_u x 0 list[float]
            Lower bounds for the control constraints
        ubg: 2*n_u x 0 list[float]
            Upper bounds for the control constraints
        """        
        if ctrl_bounds is None:
            if not self.has_ctrl_bounds:
                raise ValueError("""Either ctrl_bounds has to be specified or 
                the objects' ctrl_bounds has to be specified """)
            ctrl_bounds = self.ctrl_bounds
        
        #no feedback term. Reduces to simple feed-forward control bounds
        
        n_u,_ = np.shape(ctrl_bounds)
        u_min = ctrl_bounds[:,0]
        u_max = ctrl_bounds[:,1]
        
        if k_fb is None:
            return k_ff, u_min.tolist(), u_max.tolist()
            
        h_vec = np.vstack((u_max[:,None],-u_min[:,None]))
        h_mat = np.vstack((np.eye(n_u),-np.eye(n_u)))
        
        p_u = k_ff
        q_u = mtimes(k_fb,mtimes(q,k_fb.T))
        
        g = lin_ellipsoid_safety_distance(p_u,q_u,h_mat,h_vec)
        
        return g, [-cas.inf]*2*n_u, [0]*2*n_u

    def _eval_prior_casadi(self, state, action):
        """ symbolically evaluate the prior 
        
        Parameters
        ----------
        state: n x n_s array[casadi.SX]
            Symbolic array of states
        action: n x 1 array[casadi.SX]
            Symbolic array of actions
            
        Returns
        -------
        x_prior: n x n_s array[casadi.SX]
            The (state,action) pairs evaluated at the prior
        """
        if self.lin_prior:
            return mtimes(self.a,state.T) + mtimes(self.b,action.T)
        else:
            return state
        
    def eval_prior(self, state, action):
        """ Evaluate the prior numerically 
        
        Parameters
        ----------
        state: n x n_s array[float]
            Array of states 
        action: n x n_u array[float]       
        
        Returns
        -------
        x_prior: n x n_s array[float]
            The (state,action) pairs evaluated at the prior
        """
        if self.lin_prior:
            return np.dot(state,self.a.T) + np.dot(action,self.b.T)
        else:
            return state
        
    def dlqr(self,a,b,q,r):
        """ Get the feedback controls from linearized system at the current time step
        
        for a discrete time system Ax+Bu
        find the infinite horizon optimal feedback controller
        to steer the system to the origin
        with
        u = -K*x 
        """
        x = np.matrix(scipy.linalg.solve_discrete_are(a, b, q, r))
     
        k = np.matrix(scipy.linalg.inv(b.T*x*b+r)*(b.T*x*a))
     
        eigVals, eigVecs = scipy.linalg.eig(a-b*k)
        
        return k, x, eigVals
        
    def get_lqr_feedback(self,x_0 = None, u_0 = None):
        """ Get the initial feedback controller k_fb
        
        x_0: n_s x 1 ndarray[float], optional
        u_0: n_u x 1 ndarray[float], optional
        
        """
        q = self.wx_cost
        r = self.wu_cost
        
        if x_0 is None:
            x_0 = np.zeros((self.n_s,1))
        if u_0 is None:
            u_0 = np.zeros((self.n_u,1))
            
        if self.lin_prior:
            a = self.a
            b= self.b
            
            k_fb,_,_ = self.dlqr(a,b,q,r)
            k_fb = -k_fb
        else:
            z = np.hstack((x_0.T,u_0.T))
            jac_mu = self.gp.predictive_gradients(z)
            mu_new,_ = self.gp.predict(z)
            
            a = jac_mu[0,:,:self.n_s] + np.eye(self.n_s)
            b = jac_mu[0,:,self.n_s:]
            a_aug = np.vstack((np.hstack((a,mu_new.T-np.dot(b,u_0)-np.dot(a,x_0))),
                               np.hstack((np.zeros((1,self.n_s)),np.zeros((1,1))))))
            b_aug = np.vstack((b,np.zeros((1,self.n_u))))
            
            q = self.wx_cost
            r = self.wu_cost
            q_aug = np.vstack((np.hstack((q,np.zeros((self.n_s,1)))),np.hstack((np.zeros((1,self.n_s)),np.eye(1)))))                
            
            k_fb_aug,_,_ = self.dlqr(a_aug,b_aug,q_aug,r)
            k_fb_aug = -k_fb_aug #mind that our control is defined as u = K*x and not u = -K*x
            k_fb_aug = np.reshape(k_fb_aug,(self.n_s+1,self.n_u))
            k_fb = k_fb_aug[:self.n_s,:]
            
        return k_fb.reshape((1,self.n_s*self.n_u))
        
        
    def get_trajectory_openloop(self,x_0 , k_fb = None, k_ff = None, get_controls = False):
        """ Plan a trajectory based on an initial state and a set of controls
        
        Parameters
        ----------
        x_0: n_s x 0 1darray[float] 
            The initial state
        k_fb: T x n_u x n_s  or n_u x n_s ndarray[float], optional
            The feedback controls. Uses the most recent solution to the 
            MPC Problem (when calling solve()) if this parameter is not set
        k_ff: T x n_u, optional
            The feed-forward controls. Uses the most recent solution to the 
            MPC Problem (when calling solve()) if this parameter is not set
        get_controls: bool, optional
            Additionally returns the applied controls if this flag is set to TRUE
        Returns
        -------
        p_all: T x n_s ndarray[float]
            The centers of the trajctory ellipsoids
        q_all: T x n_s x n_s ndarray[float]
            The shape matrices of the trajectory ellipsoids
        """
        if k_fb is None:
            k_fb = np.array(self.k_fb_all)
        if k_ff is None:
            k_ff = np.array(self.k_ff_all)
            
        T, n_u = np.shape(k_ff)
        if k_fb.ndim ==2:
            _, n_s = np.shape(k_fb)
            k_tmp = k_fb
            k_fb = np.empty((T,n_u,n_s))
            for i in range(T):
                k_fb[i] = k_tmp
        _,_,p_all, q_all = multistep_reachability(x_0[:,None],self.gp,k_fb,k_ff,
                                              self.l_mu,self.l_sigm,c_safety = self.c_safety, verbose=0,a =self.a,b=self.b)
        if get_controls:
            return p_all, q_all, k_fb, k_ff
      
          
        return p_all, q_all
        
    def get_action(self,x0_mu, target, x0_sigm = None):
        """ Wrapper around the solve function 
        
        Parameters
        ----------
        x0_mu: n_s x 0 1darray[float]
            The current state of the system
        target: n_s x 0 1darray[float]        
            The target state of the system
        
        Returns
        -------
        u_apply: n_u x 0 1darray[float]
            The action to be applied to the system
            
        """        
        k_fb, k_ff, _, _, success = self.solve(x0_mu[:,None],target[:,None])
        
        safety_failure = False
            
        return k_ff.reshape(self.n_u,), safety_failure
        
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
        
        
        k_ff_all_0, k_fb_all_0 = self._get_init_controls(p_0,k_ff_all_0,k_fb_all_0)
        params = np.vstack((p_0,p_target))
        
        if self.lqr_feedback:
            u_0 = k_ff_all_0.reshape((self.n_u*self.n_safe,1))    
            params = np.vstack((params,cas_reshape(k_fb_all_0,(self.n_u*self.n_s,1))))
        else:
            u_0 = np.vstack((k_fb_all_0.reshape((self.n_s*self.n_u*self.n_safe,1)), \
                        k_ff_all_0.reshape((self.n_u*self.n_safe,1))))
        sol = self.solver(x0=u_0,lbg=self.lbg,ubg=self.ubg,p=params)
        
        return self._get_solution(sol,k_fb_all_0,sol_verbose)
        
    def _c_safety_from_p_safety(self,p_safety, n_s, n_steps):
        """ Convert a desired safety probability the corresponding safety coefficient """
        warnings.warn("Still need to implement this! Currently returning c_safety = 2.0")
        
        return 2.0
        
    def _get_solution(self,sol, k_fb_all_0 = None, sol_verbose = False):
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
        
        print(sol)
        success = True
        
        if self.lqr_feedback:
            x_ff = sol["x"]
            k_ff_all = cas_reshape(x_ff,(self.n_safe,self.n_u))
            k_ff_apply = cas_reshape(k_ff_all[0,:],(self.n_u,1))
            k_fb_all = np.reshape(k_fb_all_0,(self.n_u,self.n_s))
            k_fb_apply = k_fb_all            
            
            if self.rhc:
                self.k_fb_all = k_fb_all
                self.k_ff_all = k_ff_all
        else:
            x_opt = sol["x"]
            n_fb = self.n_s*self.n_u*self.n_safe  
            x_fb = x_opt[:n_fb]
            x_ff = x_opt[n_fb:]
            k_fb_all = cas_reshape(x_fb,(self.n_safe,self.n_s*self.n_u))
            k_ff_all = cas_reshape(x_ff,(self.n_safe,self.n_u))
            
            k_fb_apply = cas_reshape(k_fb_all[0,:],(self.n_u,self.n_s))
            k_ff_apply = cas_reshape(k_ff_all[0,:],(self.n_u,1))
            k_fb_all = np.array(k_fb_all).reshape(self.n_safe,self.n_u,self.n_s)
                
            if self.rhc:
                self.k_fb_all = k_fb_all
                self.k_ff_all = k_ff_all
                self.do_shift_solution = True
                
        fb_apply_out = np.array(k_fb_apply)
        fb_all_out = np.array(k_fb_all)
        ff_apply_out = np.array(k_ff_apply)
        ff_all_out = np.array(k_ff_all)
            
        if self.verbosity > 0:
                print("Optimized feed-forward controls:")
                print(k_ff_all)
                print("LQR feedback controls:")
                print(k_fb_all) 
                
                if self.verbosity > 1:
                    print("\n===Constraint values:===")
                    print(sol["g"])
                    print("==========================\n")
                    
        if sol_verbose:
            constr_values = np.array(sol["g"])
            return fb_apply_out, ff_apply_out, fb_all_out, ff_all_out, constr_values, success
        return fb_apply_out, ff_apply_out, fb_all_out, ff_all_out, success
        
    def _get_init_controls(self,x_0, k_ff_all_0 = None, k_fb_all_0 = None):
        """ Initialize the controls for the MPC step
        
        """
        if k_ff_all_0 is None:
            if self.do_shift_solution:
                k_ff_old = np.copy(self.k_ff_sol) 
                k_ff_all_0 = np.vstack((k_ff_old[1:,:]),np.zeros((1,self.n_s*self.n_u)))  
            else:
                k_ff_all_0 = .1*np.random.rand(self.n_safe,self.n_u)
            
        if self.lqr_feedback:
            if k_fb_all_0 is None:
                k_fb_all_0 = self.get_lqr_feedback()
                
            return k_ff_all_0, k_fb_all_0
        else:
            if k_fb_all_0 is None:
                if self.do_shift_solution:
                    k_fb_old = np.copy(self.k_fb_sol) 
                    k_fb_all_0 = np.vstack((k_fb_old[1:,:]),np.zeros((1,self.n_s*self.n_u))) 
                else:
                    k_fb_all_0 = .1*np.random.rand(self.n_safe,self.n_s*self.n_u)
                    
            return k_ff_all_0, k_fb_all_0 
                
    def update_model(self, x, y, train = True):
        """ Update the model of the dynamics 
        
        Parameters
        ----------
        x: n x (n_s+n_u) array[float]
            The raw training input (state,action) pairs
        y: n x (n_s) array[float]
            The raw training targets
        """
        n_train = np.shape(x)[0]
        x_s = x[:,:self.n_s].reshape((n_train,self.n_s))
        x_u = x[:,self.n_s:].reshape((n_train,self.n_u))
        y_prior = self.eval_prior(x_s,x_u)
        
        self.gp.update_model(x,y-y_prior,train)
        self.init_solver()