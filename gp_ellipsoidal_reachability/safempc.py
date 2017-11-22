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

from casadi import SX, mtimes, vertcat, sum2, sqrt
from casadi import reshape as cas_reshape
from gp_reachability_casadi import multi_step_reachability as cas_multistep

from gp_reachability_casadi import onestep_reachability, lin_ellipsoid_safety_distance, objective

from gp_reachability import multistep_reachability

from ilqr_cython import CILQR

class SafeMPC:
    """ Gaussian Process MPC with safety bounds
    
    
    
    """
    
    def __init__(self, n_safe, n_perf, gp, l_mu, l_sigma, h_mat_safe, h_safe,
                 h_mat_obs, h_obs, wx_cost, wu_cost, dt,beta_safety = 2.5,
                 rhc = True, ilqr_init = True, lin_model = None, ctrl_bounds = None,
                 safe_policy = None):
        """ Initialize the SafeMPC object with dynamic model information
        
        
        """ 
        self.rhc = rhc
        self.gp = gp
        self.n_safe = n_safe
        self.n_perf = n_perf
        self.n_fail = self.n_safe #initialize s.t. there is no backup strategy
        self.n_s = self.gp.n_s
        self.n_u = self.gp.n_u
        self.l_mu = l_mu
        self.l_sigma = l_sigma
        self.dt = dt
        self.has_openloop = False
        
        self.safe_policy = safe_policy
        
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
        self.wx_feedback = wx_cost
        self.wu_feedback = 1*wu_cost
        
        self.do_shift_solution = True
        self.solver_initialized = False
        
        self.beta_safety = beta_safety
        self.verbosity = 2
        self.ilqr_init = ilqr_init
        
        
        self.lin_prior = False
        self.a = np.zeros((self.n_s,self.n_s))
        self.b = np.zeros((self.n_s,self.n_u))
        if not lin_model is None:
            self.a,self.b = lin_model
            self.lin_prior = True
            if self.safe_policy is None:
                #no safe policy specified? Use lqr as safe policy
                K = self.get_lqr_feedback()
                self.safe_policy = lambda x: np.dot(K,x)
                
        self.k_fb_all = None
        if self.safe_policy is None:
            warnings.warn("No SafePolicy!")
        if self.gp.gp_trained:
            self.init_solver()     
        if ilqr_init:
            self.init_ilqr_initializer()
            
    def init_solver(self):
        """ Initialize a casadi solver object with safety bounds information
        
        """     
        x_cstr_scaling = 1        
        
        k_fb_ctrl = SX.sym("feedback controls", (1,self.n_s*self.n_u))

            
        u_0 = SX.sym("init_control",(self.n_u,1))
        k_ff_all = SX.sym("feed-forward control",(self.n_safe-1,self.n_u))
        g = []
        lbg = []
        ubg = []
        g_name = []
        
        p_0 = SX.sym("initial state",(self.n_s,1))
        x_target = SX.sym("target state",(self.n_s,1)) 
        
        k_fb_0 = SX.sym("base feedback matrices",(self.n_safe-1,self.n_s*self.n_u))
        
         
        p_all, q_all = cas_multistep(p_0,u_0,k_fb_0,k_fb_ctrl,k_ff_all,self.gp,self.l_mu,self.l_sigma,self.beta_safety,self.a,self.b)
        
        g_safe, lbg_safe,ubg_safe, g_names_safe = self.generate_safety_constraints(p_all,q_all,u_0,k_fb_0,k_fb_ctrl,k_ff_all)
        g = vertcat(g,g_safe)
        lbg += lbg_safe
        ubg += ubg_safe
        g_name += g_names_safe
        
        u_perf = []
        if self.n_perf > 1:
            u_perf = SX.sym("u_perf",(self.n_perf-1,self.n_u))
            x_perf = p_all[0,:]
            x_perf_new = x_perf.T
            for i in range(self.n_perf-1):
                u_i = u_perf[i,:].T
                mu_pred, _ = self.gp.predict_casadi_symbolic(cas.horzcat(x_perf_new.T,u_i.T)) 
                x_perf_new = mtimes(self.a,x_perf_new) + mtimes(self.b,u_perf[i,:].T) + mu_pred.T
                x_perf = vertcat(x_perf,x_perf_new.T)
                
                if self.has_ctrl_bounds:
                    g_u_i, lbu_i, ubu_i = self._generate_control_constraint(u_i)
                    g = vertcat(g,g_u_i)
                    lbg += lbu_i
                    ubg += ubu_i
                    g_name += ["ctrl_constr_performance_{}".format(i)]
            u_perf = u_perf.reshape((-1,1))
        
        cost = 0
        if self.n_perf > 1:
            for i in range(self.n_perf):
                cost += mtimes((x_perf[i,:].T-x_target).T,mtimes(self.wx_cost,x_perf[i,:].T-x_target))
                
            n_cost_deviation = np.minimum(self.n_perf-1,self.n_safe-1)
            for i in range(1,n_cost_deviation):
                cost += mtimes(x_perf[i,:]-p_all[i,:],mtimes(.1*self.wx_cost,(x_perf[i,:]-p_all[i,:]).T))
        #objective(x_perf,q_all,x_target,k_ff_all,self.wx_cost,self.wu_cost)
                
        _,sigm = self.gp.predict_casadi_symbolic(vertcat(p_0,u_0).T)
        cost += -sum2(sqrt(sigm))        
        
        opt_vars = vertcat(u_0,u_perf,k_ff_all.reshape((-1,1)),k_fb_ctrl.reshape((-1,1)))
        opt_params = vertcat(p_0,x_target,k_fb_0.reshape((-1,1)))
        
        prob = {'f':cost,'x': opt_vars,'p':opt_params,'g':g}
        opt = {'ipopt':{'hessian_approximation':'limited-memory',"max_iter":40,"expect_infeasible_problem":"yes"}} #ipopt 
        #opt = {'qpsol':'qpoases','max_iter':80,'hessian_approximation':'limited-memory'} #sqpmethod #,'hessian_approximation':'limited-memory'
        solver = cas.nlpsol('solver','ipopt',prob,opt)
        
        self.solver = solver
        self.lbg = lbg
        self.ubg = ubg
        self.solver_initialized = True
        self.g_name = g_name
    
    def generate_safety_constraints(self, p_all, q_all, u_0, k_fb_0, k_fb_ctrl, k_ff_all):
        """ Generate all safety constraints
        
        Parameters
        ----------
        p_all:
        q_all:
        k_fb_0:
        k_fb_ctrl:
        k_ff:
        ctrl_bounds:
        
        Returns
        -------
        g: list[casadi.SX]
        lbg: list[casadi.SX]
        ubg: list[casadi.SX]
        """
        g = []
        lbg = []
        ubg = []
        g_name = []
        
        H = np.shape(p_all)[0]
        # control constraints
        if self.has_ctrl_bounds:
            g_u_0, lbg_u_0, ubg_u_0 = self._generate_control_constraint(u_0)
            g = vertcat(g,g_u_0)
            lbg+= lbg_u_0
            ubg+= ubg_u_0
            g_name += ["u_0_ctrl_constraint"]
            
            for i in range(H-1):
                p_i = p_all[i,:].T
                q_i = q_all[i,:].reshape((self.n_s,self.n_s))
                k_ff_i = k_ff_all[i,:].reshape((self.n_u,1))
                k_fb_i = (k_fb_0[i] + k_fb_ctrl).reshape((self.n_u,self.n_s))
                
                g_u_i, lbg_u_i, ubg_u_i = self._generate_control_constraint(k_ff_i,q_i,k_fb_i)
                g = vertcat(g,g_u_i)
                lbg+= lbg_u_i
                ubg+= ubg_u_i
                g_name += ["ellipsoid_ctrl_constraint_{}".format(i)]*len(lbg_u_i)
            
        # intermediate state constraints
        for i in range(H-1):
            p_i = p_all[i,:].T
            q_i = q_all[i,:].reshape((self.n_s,self.n_s))
            g_state = lin_ellipsoid_safety_distance(p_i,q_i,self.h_mat_obs,self.h_obs)
            g = vertcat(g,g_state)
            lbg += [-cas.inf]*self.m_obs
            ubg += [0]*self.m_obs
            g_name += ["ellipsoid_ctrl_constraint_{}".format(i)]*self.m_obs
            
        # terminal state constraint
        p_T = p_all[-1,:].T
        q_T = q_all[-1,:].reshape((self.n_s,self.n_s))
        g_terminal = lin_ellipsoid_safety_distance(p_T,q_T,self.h_mat_safe,self.h_safe)
        g = vertcat(g,g_terminal)
        g_name += ["terminal constraint"]*self.m_safe
        lbg += [-cas.inf]*self.m_safe
        ubg += [0]*self.m_safe
        
        return g,lbg,ubg,g_name
            
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
        
        return np.asarray(k), np.asarray(x), eigVals
        
    def get_lqr_feedback(self,x_0 = None, u_0 = None):
        """ Get the initial feedback controller k_fb
        
        x_0: n_s x 1 ndarray[float], optional
        u_0: n_u x 1 ndarray[float], optional
        
        """
        q = self.wx_feedback
        r = self.wu_feedback
        
        if x_0 is None:
            x_0 = np.zeros((self.n_s,1))
        if u_0 is None:
            u_0 = np.zeros((self.n_u,1))
            
        if self.lin_prior:
            a = self.a
            b= self.b
            
            k_lqr,_,_ = self.dlqr(a,b,q,r)
            k_fb = -k_lqr
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
        if not self.has_openloop:
            return None,None
            
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
                                              self.l_mu,self.l_sigma,c_safety = self.beta_safety, verbose=0,a =self.a,b=self.b)
    
        if get_controls:
            return p_all, q_all, k_fb, k_ff
      
          
        return p_all, q_all
        
    def get_action(self,x0_mu, x_target, x_safe, lqr_only = False):
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
        success: bool
            The control was not successful if we are outside the safezone
            AND we have to revert to the safe controller.
        """ 
        safety_failure = False
        if lqr_only:
            u_apply = self.safe_policy(x0_mu)
            
            return u_apply, safety_failure
            
        u_apply, success = self.solve(x0_mu[:,None],x_target[:,None],x_safe[:,None])
        
        return u_apply.reshape(self.n_u,), success
        
    def solve(self,p_0, p_target, p_safe, k_ff_all_0 = None, k_fb_all_0 = None, sol_verbose = False):
        """ Solve the MPC problem for a given set of input parameters
        
        
        Parameters
        ----------
        p_0: n_s x 1 array[float]
            The initial (current) state
        p_target n_s x 1 array[float]
            The target state
        p_safe n_s x 1 array[float]
            A safe state we may want to return to (required for ilqr initialization)
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
        
        
        u_0, k_ff_all_0, k_fb_0, u_perf_0, k_fb_ctrl_0  = self._get_init_controls(p_0,p_target,p_safe)
        
        params = np.vstack((p_0,p_target,cas_reshape(k_fb_0,(-1,1))))
        u_init = vertcat(cas_reshape(u_0,(-1,1)),cas_reshape(u_perf_0,(-1,1)), \
                         cas_reshape(k_ff_all_0,(-1,1)),cas_reshape(k_fb_ctrl_0,(-1,1)))
        
        sol = self.solver(x0=u_init,lbg=self.lbg,ubg=self.ubg,p=params)
        return self._get_solution(p_0,sol,k_fb_0,sol_verbose)
        
        
    def _get_solution(self,x_0,sol, k_fb_0, sol_verbose = False,feas_tol = 1e-4):
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
        g_res = np.array(sol["g"]).squeeze()

        success = True
        feasible = True
        if np.any(np.array(self.lbg) - feas_tol > g_res ) or np.any(np.array(self.ubg) + feas_tol < g_res ):      
            feasible = False
            
        if self.verbosity > 1:
            print("\n=== Constraint values:")
            for i in range(len(g_res)):
                print(" constraint: {}, lbg: {}, g: {}, ubg: {} ".format(self.g_name[i],self.lbg[i],g_res[i],self.ubg[i]))
            print("\n")
        
        if self.verbosity > 0:
            print("===== SOLUTION FEASIBLE: {} ========".format(feasible))
            
        if feasible:
            self.n_fail = 0
            x_opt = sol["x"]
            self.has_openloop = True
            
            #get indices of the respective variables
            n_u_0 = self.n_u
            n_u_perf = (self.n_perf-1)*self.n_u
            n_k_ff = (self.n_safe-1)*self.n_u
            n_k_fb_ctrl = self.n_s*self.n_u
            c=0
            idx_u_0 = np.arange(n_u_0)
            c+= n_u_0
            idx_u_perf = np.arange(c,c+n_u_perf)
            c+= n_u_perf
            idx_k_ff = np.arange(c,c+n_k_ff)
            c+= n_k_ff 
            idx_k_fb = np.arange(c,c+n_k_fb_ctrl)
            
            u_apply = np.array(x_opt[idx_u_0]).reshape((1,self.n_u))
            u_perf = np.array(cas_reshape(x_opt[idx_u_perf],(self.n_perf-1,self.n_u)))
            u_perf_all = np.vstack((u_apply,u_perf))
            k_safe = np.array(cas_reshape(x_opt[idx_k_ff],(self.n_safe-1,self.n_u)))
            k_ff_all = np.vstack((u_apply,k_safe))
            k_fb_ctrl = np.array(cas_reshape(x_opt[idx_k_fb],(1,self.n_u*self.n_s)))
            k_fb = k_fb_0 + np.matlib.repmat(k_fb_ctrl,self.n_safe-1,1)
            
            k_fb_apply = np.empty((self.n_safe-1,self.n_u,self.n_s))
            for i in range(self.n_safe-1):
                k_fb_apply[i] = cas_reshape(k_fb[i],(self.n_u,self.n_s))
                
            p_ctrl , _ = self.get_trajectory_openloop(x_0.squeeze(),k_fb_apply,k_ff_all)
            
            if self.rhc:
                self.k_fb_all = k_fb_apply
                self.k_ff_all = k_ff_all
                self.u_perf_all = u_perf_all
                self.p_ctrl = p_ctrl
                
        else:
            if self.verbosity > 1:
                print("Infeasible solution!")
            
            self.n_fail += 1
            if self.n_fail >= self.n_safe:
                ## Too many infeasible solutions -> switch to safe controller
                u_apply = self.safe_policy(x_0)
            else:
                ## can apply previous solution
                u_apply = self.get_old_solution(x_0)
            
        return u_apply, success
        
    def get_old_solution(self, x, k = None):
        """ Shift previously obtained solutions in time and return solution to be applied
        
        Prameters
        ---------
        k: int, optional
            The number of steps to shift back in time. This is number is
            already tracked by the algorithm, so a custom value should be used with caution
            
        Returns
        -------
        u_apply: n_s x 0 1darray[float]
            The controls to be applied at the current time step
        """
        if self.n_safe > self.n_safe:
            warnings.warn("There are no previous solution to be applied. Returning None")
            return None
        k = self.n_fail
        
        k_fb_old = self.k_fb_all[k-1]
        k_ff = self.k_ff_all[k,:,None]
        p_ctrl = self.p_ctrl[k-1,:,None]
        
        return self.feedback_ctrl(x,k_ff,k_fb_old,p_ctrl)
        
    def feedback_ctrl(self,x,k_ff,k_fb = None,p=None):
        """ The feedback control structure """
        
        if k_fb is None:
            return k_ff
            
        return np.dot(k_fb,(x-p)) + k_ff
        
    def init_ilqr(self,x_0,p_target,p_safe):
        """ Solve a iLQR problem to compute initial values for the MPC problem"""
        ilqr_initializer = self.ilqr_initializer
        ilqr_initializer.cost.x_target = p_target
        ilqr_initializer.cost.x_safe = p_safe
        if self.rhc and self.n_fail == 0:
            u_old = self.k_ff_all
            u_0 = np.vstack((u_old[1:],np.zeros((1,self.n_u))))
        else:
            u_0 = np.zeros((self.n_safe,self.n_u))
        _,u_ilqr,_,k_fb_ilqr,k_ilqr,alpha_opt = ilqr_initializer.ilqr(x_0,u_0)
        
        k_fb_0 = np.empty((self.n_safe-1,self.n_s*self.n_u))
        
        u_0 = u_ilqr[0]
        k_ff_0 = u_ilqr[1:,:]
        
        for i in range(self.n_safe-1):
            k_fb_0[i] = cas_reshape(k_fb_ilqr[i+1],(1,self.n_s*self.n_u))
            
            if self.verbosity > 1:
                if i == 0:
                    print("\nu_0:")
                    print(u_0)
                print("\nfeedback and feed forward init ilqr step {}:".format(i))
                print(k_fb_ilqr[i+1])
                print(u_ilqr[i+1])
        return u_0,k_fb_0, k_ff_0 
        
    def _get_init_controls(self,x_0,p_target,p_safe):
        """ Initialize the controls for the MPC step
        
        """
       
        if self.ilqr_init:
            
            u_0, k_fb_0, k_ff_all_0 = self.init_ilqr(x_0,p_target,p_safe)
            
            k_fb_ctrl_0 = np.zeros((self.n_s*self.n_u,1))
            if self.do_shift_solution and self.n_fail == 0:
                u_perf_old = np.copy(self.u_perf_all)
                u_perf_0 = np.vstack((u_perf_old[2:,:],np.zeros((1,self.n_u))))
            else:
                u_perf_0 = np.zeros((self.n_perf-1,self.n_u))
        else:
            if self.do_shift_solution and self.n_fail == 0:
                k_ff_old = np.copy(self.k_ff_all) 
                u_perf_old = np.copy(self.u_perf_all)
                u_0 = (k_ff_old[0,:] + u_perf_old[0,:])/2
                k_ff_all_0 = np.vstack((k_ff_old[2:,:],np.zeros((1,self.n_u))))  
                u_perf_0 = np.vstack((u_perf_old[2:,:],np.zeros((1,self.n_u))))  
            else:
                k_ff_all_0 = 2*np.random.randn(self.n_safe-1,self.n_u)
                u_perf_0 = 2*np.random.randn(self.n_perf-1,self.n_u)
                u_0 = 2*np.random.randn(self.n_u,1)
                    
        return u_0, k_ff_all_0, k_fb_0, u_perf_0, k_fb_ctrl_0 
                
    def update_model(self, x, y, train = True, replace_old = True):
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
        
        self.gp.update_model(x,y-y_prior,train,replace_old)
        self.init_solver()
    
    def init_ilqr_initializer(self):
        """ Initialize the iLQR method to get initial values for the NLP method """
        model = SafeMPCModelILQR(self)
        cost = SafeMPCCostILQR(self)
        u_min = None
        u_max = None
        if self.has_ctrl_bounds:
            u_bounds = self.ctrl_bounds
            u_min = u_bounds[:,0]
            u_max = u_bounds[:,1]
            
        self.ilqr_initializer = CILQR(model,cost,H=self.n_safe,u_min=u_min,u_max=u_max,w_x= 1e3,w_u=1e2)
        
        
class SafeMPCModelILQR:
    """ Utiliy class which creates a model for the ilqr initialization
    
    Attributes
    ----------
    a
    b: 
    gp: SimpleGPModel
        The GPModel
    n_s: int
        Number of states
    n_u: int
        Number of actions
    H: int
        The length of the control sequence
    dt: float
        The control frequency    
    """
    
    def __init__(self,safempc):
        """ Initialize based on a pre-existing SafeMPC object
        
        Parameters
        ----------
        safempc: SafeMPC
            The underlying SAFEMPC object
        """
        self.a = safempc.a
        self.b = safempc.b
        self.gp = safempc.gp
        self.n_u = safempc.n_u
        self.n_s = safempc.n_s
        self.H = safempc.n_safe
        self.dt = safempc.dt
        
    def simulate(self,x_0,U):
        """ Simulate a rollout/forward pass of the system
        
        Simulate a trajectory 
            x_{t+1} = x_t + df(x_t,u_t) , t = 0,..,H
                    \approx x_t + dt*A_t *x_t + dt*B_t *u_t
        Parameters
        ----------
        x_0: n_s x 1 ndarray[float]
            The start state of the trajectory
        U: H x n_u ndarray[float]
            The control trajectory
            
        Returns
        -------
        x_all: H x n_s ndarray[float]
        j_x_all: H x n_s x n_s ndarray[float]
        j_u_all: H x n_s x n_u ndarray[float]
        
        """
        x_all = np.empty((self.H+1,self.n_s))
        j_x_all = np.empty((self.H,self.n_s,self.n_s))
        j_u_all = np.empty((self.H,self.n_s,self.n_u))
        
        x_all[0] = x_0.squeeze()
        x = x_0
        for i in range(self.H):
            u = U[i,:,None]
            x_diff, j_x, j_u = self.forward_step(x,u,compute_grads=True)
            x = x + x_diff
            
            x_all[i+1] = x.squeeze()
            j_x_all[i,:,:] = j_x
            j_u_all[i,:,:] = j_u
            
        return x_all, j_x_all, j_u_all
            
    def forward_step(self, x, u, compute_grads = False):
        """ Simulate the system one step forward
        
            x_+ = x + df(x,u)
                \approx x + A*x + B *u
                = x + J_df^x*x + J_df^u*u
        Parameters
        ----------
        x: n_s x 1 ndarray[float]
            The current state
        u: n_u x 1 ndarray[float]
            The current action
        
        Returns
        -------
        x_diff: n_s x 1 ndarray[float]
            Corresponds to df(x,u) (see above)
        J_x: n_s x n_s ndarray[float]
            Corresponds to the jacobian w.r.t. x (see above)
        J_u: n_s x n_u ndarray[float]
            Corresponds to the jacobian w.r.t. u (see above)
        """
        inp = np.hstack((x.T,u.T))
        pred = self.gp.predict(inp,None,compute_grads)
        
        x_diff_mu = pred[0]
        x_diff = np.dot(self.a,x) + np.dot(self.b,u) + x_diff_mu.T - x       
        if compute_grads:        
            j_mu_inp = pred[2]
            j_mu_x = j_mu_inp[0,:,:self.n_s]
            j_mu_u = j_mu_inp[0,:,self.n_s:]
            j_x = self.a-np.eye(self.n_s)+j_mu_x
            j_u = self.b+j_mu_u
            
            return x_diff, j_x, j_u    
        return x_diff
        

class SafeMPCCostILQR:
    """ Utiliy class which creates a cost function for the ilqr initialization 
    
    Attributes
    ----------
    x_target: n_s x 1 np.array[float]
    x_safe: n_x x 1 np.array[float]
    
    """
    
    def __init__(self,safempc):
        """ Initialize based on a pre-existing SafeMPC object
        
        Parameters
        ----------
        safempc: SafeMPC
            The underlying SAFEMPC object
        """
        self.n_s = safempc.n_s
        self.n_u = safempc.n_u
        self.x_safe = None
        self.x_target = None
        
    def total_cost(self, x_all, u_all, w_x = None, w_u = None):
        """ total cost of trajectory
        x_all: n_s x 1 ndarray[float]
            
        """
        tN, _  = np.shape(u_all)
        c = 0
        for t in range(tN):
            c += self.cost(x_all[t,:,None],u_all[t,:,None],t,w_x,w_u)[0]
        c += self.cost_final(x_all[-1,:,None],w_x)[0]
        
        return c
        
    def cost(self, x, u, t, w_x = None, w_u = None):
        """ intermediate cost
        
        """
        c = .5*np.dot(u.T,np.dot(w_u,u))
        c_u = np.dot(w_u,u)
        c_uu = w_u
        c_ux = np.zeros((self.n_u,self.n_s))
            
        c_x = np.zeros((self.n_s,1))
        c_xx = np.zeros((self.n_s,self.n_s))
        
        if t == 1:
            c += .5**np.dot((x-self.x_target).T,np.dot(w_x,x-self.x_target))
            c_x += np.dot(w_x,x-self.x_target)
            c_xx += w_x
                
        if t > 1:            
            c += .5*np.dot((x-self.x_safe).T,np.dot(w_x,x-self.x_safe))
            c_x += np.dot(w_x,x-self.x_safe)
            c_xx += w_x
        
        return c, c_x.squeeze(), c_xx, c_u.squeeze(), c_uu, c_ux
            
    def cost_final(self, x, w_x = None):
        """ terminal cost
        
        """
        c = .5*np.dot((x-self.x_safe).T,np.dot(w_x,x-self.x_safe))
        c_x = np.dot(w_x,x-self.x_safe)
        c_xx = w_x
        
        return c, c_x.squeeze(), c_xx
        
        
        
        
    