# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:16:45 2017

@author: tkoller
"""
import abc
import numpy as np
import warnings
from utils_visualization import plot_ellipsoid_2D
from scipy.integrate import ode,odeint
from scipy.signal import cont2discrete
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt


class Environment:
    """ Base class for environments
    
    
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,name, n_s, n_u, dt, start_state, init_std, plant_noise,
                 u_min, u_max, target):
        """
        
        """
        self.name = name
        self.n_s = n_s
        self.n_u = n_u
        self.dt = dt
        self.start_state = start_state
        self.is_initialized = False
        self.iteration = 0
        self.init_std = init_std
        self.u_min = u_min
        self.u_max = u_max
        self.plant_noise = plant_noise
        self.target = target
        
    @abc.abstractmethod
    def reset(self,mean = None, std = None):
        """ Reset the system."""
        self.is_initialized = True
        self.iteration = 0
    
    @abc.abstractmethod    
    def step(self, action):
        """ Apply action to system and output current state and other information.""" 
        pass
    
    @abc.abstractmethod
    def _dynamics(self,t,state,action):
        """ Evaluate the system dynamics """
        pass
     
    @abc.abstractmethod
    def state_to_obs(self,current_state = None):
        """ Transform the dynamics state to the state to be observed """
        pass
    
    @abc.abstractmethod
    def random_action(self):
        """ Apply a random action to the system """
        pass
    
    @abc.abstractmethod
    def plot_ellipsoid_trajectory(self, p, q, vis_safety_bounds = True):
        """ Visualize the reachability ellipsoid"""
        pass
    
    def _sample_start_state(self, mean = None, std = None):
        """ """
        init_std = self.init_std
        if not std is None:
            init_std = std
            
        init_m = mean
        if init_m is None:
            init_m = self.start_state
        
        return init_std*np.random.randn(self.n_s)+init_m
        
    def get_target(self):
        """ Return the target state 
    
        Returns
        -------
        target: n_sx0 1darray[float]
            The target state in observation space
        """
        return self.state_to_obs(self.target)
        
class InvertedPendulum(Environment):
    """ The Inverted Pendulum environment
    
    The simple two-dimensional Inverted Pendulum environment.
    The system consists of two states and one action:
    States:
        0. d_theta
        1. theta
    
    TODO: Need to define a safety/fail criterion
    """
    def __init__(self,name = "InvertedPendulum", l = .5, m = .15, g = 9.82, b = 0.,
                 dt = .05, start_state = [0,0], init_std = .01, plant_noise = np.array([0.001,0.001])**2,
                 u_min = np.array([-1.]), u_max = np.array([1.]),target = np.array([0.0,0.0]),
                 verbosity = 1, norm_x = None, norm_u = None):
        """
        Parameters
        ----------
        name: str, optional
            The name of the system
        l: float, optional
            The length of the pendulum
        m: float, optional 
            The mass of the pendulum
        g: float, optional
            The gravitation constant
        b: float, optional
            The friction coefficient of the system
        start_state: 2x0 1darray[float], optional 
            The initial state mean
        init_std: float, optional
            The standard deviation of the start state sample distribution.
            Note: This is not(!) the uncertainty of the state but merely allows
            for variation in the initial (deterministic) state.
        u_min: 1x0 1darray[float], optional
            The maximum torque applied to the system
        u_max: 1x0 1darray[float], optional
            The maximum torquie applied to the system
        target: 2x0 1darray[float], optional
            The target state
        """
        super(InvertedPendulum,self).__init__(name,2,1,dt,start_state,init_std,plant_noise,u_min,u_max,target)
        self.odesolver = ode(self._dynamics)
        self.l = l
        self.m = m
        self.g = g
        self.b = b
        self.p_origin = np.array([0.0,0.0])
        self.l_mu = np.array([0.1,.05]) #TODO: This should be somewhere else
        self.l_sigm = np.array([0.1,.05])
        self.target = target
       
        self.verbosity = verbosity
        
        max_deg = 30
        if norm_x is None:
            norm_x = np.array([np.sqrt(g/l), np.deg2rad(max_deg)])
        
        if norm_u is None:
            norm_u = np.array([g*m*l*np.sin(np.deg2rad(max_deg))])
            
        self.norm = [norm_x,norm_u]
        self.inv_norm = [arr ** -1 for arr in self.norm]
        
        self._init_safety_constraints()
         
    def reset(self, mean = None, std = None):
        """ Reset the system and sample a new start state
        
        
        """
        super(InvertedPendulum,self).reset()
        self.current_state = self._sample_start_state(mean = mean,std = std)
        self.odesolver.set_initial_value(self.current_state,0.0)
        
        return self.state_to_obs(self.current_state)
        
    def simulate_onestep(self, state, action):
        """ """
        
        one_step_dyn = lambda s,t,a: self._dynamics(t,s,a).squeeze()
        
        
        #unnormalize state and action
        state = state*self.norm[0]
        action = action * self.norm[1]

        sol = odeint(one_step_dyn,state,np.array([0.0,self.dt]),args=tuple(action))
        next_state = sol[1,:]
        
        return self.state_to_obs(next_state),self.state_to_obs(next_state,True)
        
        
    def _dynamics(self, t, state, action):
        """ Evaluate the system dynamics 
        
        Parameters
        ----------
        t: float
            Input Parameter required for the odesolver for time-dependent
            odes. Has no influence in this system.
        state: 2x1 array[float]
            The current state of the system
        action: 1x1 array[float]
            The action to be applied at the current time step
            
        Returns
        -------
        dz: 2x1 array[float]
            The ode evaluated at the given inputs.
        """
        
        inertia = self.m* self.l**2
        dz = np.zeros((2,1))
        dz[0] = self.g / self.l * np.sin(state[1]) + action / inertia - self.b/ inertia * state[0]
        dz[1] = state[0]
        
        return dz
        
    def _jac_dynamics(self,t,state,action):
        """ Evaluate the jacobians of the system dynamics
        
        Parameters
        ----------
        t: float
            Input Parameter required for the odesolver for time-dependent
            odes. Has no influence in this system.
        state: 2x1 array[float]
            The current state of the system
        action: 1x1 array[float]
            The action to be applied at the current time step
            
        Returns
        -------
        jac: 2x3 array[float]
            The jacobian of the dynamics w.r.t. the state and action
        """
        inertia = self.m* self.l**2
        jac_0 = np.zeros((1,3)) #jacobian of the first equation (dz[0])
        jac_0[0,0] = self.b/ inertia # derivative w.r.t. d_theta
        jac_0[0,1] = self.g/self.l * np.cos(state[1]) # derivative w.r.t. theta
        jac_0[0,2] = 1 / inertia # derivative w.r.t. u
        
        jac_1 = np.eye(1,3) #jacobian of the second equation
        
        return np.vstack((jac_0,jac_1))
        
    def linearize_discretize(self,x_center=None,u_center=None, normalize = True):
        """ Discretize and linearize the system around an equilibrium point
        
        Parameters
        ----------
        x_center: 2x0 array[float], optional
            The linearization center of the state. 
            Default: the origin
        u_center: 1x0 array[float], optional
            The linearization center of the action
            Default: zero
        """
        if x_center is None:
            x_center = self.p_origin
        if u_center is None:
            u_center = np.zeros((self.n_s,))
            
        jac_ct = self._jac_dynamics(0,x_center,u_center)
        A_ct = jac_ct[:,:self.n_s]
        B_ct = jac_ct[:,self.n_s:]
        
        if normalize:
            m_x = np.diag(self.norm[0])
            m_u = np.diag(self.norm[1])
            m_x_inv = np.diag(self.inv_norm[0])
            m_u_inv = np.diag(self.inv_norm[1])
            A_ct = np.linalg.multi_dot((m_x_inv,A_ct,m_x))
            B_ct = np.linalg.multi_dot((m_x_inv,B_ct,m_u))
        
        ct_input = (A_ct,B_ct,np.eye(self.n_s),np.zeros((self.n_s,self.n_u)))
        A,B,_,_,_ = cont2discrete(ct_input,self.dt)
        
        return A,B
        
    def normalize(self,state = None,action = None):
        """ Normalize the inputs"""
        if not state is None:
            state = self.inv_norm[0]*state
            
        if not action is None:
            action = self.inv_norm[1]*action
            
        return state, action
        
    def unnormalize(self,state = None, action = None):
        """ Unnormalize the inputs"""
        if not state is None:
            state = self.norm[0]*state
            
        if not action is None:
            action = self.norm[1]*action
            
        return state, action
        
    def state_to_obs(self, state = None, add_noise = False):
        """ Transform the dynamics state to the state to be observed
        
        Parameters
        ----------
        state: 2x0 1darray[float]
            The internal state of the system.
        add_noise: bool, optional
            If this is set to TRUE, a noisy observation is returned
            
        Returns
        -------
        state: 2x0 1darray[float]
            The state as is observed by the agent. 
            In the case of the inverted pendulum, this is the same.
        
        """
        if state is None:
            state = self.current_state
        noise = 0
        if add_noise:
            noise += np.random.randn(self.n_s)*np.sqrt(self.plant_noise)
        
        state_noise = state + noise
        state_norm = state_noise * self.inv_norm[0]
           
        return state_norm
        
    def plot_state(self, ax, x = None, color = "b",normalize =True):
        """ Plot the current state or a given state vector
        
        Parameters:
        -----------
        ax: Axes Object
            The axes to plot the state on
        x: 2x0 array_like[float], optional
            A state vector of the dynamics
        Returns
        -------
        ax: Axes Object
            The axes with the state plotted 
        """ 
        if x is None:
            x = self.current_state
            if normalize:
                x,_ = self.normalize(x)
        assert len(x) == self.n_s, "x needs to have the same number of states as the dynamics"
        plt.sca(ax)
        ax.plot(x[0],x[1],"{}x".format(color))
        return ax
        
    def plot_ellipsoid_trajectory(self,p,q, vis_safety_bounds = True,ax = None, color = "r"):
        """ Plot the reachability ellipsoids given in observation space
        
        TODO: Need more principled way to transform ellipsoid to internal states
        
        Parameters
        ----------
        p: n x n_s array[float]
            The ellipsoid centers of the trajectory
        q: n x n_s x n_s  ndarray[float]
            The shape matrices of the trajectory
        vis_safety_bounds: bool, optional
            Visualize the safety bounds of the system            
        
        """
        new_ax = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            new_ax = True
        plt.sca(ax)
            
        n, n_s = np.shape(p)
        handles = [None]*n
        for i in range(n):
            p_i = p[i,:].reshape((n_s,1)) + self.p_origin.reshape((n_s,1))
            q_i = q[i]
            ax, handles[i] = plot_ellipsoid_2D(p_i,q_i,ax,color = color)
        
        if vis_safety_bounds:
            ax = self.plot_safety_bounds(ax)
            
        if new_ax: 
            plt.show()
            
        return ax, handles
        
    def plot_safety_bounds(self,ax = None, plot_safe_bounds = True,plot_obs = False, normalize = True):
        """ Given a 2D axes object, plot the safety bounds on it 
    
        Parameters
        ----------
        ax: Axes object,
            The input axes object to plot on
            
        Returns
        ------- 
        ax: Axes object
            The same Axes object as the input ax but now contains the rectangle
        """
        new_fig = False
        if ax is None:
            new_fig = True
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            
            
        if not (plot_safe_bounds or plot_obs):
            warnings.warn("plot_safety_bounds doesn't plot anything")
        
        x_polygon = self.corners_polygon
        if normalize:
            m_x = np.diag(self.inv_norm[0])
            x_polygon = np.dot(x_polygon,m_x.T)
 
        if plot_safe_bounds:     
            ax.add_patch(mpatch.Polygon(x_polygon,fill = False))     
        if new_fig:
            #ax.set_xlim(-2.,2.)
            #ax.set_ylim(-.6,.6)
            
            return fig, ax
            
        return ax
        
    def get_safe_bounds(self):
        """ Returns the parameters of a rectangle visualizing safety bounds
        
        Returns
        -------
        p_safe: 2x0 tuple[float]
            The lower left corner of the rectangle representing the safe zone
        width_safe: float
            The width of the safety rectangle
        height_safe: float
            The height of the safety rectangle
        p_safe: 2x0 tuple[float]
            The lower left corner of the rectangle representing the obstacle free zone
        width_safe: float
            The width of the obstacle free rectangle
        height_safe: float
            The height of the obstacle free rectangle               
        """
        raise DeprecationWarning("We replace rectangles with polygons. Simpler for drawing")
        dtheta_max_safe = self.h_safe[0]
        dtheta_min_safe = -self.h_safe[1]
        theta_max_safe = self.h_safe[2]
        theta_min_safe = -self.h_safe[3]
        
        width_safe = dtheta_max_safe - dtheta_min_safe
        height_safe = theta_max_safe - theta_min_safe
        p_safe = (dtheta_min_safe + self.p_origin[0],theta_min_safe+ self.p_origin[1])
               
        return p_safe, width_safe, height_safe
        
    def step(self, action):
        """ Apply action to system and output current state and other information.
        
        Parameters
        ----------
        action: n_u x 0 1darray[float]
            The normalized(!) action
        """ 
        
        action_clipped = np.clip(np.nan_to_num(action),self.u_min,self.u_max) #clip to normalized max action
        action = self.norm[1] * action_clipped #unnormalize
        
        self.odesolver.set_f_params(action)
        old_state = np.copy(self.current_state)
        self.current_state = self.odesolver.integrate(self.odesolver.t+self.dt) 
        
        self.iteration += 1
        done = False
        
        new_state_noise_obs = self.state_to_obs(np.copy(self.current_state),add_noise=True)
        new_state_obs = self.state_to_obs(np.copy(self.current_state))
        
        if self.odesolver.successful():
            
            if self.verbosity>0:
                print("\n===Old state:")
                print(old_state)
                print("===Action:")
                print(action)
                print("===Next state:")
                print(self.current_state)
            
            return action_clipped,new_state_obs,new_state_noise_obs,done
        raise ValueError("Odesolver failed!")
        
    def random_action(self):
        """ Apply a random action to the system 
        
        Returns
        -------
        action: 1x0 1darray[float]
            A (valid) random action applied to the system.
        
        """
        c = 0.5
        return c*(np.random.rand(self.n_u) * (self.u_max - self.u_min) + self.u_min)
        
    def _init_safety_constraints(self):
        """ Get state and safety constraints
        
        We define the state constraints as:
            x_0 - 3*x_1 <= 1
            x_0 - 3*x_1 >= -1
            x_1 <= max_rad
            x_1 >= -max_rad
        """
        
        max_deg = 12
        
        max_rad = np.deg2rad(max_deg)
        max_dtheta = .4
        h_mat_safe_dtheta = np.asarray([[1.,0.],[-1.,0.]])
        h_safe_dtheta = np.array([max_dtheta,max_dtheta]).reshape(2,1)
        h_mat_safe_theta = np.asarray([[0.,1.],[0.,-1.]])
        h_safe_theta = np.array([max_rad,max_rad]).reshape(2,1)
        
        
        #normalize safety bounds
        self.h_mat_safe = np.vstack((h_mat_safe_dtheta,h_mat_safe_theta))
        self.h_safe = np.vstack((h_safe_dtheta,h_safe_theta))
        self.h_mat_obs = None#p.asarray([[0.,1.],[0.,-1.]])
        self.h_obs = None #np.array([.6,.6]).reshape(2,1)
        
        #upper left, lower 
        self.corners_polygon = np.array([[max_dtheta,-max_rad],\
                                           [max_dtheta,max_rad ],\
                                           [ -max_dtheta,max_rad],\
                                           [-max_dtheta,-max_rad]])
                                           
    def get_safety_constraints(self, normalize = True):
        """ Return the safe constraints
        
        Parameters
        ----------
        normalize: boolean, optional
            If TRUE: Returns normalized constraints
        """
        if normalize:
            m_x= np.diag(self.norm[0])
            h_mat_safe = np.dot(self.h_mat_safe,m_x)
        else:
            h_mat_safe = self.h_mat_safe
            
        return h_mat_safe,self.h_safe,self.h_mat_obs,self.h_obs
        
if __name__ == "__main__":
    pend = InvertedPendulum()
    s = pend.reset()
    print(s)
    a = pend.random_action()
    print(a)
    _,s_new,_ = pend.step(a)
    print(s_new)
    
    p = np.vstack((s.reshape((1,-1)),s_new.reshape((1,-1))))
    q = .1*np.eye(2).reshape((1,-1))
    q = np.stack((q,q))
    pend.plot_ellipsoid_trajectory(p,q,True)
        
        
        