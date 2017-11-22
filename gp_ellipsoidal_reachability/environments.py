# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:16:45 2017

@author: tkoller
"""
import abc
import numpy as np
import warnings
from utils_visualization import plot_ellipsoid_2D
from scipy.integrate import ode
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
    def state_to_obs(self,current_state):
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
    def __init__(self,name = "InvertedPendulum", l = 1., m = 1., g = 9.82, b = .01,
                 dt = .05, start_state = [0,0], init_std = .01, plant_noise = np.array([0.01,0.01])**2,
                 u_min = np.array([-20.]), u_max = np.array([20.]),target = np.array([0.0,0.0])):
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
        self.p_origin = np.array([0,0])
        self.l_mu = np.array([0.01]*2) #TODO: This should be somewhere else
        self.l_sigm = np.array([0.01]*2)
        self.target = target
        self._init_safety_bounds()
        
    def reset(self, mean = None, std = None):
        """ Reset the system and sample a new start state
        
        
        """
        super(InvertedPendulum,self).reset()
        self.current_state = self._sample_start_state(mean = mean,std = std)
        self.odesolver.set_initial_value(self.current_state,0.0)
        
        return self.state_to_obs(self.current_state)
        
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
        dz = np.zeros((2,1))
        dz[0] = (action - self.b*state[0] + self.m*self.g*self.l*np.sin(state[1])) / (self.m*self.l**(3/2)*self.g**(1/2))
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
        jac_0 = np.zeros((1,3)) #jacobian of the first equation (dz[0])
        jac_0[0,0] = -self.b/(self.m*self.l**(3/2)*self.g**(1/2)) # derivative w.r.t. d_theta
        jac_0[0,1] = (self.g**(1/2)/self.l**(1/2)) * np.cos(state[1]) # derivative w.r.t. theta
        jac_0[0,2] = 1/(self.m*self.l**(3/2)*self.g**(1/2)) # derivative w.r.t. u
        
        jac_1 = np.eye(1,3) #jacobian of the second equation
        
        return np.vstack((jac_0,jac_1))
        
    def linearize_discretize(self,x_center=None,u_center=None):
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
        A_ct = np.eye(self.n_s)+jac_ct[:,:self.n_s]
        B_ct = jac_ct[:,self.n_s:]
        
        ct_input = (A_ct,B_ct,np.eye(self.n_s),np.zeros((self.n_s,self.n_u)))
        A,B,_,_,_ = cont2discrete(ct_input,self.dt)
        
        return A,B
        
    def state_to_obs(self, state, add_noise = False):
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
        noise = 0
        if add_noise:
            noise += np.random.randn(self.n_s)*np.sqrt(self.plant_noise)
            
        return state - self.p_origin + noise
        
    def plot_state(self, ax, x = None, color = "b"):
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
        
    def plot_safety_bounds(self,ax, plot_safe_bounds = True, plot_obs = True):
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
        
        if not (plot_safe_bounds or plot_obs):
            warnings.warn("plot_safety_bounds doesn't plot anything")
        
        p_safe, width_safe, height_safe, p_obs, width_obs, height_obs = self.get_safe_bounds()
        if plot_safe_bounds:     
            ax.add_patch(mpatch.Rectangle(p_safe,width_safe,height_safe,fill = False))     
        if plot_obs:       
            ax.add_patch(mpatch.Rectangle(p_obs,width_obs,height_obs,fill = False))
        
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
        dtheta_max_safe = self.h_safe[0]
        dtheta_min_safe = -self.h_safe[1]
        theta_max_safe = self.h_safe[2]
        theta_min_safe = -self.h_safe[3]
        
        width_safe = dtheta_max_safe - dtheta_min_safe
        height_safe = theta_max_safe - theta_min_safe
        p_safe = (dtheta_min_safe + self.p_origin[0],theta_min_safe+ self.p_origin[1])
        
        theta_max_obs = self.h_obs[0]
        theta_min_obs = -self.h_obs[1]     
        
        width_obs = dtheta_max_safe - dtheta_min_safe #there are no specific bounds on dtheta as obstacle
        height_obs =  theta_max_obs - theta_min_obs
        p_obs = (dtheta_min_safe+ self.p_origin[0],theta_min_obs+ self.p_origin[1])
        
        return p_safe, width_safe, height_safe, p_obs, width_obs, height_obs
        
    def step(self, action):
        """ Apply action to system and output current state and other information.
        
        Parameters
        ----------
        action: 1x0 1darray[float]
        """ 
        action = np.clip(np.nan_to_num(action),self.u_min,self.u_max)
        print(action)
        self.odesolver.set_f_params(action)
        self.current_state = self.odesolver.integrate(self.odesolver.t+self.dt) 
        
        self.iteration += 1
        done = False
        new_state_noise_obs = self.state_to_obs(np.copy(self.current_state),add_noise=True)
        new_state_obs = self.state_to_obs(np.copy(self.current_state))
        
        if self.odesolver.successful():
            return action,new_state_obs,new_state_noise_obs,done
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
        
    def _init_safety_bounds(self):
        """ Get state and safety constraints"""
        
        h_mat_safe_dtheta = np.asarray([[1.,0.],[-1.,0.]])
        h_safe_dtheta = np.array([.8,.8]).reshape(2,1)
        h_mat_safe_theta = np.asarray([[0.,1.],[0.,-1.]])
        h_safe_theta = np.array([0.1,0.1]).reshape(2,1)
        
        
        self.h_mat_safe = np.vstack((h_mat_safe_dtheta,h_mat_safe_theta))
        self.h_safe = np.vstack((h_safe_dtheta,h_safe_theta))
        self.h_mat_obs = np.asarray([[0.,1.],[0.,-1.]])
        self.h_obs = np.array([.5,.5]).reshape(2,1)
        
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
        
        
        