# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:16:45 2017

@author: tkoller
"""
import abc
import numpy as np
from utils_visualization import plot_ellipsoid_2D
from scipy.integrate import ode
import matplotlib.pyplot as plt

class Environment:
    """ Base class for environments
    
    
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,name, n_s, n_u, dt, start_state, init_std, plant_noise,
                 u_min, u_max):
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
                 dt = .05, start_state = [0,np.pi], init_std = .01, plant_noise = np.array([0.01,0.01])**2,
                 u_min = np.array([-10.]), u_max = np.array([10.])):
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
        """
        super(InvertedPendulum,self).__init__(name,2,1,dt,start_state,init_std,plant_noise,u_min,u_max)
        self.odesolver = ode(self._dynamics)
        self.l = l
        self.m = m
        self.g = g
        self.b = b
        self.p_origin = np.array([0,np.pi])
        
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
        dz[0] = (action - self.b*state[0] - self.m*self.g*self.l*np.sin(state[1])/2) / (self.m*self.l**2/3)
        dz[1] = state[0]
        
        return dz
        
    def state_to_obs(self, state):
        """ Transform the dynamics state to the state to be observed
        
        Parameters
        ----------
        state: 2x0 1darray[float]
            The internal state of the system.
        Returns
        -------
        state: 2x0 1darray[float]
            The state as is observed by the agent. 
            In the case of the inverted pendulum, this is the same.
        
        """
        return state - self.p_origin
        
    def plot_ellipsoid_trajectory(self,p,q, vis_safety_bounds = True):
        """ Plot the reachability ellipsoids 
        
        TODO: Need more principled way to transform ellipsoid to internal states
        
        Parameters
        ----------
        p: n x n_s array[float]
            The ellipsoid centers of the trajectory
        q: n x (n_s * n_s) array[float]
            The shape matrices of the trajectory
        vis_safety_bounds: bool, optional
            Visualize the             
        
        """
        
        n, n_s = np.shape(p)
        
        if vis_safety_bounds:
            raise NotImplementedError("Need to visualize safety bounds")
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(n):
            p_i = p[i,:].reshape((n_s,1)) + self.p_origin.reshape((n_s,1))
            q_i = q[i,:].reshape((n_s,n_s))
            ax = plot_ellipsoid_2D(p_i,q_i,ax)
         
        plt.show()
        
    def step(self, action):
        """ Apply action to system and output current state and other information.
        
        Parameters
        ----------
        action: 1x0 1darray[float]
        """ 
        action = np.clip(np.nan_to_num(action),self.u_min,self.u_max)
        self.odesolver.set_f_params(action)
        
        self.current_state = self.odesolver.integrate(self.odesolver.t+self.dt) + np.random.randn(self.n_s)*np.sqrt(self.plant_noise)
        self.iteration += 1
        done = False
        new_state_obs = self.state_to_obs(self.current_state)
        
        if self.odesolver.successful():
            return action,new_state_obs,done
        raise ValueError("Odesolver failed!")
        
    def random_action(self):
        """ Apply a random action to the system 
        
        Returns
        -------
        action: 1x0 1darray[float]
            A (valid) random action applied to the system.
        
        """
        return np.random.rand(self.n_u) * (self.u_max - self.u_min) + self.u_min
        
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
    pend.plot_ellipsoid_trajectory(p,q,False)
        
        
        