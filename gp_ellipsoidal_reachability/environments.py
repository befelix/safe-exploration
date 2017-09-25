# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:16:45 2017

@author: tkoller
"""
import abc
import numpy as np

class Environment:
    """ Base class for environments
    
    
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,name, n_s, n_u, init_std = 0.01):
        """
        
        """
        self.name = name
        self.n_s = n_s
        self.n_u = n_u
        self.start_state = np.array([0.0]*n_s)
        self.is_initialized = False
        self.iteration = 0
        self.init_std = 0.01
        
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
    """ The inverted Pendulum environment
    
    """
    def __init__(self,name = "InvertedPendulum",n_s = 2, n_u = 1):
        """
        
        Parameters
        ----------
        name: str
            The name of the system
        n_s: int
            The number of states
        n_u: int 
            The number of actions
        """
        super(InvertedPendulum,self).__init__(name,n_s,n_u)
    
    def reset(self,mean = None, std = None):
        """ Reset the system.
        
        
        """
        super(InvertedPendulum,self).reset()
        self.current_state = self._sample_start_state(mean = mean,std = std)
        self.odesolver.set_initial_value(self.current_state,0.0)
        
        return self.state_to_obs(self.current_state)
        