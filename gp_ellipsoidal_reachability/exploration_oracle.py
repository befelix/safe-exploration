# -*- coding: utf-8 -*-
"""
Implements different exploration oracles which provide informative samples
or exploration objectives to be used in a MPC setting.

Created on Tue Nov 14 10:08:45 2017

@author: tkoller
"""

class MPCExplorationOracle:
    """ Oracle which finds informative samples
    similar to the GP-UCB setting but with safety constraint and MPC setting
    
    Attributes
    ----------
    gp: SimpleGPModel
        The underlying GP
    
    """
    
    
    def __init__(self,safempc):
        """ Initialize with a pre-defined safempc object"""
        raise NotImplementedError()
        
    def init_solver(self):
        """ Init the casadi sovlver"""
        raise NotImplementedError()
        
    def find_max_variance(self):
        """ Find the most informative sample in the space constrained by the mpc structure"""
        raise NotImplementedError()