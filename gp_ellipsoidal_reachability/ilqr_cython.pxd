# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:38:03 2017

@author: Torsten
"""
cimport numpy as np
import kissgp
#cimport kissgp
#from environment import Environment

cdef class CILQR:
    
    cdef int n_s
    cdef int n_u
    cdef int n_u_env
    cdef int H
    cdef int ilqr_frequency
    cdef int max_iter_ilqr
    cdef int max_iter
    cdef double lamb_max
    cdef double eps_converge
    cdef double lamb_factor 
    cdef double lamb
    cdef double lamb_regularization
    
    #cdef double w_pos
    cdef w_x
    cdef w_u
    cdef control_bounds
    
    cdef np.ndarray u_min
    cdef np.ndarray u_max
    cdef np.ndarray alpha

    
    cdef public cost
    cdef public model
    
    cpdef ilqr(self, np.ndarray[np.double_t,ndim =2 ] x0, np.ndarray[np.double_t,ndim =2 ] U,verbose =*)
    cdef solve_QP(self,H,q,u_min,u_max,x0,np.float32_t gamma = *,
                  np.float32_t beta = *,np.float32_t eps_tol = *,
                  np.float32_t min_rel_imp = *,int max_iter = *)
    
    cdef __decompose_matrix(self,np.ndarray[np.double_t,ndim=2] H,np.ndarray[np.double_t,ndim=2] q,
                            np.ndarray[np.double_t,ndim=2] x,np.ndarray[np.int_t, ndim=1] id_c,np.ndarray[np.int_t, ndim=1] id_f)
    
    cdef __get_clamped_indices(self,np.ndarray[np.double_t,ndim=1] x0,np.ndarray[np.double_t,ndim=1] g,int n_x,np.ndarray[np.double_t, ndim=1] u_max,np.ndarray[np.double_t, ndim=1] u_min)
    
    cdef _armijo_linesearch(self,np.ndarray[np.double_t,ndim=2] H,np.ndarray[np.double_t,ndim=2] q,
                            np.ndarray[np.double_t,ndim=2] g,np.ndarray[np.double_t,ndim=2] x,
                            np.ndarray[np.double_t,ndim=2] delta_x,f_old,u_min,u_max,gamma,beta,min_step =*)
    
    cdef _init_control_bounds(self,u_min,u_max)
    
    cdef clamp(self,x,u_min,u_max)