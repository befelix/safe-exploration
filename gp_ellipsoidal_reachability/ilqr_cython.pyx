# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:30:04 2017

@author: Torsten
"""

import numpy as np
import warnings
import time
cimport numpy as np
import numpy.linalg as nLa
import sys

cdef class CILQR:
    """ ILQR with control constraints implemented in Cython
    
    
    
    """ 
    def __cinit__(self,model,cost,H = 15, max_iter_ilqr = 3,w_x = 1e3,
                  w_u= 1e2,lamb_factor = 10,lamb = 1.0,
                  u_min = None, u_max = None,alpha = np.power(10,np.linspace(0,-3,8))):
        
        self.n_u = model.n_u
        self.n_s = model.n_s
        self.H = H 
        
        self.cost = cost
        self.model = model
        
        self.max_iter_ilqr = max_iter_ilqr
        self.lamb_factor = lamb_factor
        self.lamb_max = 4000
        self.lamb = lamb
        self.lamb_regularization = 1.8
        
        self.eps_converge = 1e-4
        self.w_x = w_x
        self.w_u = w_u
        
        if u_min is None:
            u_min = model.u_min
        if u_max is None:
            u_max = model.u_max
            
        if not alpha is None:
            self.alpha = alpha
        
        if (not u_min is None) or (not u_max is None):
            self.control_bounds = True
        
        self.u_min,self.u_max = self._init_control_bounds(u_min,u_max)
        
    
    cdef _init_control_bounds(self,u_min,u_max):
        """
        
        """
        
        lb = -1e10
        ub = 1e10
        if u_min is None:
            u_min = np.array([lb]*self.n_u)
        if u_max is None:
            u_max = np.array([ub]*self.n_u)

        assert(len(u_min) == self.n_u)
        assert(len(u_max) == self.n_u)
        
        return u_min,u_max
          
    cpdef ilqr(self, np.ndarray[np.double_t,ndim =2 ] x0, np.ndarray[np.double_t,ndim =2 ] U,verbose = 1): 
        """ use iterative linear quadratic regulation to find a control 
        sequence that minimizes the cost function 
        
        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """

        cdef int ii,tN,n_u,num_states,t
        cdef double dt,lamb
        cdef double alpha_opt = 0
        cdef success_qp = True
        dt = self.model.dt # time step
        tN = U.shape[0] # number of time steps

        n_u = self.n_u # number of degrees of freedom of plant 
        num_states = self.n_s # number of states (position and velocity)
        lamb = self.lamb
        
        
        cdef np.ndarray[np.double_t, ndim=3] l_xx = np.zeros((tN+1, num_states, num_states))
        cdef np.ndarray[np.double_t, ndim=3] l_ux = np.zeros((tN, n_u, num_states))
        cdef np.ndarray[np.double_t, ndim=3] l_uu = np.zeros((tN, n_u, n_u))
        cdef np.ndarray[np.double_t,ndim=2] l = np.zeros((tN+1,1))
        cdef np.ndarray[np.double_t,ndim=2] l_u = np.zeros((tN, n_u))
        cdef np.ndarray[np.double_t,ndim=2] l_x = np.zeros((tN+1, num_states))
        
        cdef np.ndarray[np.double_t, ndim=3] K = np.zeros((tN, n_u, num_states)) # feedback gain
        cdef np.ndarray[np.double_t,ndim=2] k = np.zeros((tN, n_u)) # feedforward modification
        
        cdef np.ndarray[np.double_t, ndim=3] f_x = np.zeros((tN, num_states, num_states))
        cdef np.ndarray[np.double_t, ndim=3] f_u = np.zeros((tN, num_states, n_u))
        cdef np.ndarray[np.double_t,ndim=2] X = np.zeros((tN+1,num_states))
        cdef np.ndarray[np.double_t,ndim=3] A = np.zeros((tN,num_states,num_states))
        cdef np.ndarray[np.double_t,ndim=3] B = np.zeros((tN,num_states,n_u)) 
        
        cdef np.ndarray Q_x,Q_u
        cdef double c,c_new
        
         # regularization parameter
        sim_new_trajectory = True
        
        for ii in range(self.max_iter_ilqr):
            
            if sim_new_trajectory == True: 
                # simulate forward using the current control trajectory
            

                X,A,B = self.model.simulate(x0,U)

                c = self.cost.total_cost(X,U,self.w_x,self.w_u)
                c_old = np.copy(c) # copy for exit condition check
            
                for t in range(tN):
                    # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                    
                    f_x[t] = np.eye(num_states) + A[t]
                    f_u[t] = B[t]
                
                    (l[t], l_x[t], l_xx[t], l_u[t], 
                        l_uu_env, l_ux[t]) = self.cost.cost(X[t,:,None], U[t],t,self.w_x,self.w_u)
                        
                    
                    l_uu[t] = l_uu_env
                    
                    l[t] *= dt
                    l_x[t] *= dt
                    l_xx[t] *= dt
                    l_u[t] *= dt
                    l_uu[t] *= dt
                    l_ux[t] *= dt
                # aaaand for final state
                l[-1], l_x[-1], l_xx[-1] = self.cost.cost_final(X[-1,:,None],self.w_x)
    
                sim_new_trajectory = False

            # optimize things! 
            # initialize Vs with final state cost and set up k, K 
            V = l[-1].copy() # value function
            V_x = l_x[-1].copy() # dV / dx
            V_xx = l_xx[-1].copy() # d^2 V / dx^2
    
            # work backwards to solve for V, Q, k, and K
            success_qp = True
            for t in range(tN-1, -1, -1):
    
                # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x
    
                # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
                Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
                # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
                        
                Q_u = l_u[t] + np.dot(f_u[t].T, V_x)
    
                # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
                # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
                
                # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
                Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
                # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
                Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
                # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
                
                
                Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t])) 
                
                
                # Calculate Q_uu^-1 with regularization term set by 
                # Levenberg-Marquardt heuristic (at end of this loop)
                
                
                if self.control_bounds:
                    
                    Q_uu_qp = Q_uu + lamb * np.eye(self.n_u)                    
                    
                    if t == tN-1:
                        k_init = np.copy(k[t])
                        K_t = np.copy(K[t])
                    else:
                        k_init = np.copy(k[t+1])
                        K_t = np.copy(K[t+1])
                        
                    k[t],Quu_f,id_f,id_c,success_t = self.solve_QP(Q_uu_qp,Q_u.reshape((-1,1)),self.u_min-U[t],self.u_max-U[t],k_init.reshape((-1,1)))
                    
                    if success_t:
                        success_qp = success_qp and success_t
                        K_t = np.zeros((self.n_u,self.n_s))
                        if len(id_f) > 0:
                            K_t[id_f,:] = -nLa.solve(Quu_f,nLa.solve(Quu_f.T,Q_ux[id_f]))
                        
                    else:
                        k[t] = k_init
                    
                    
                    K[t] = K_t
    
                else:
                    try:
                        Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                        Q_uu_evals[Q_uu_evals < 0] = 0.0
                        Q_uu_evals += lamb
                        Q_uu_inv = np.dot(Q_uu_evecs, 
                            np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))
                        
                        # 5b) k = -np.dot(Q_uu^-1, Q_u)
                        k[t] = -np.dot(Q_uu_inv, Q_u)
                        # 5b) K = -np.dot(Q_uu^-1, Q_ux)
                        K[t] = -np.dot(Q_uu_inv, Q_ux)
                    except:
                        # inversion error 
                        if t == tN-1:
                            pass
                        else:
                            k[t] = k[t+1]
                            K[t] = K[t+1]
                            
                        
    
                # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
                # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
                V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
                # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
                V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))
            
            
            # calculate the optimal change to the control trajectory
            c_improved = False
            
            for jj in range(len(self.alpha)):
                Unew = np.zeros((tN, n_u))
                xnew = x0.copy().reshape((-1,))# np.reshape(x0.copy(),(-1,)) # 7a)
                Xnew = X.copy()
                
                #c_test = 0
                for t in range(tN): 
                    Xnew[t] = xnew
                    xnew_diff = xnew - X[t]
                    Unew[t] = U[t] + self.alpha[jj]*k[t] + np.dot(K[t], xnew_diff.T) # 7b)
                    
                    if self.control_bounds:
                        Unew[t] = self.clamp(Unew[t],self.u_min,self.u_max)
                    
                    # given this u, find our next state
                    x_diff = self.model.forward_step(xnew[:,None], Unew[t,:,None]) # 7c)
                    xnew = xnew + x_diff.reshape((-1,))
                    
                Xnew[-1] = xnew
                #c_test+= self.environment.cost_final(X[-1],w_pos = self.w_pos,w_vel = self.w_vel)[0]
                
                c_new = self.cost.total_cost(Xnew,Unew,self.w_x,self.w_u)
                c_improved = (c_new < c)
                
                if c_improved:
                    alpha_opt = self.alpha[jj]
                    if verbose > 0 :
                        print("cost in ( iteration {} ; line-search {} ) improved for alpha = {}".format(ii,jj,self.alpha[jj]))
                        print("old cost: {}".format(c))
                        print("new cost: {}".format(c_new))
                    
                    break
                
            # Levenberg-Marquardt heuristic
            if c_improved: 
                # decrease lambda (get closer to Newton's method)
            
                if success_qp:
                    lamb /= self.lamb_factor
                else:
                    lamb *= self.lamb_regularization
                X = np.copy(Xnew) # update trajectory 
                U = np.copy(Unew) # update control signal

                c_old = np.copy(c)
                c = np.copy(c_new)
    
                sim_new_trajectory = True # do another rollout
    
                # print("iteration = %d; Cost = %.4f;"%(ii, costnew) + 
                #         " logLambda = %.1f"%np.log(lamb))
                # check to see if update is small enough to exit
                if ii > 0 and ((abs(c_old-c)/c) < self.eps_converge):
                    print("Converged at iteration = %d; Cost = %.4f;"%(ii,c_new) + 
                            " logLambda = %.1f"%np.log(lamb))
                    print("-------------------------CONVERGENCE------------------------------------")
                    break
    
            else:
                               
                # increase lambda (get closer to gradient descent)
                lamb *= self.lamb_factor
                # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lamb_max: 
                    print("lambda > max_lambda at iteration = %d;"%ii + 
                        " Cost = %.4f; logLambda = %.1f"%(c, 
                                                          np.log(lamb)))
                    break
        return X,U,c,K,k,alpha_opt
        
    cdef solve_QP(self,H,q,u_min,u_max,x0,np.float32_t gamma = 0.1, np.float32_t beta = 0.8, np.float32_t eps_tol = 1e-8, np.float32_t min_rel_imp =1e-8, int max_iter = 100):
        """ solve a simple QP
        
        """
        #print(x0)
        cdef np.ndarray[np.double_t,ndim=2] g_clamped,g_clamped_free,delta_xf,delta_x,x_c_all
        cdef int result,ii,n_x
        cdef double f_new,f_old        
        cdef factorize = False
        cdef success = True
        
        result = 0
        n_x = np.shape(x0)[0]

        x0 = self.clamp(x0,u_min,u_max)
        x = x0

        cdef np.ndarray[np.double_t,ndim=2] g = q + np.dot(H,x)
        
        
        f_new =  0.5*np.dot(x.T,np.dot(H,x)) + np.dot(q.T,x)       
        
        cdef np.ndarray[np.double_t,ndim=2] H_free = np.zeros((n_x,n_x))
        
        cdef np.ndarray[np.int_t, ndim=1] id_c_old = np.array([],dtype = np.int_)
        cdef np.ndarray[np.int_t, ndim=1] id_f_old = np.array([],dtype = np.int_)
        
        for ii in range(max_iter):
            
            if result > 0:
                break
            
            
            if (ii > 1) and (f_old-f_new) < min_rel_imp * np.abs(f_old):
                result = 4
                break
            
            f_old = f_new
            id_c,id_f = self.__get_clamped_indices(x.reshape(-1,),g.reshape(-1,),n_x,u_max,u_min)
            
            
            if len(id_f) == 0:
                result = 1
                break
            
            H_ff,q_f,q_c,x_f,x_c = self.__decompose_matrix(H,q,x,id_c,id_f)
            qq = q_f
            
            if ii == 0:
                factorize = True
            else:
                # this works if there are no duplicates in either list
                # which should be the case anyways
                factorize = not (set(id_c_old) == set(id_c))
            
            if factorize:
                try:
                    H_free = nLa.cholesky(H_ff)
                except:
                    
                    warnings.warn("Matrix not positive-definite!")
                    success = False
                    break
            if nLa.norm(g[id_f,0]) < eps_tol:
                result = 5
                break
            
            
            
            x_c_all = np.zeros((n_x,1))
            
            if len(id_c) > 0:
                x_c_all[id_c,0] = x_c.reshape(-1,)

            
            g_clampled = g + np.dot(H,x_c_all)

            g_clamped_free = g_clampled[id_f].reshape((-1,1))

            delta_xf = -nLa.solve(H_free,nLa.solve(H_free.T,g_clamped_free)) - x_f
            
            delta_x = np.zeros((n_x,1))
            delta_x[id_f,0] = delta_xf.squeeze()
                   
            
            x_plus,f_new,result_armijo = self._armijo_linesearch(H,q,g,x,delta_x,f_old,u_min,u_max,gamma,beta)
            
            if result_armijo >0:
                result = result_armijo
                break
                
            x = x_plus


        return x.squeeze(),H_free,id_f,id_c,success
        
    cdef clamp(self,x,u_min,u_max):
        """
        
        """
        shape_x = np.shape(x)
        x = np.clip(x.squeeze(),u_min,u_max)
        return x.reshape(shape_x)
        
    cdef _armijo_linesearch(self,np.ndarray[np.double_t,ndim=2] H,np.ndarray[np.double_t,ndim=2] q,
                            np.ndarray[np.double_t,ndim=2] g, np.ndarray[np.double_t,ndim=2] x,
                            np.ndarray[np.double_t,ndim=2] delta_x,f_old,u_min,u_max,gamma,beta,min_step = 1e-8):
        """
        
        
        """
        cdef double alpha = 1.0
        cdef int n_step = 0
        cdef int result = 0
         
        cdef np.ndarray[np.double_t,ndim=2] x_plus = self.clamp(x + alpha * delta_x,u_min,u_max)

        cdef double f_new = 0.5*np.dot(x_plus.T,np.dot(H,x_plus)) + np.dot(q.T,x_plus)
        
        cdef np.ndarray[np.double_t,ndim=2] g_dx = np.dot(g.T,delta_x)
        

        while (f_new-f_old)/(alpha*g_dx) < gamma:
            n_step += 1
            alpha = alpha * beta
            x_plus = self.clamp(x + alpha * delta_x,u_min,u_max)
            
            f_new = 0.5*np.dot(x_plus.T,np.dot(H,x_plus)) + np.dot(q.T,x_plus)
            
            if alpha < min_step:
                result = 2
                break
            
        return x_plus,f_new,result
            
    cdef __get_clamped_indices(self,np.ndarray[np.double_t,ndim=1]  x0,np.ndarray[np.double_t,ndim=1] g,int n_x,np.ndarray[np.double_t, ndim=1] u_max,np.ndarray[np.double_t, ndim=1] u_min):
        """
        
        """
        cdef double tol = 1e-8
        
        cdef np.ndarray[np.int_t, ndim=2] c_upper = np.argwhere((x0 == u_max) & (g < 0))
        cdef np.ndarray[np.int_t, ndim=2] c_lower = np.argwhere((x0 == u_min) & (g > 0))

        cdef np.ndarray[np.int_t, ndim=1] id_c = np.append(c_upper,c_lower)

        cdef np.ndarray[np.int_t, ndim=1] id_f = np.setdiff1d(np.arange(n_x),id_c)

        
        return id_c,id_f
        
    cdef __decompose_matrix(self,np.ndarray[np.double_t,ndim=2]  H,np.ndarray[np.double_t,ndim=2] q,
                            np.ndarray[np.double_t,ndim=2] x,np.ndarray[np.int_t, ndim=1] id_c,np.ndarray[np.int_t, ndim=1]id_f):
        """
        
        """
        H_ff = H[np.ix_(id_f,id_f)]
        q_f = q[id_f]
        #print(id_c)
        if len(id_c) > 0:
            q_c = q[id_c]
            x_c = x[id_c]
        else:
            q_c = []
            x_c = []
       
        x_f = x[id_f]
    
        return H_ff,q_f,q_c,x_f,x_c
        
        
        