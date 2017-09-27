# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
"""

import numpy as np
import GPy
import casadi as cas
import casadi.tools as ctools

from gp_models_utils_casadi import gp_pred_function
from GPy.kern import RBF, Linear

class SimpleGPModel():
    """ Simple Wrapper around GPy 
    
    Wrapper around the GPy library
    to get predictions per dimension.
    
    TODO: Shouldnt be allowed to initialize model without training data!
    
    Attributes:
        gp_trained (bool): Is set to TRUE once the train() method 
            was called. 
        n_s (int): number of state dimensions of the dynamic system
        n_u (int): number of action/control dimensions of the dynamic system
        gps (list[GPy.core.GP]): A list of length n_s with the trained GPs
        
    """
    
    def __init__(self,X=None,y=None,m=None):
        """ Initialize GP Model ( possibly without training set)
        
        Args:
            X (np.ndarray[float], optional): Training inputs
            y (np.ndarray[float], optional): Training targets
            m (int, optional): number of datapoints to be selected (randomly) 
                    from the training set 
            
        """
        
        self.gp_trained = False
        
        if (not X is None) and (not y is None):
            self.train(X,y,m=m)
                 
    def train(self,X,y,m = None,kern_type = "prod_lin_rbf"):
        """ Train a GP for each state dimension
        
        Args:
            X: Training inputs of size [N, n_s + n_u]
            y: Training targets of size [N, n_s]
        """
        n_data, input_dim = np.shape(X)
        _,self.n_s = np.shape(y)
        self.n_u = input_dim - self.n_s
        
        if not m is None:
            if n_data < m:
                warnings.warn("""The desired number of datapoints is not available. Dataset consist of {}
                       Datapoints! """.format(n_data))
                X_train = X
                y_train = y
            else:
                idx = np.random.choice(n_data,size=m,replace = False)
                X_train = X[idx,:]
                y_train = y[idx,:]
              
                n_data = m
        else:
            X_train = X
            y_train = y
        
        beta = np.empty((n_data,self.n_s))
        inv_K = [None]*self.n_s
        process_noise = np.empty((self.n_s,))
        gps = [None]*self.n_s
        
        for i in range(self.n_s):
            kern = self._init_kernel_function(kern_type)
            y_i = y_train[:,i].reshape(-1,1)
            model_gp = GPy.models.GPRegression(X_train,y_i,kernel = kern)
            model_gp.optimize(max_iters = 1000,messages=True)
            
            post = model_gp.posterior
            inv_K[i] = post.woodbury_inv
            beta[:,i] = post.woodbury_vector.reshape(-1,)
            process_noise[i] = model_gp.Gaussian_noise.variance
            gps[i] = model_gp
                       
        # create a dictionary of kernel paramters
        self.hyp = self._create_hyp_dict(gps,kern_type)
        #update the class attributes      
        self.inv_K = inv_K
        self.beta = beta
        self.gps = gps
        self.gp_trained = True
        self.x_train = X_train
        self.kern_type = kern_type
        
    def _init_kernel_function(self,kern_type):
        """ Initialize GPy kernel functions based on name. Check if supported.
        
        Utility function to return a kernel based on its type name.
        Checks if the kernel type is supported.
        
        Parameters
        ----------
        kern_type: str
            The name of the kernel
            
        Returns
        -------
        kern: GPy.Kern
            The Gpy kernel function   
        """
        input_dim = self.n_s+self.n_u
        if kern_type == "rbf":
            return RBF(input_dim, ARD = True)
        elif kern_type == "prod_lin_rbf":
            return RBF(input_dim, ARD = True)*Linear(input_dim, ARD = True)
        else:
            raise ValueError("kernel type not supported")
            
    def _create_hyp_dict(self,gps,kern_type):
        """ Create a hyperparameter dict from the individual supported kernels 
        
        Parameters
        ----------
        gps: n_s x 0 array_like[GPy.GP]
            The list of trained GPs
        kern_type:
            The kernel identifier
            
        Returns
        -------
        hyp: list[dict]
            A list of dictionaries containing the hyperparameters of the kernel type
            for each dimension.
        """
        
        hyp = [None]*self.n_s
        if kern_type == "rbf":          
            for i in range(self.n_s):
                hyp_i = dict()
                hyp_i["rbf_lengthscales"] = np.reshape(gps[i].kern.lengthscale,(-1,))
                hyp_i["rbf_variance"] = gps[i].kern.variance
                hyp[i] = hyp_i
            
        elif kern_type == "prod_lin_rbf":
            for i in range(self.n_s):
                hyp_i = dict()
                hyp_i["rbf_lengthscales"] = np.reshape(gps[i].kern.rbf.lengthscale,(-1,))
                hyp_i["rbf_variance"] = gps[i].kern.rbf.variance
                hyp_i["lin_variances"] = np.reshape(gps[i].kern.linear.variances,(-1,))
                hyp[i] = hyp_i
        else:
            raise ValueError("kernel type not supported")
        
        return hyp
                
    def predict(self,X_new,quantiles = None):
        """ Compute the predictive mean and variance for a set of test inputs
        
        
        """
        assert self.gp_trained,"Cannot predict, need to train the GP first!" 
        
        T = np.shape(X_new)[0]
        y_mu_pred = np.empty((T,self.n_s))
        y_sigm_pred = np.empty((T,self.n_s))
        
        
        for i in range(self.n_s):
            
            if quantiles is None:
                y_mu_pred[:,i],y_sigm_pred[:,i] = self.gps[i].predict_noiseless(X_new)
            else:
                raise NotImplementedError()
        return y_mu_pred,y_sigm_pred
        
    def predict_casadi_symbolic(self,x_new,compute_grads = False):
        """ Return a symbolic casadi function representing predictive mean/variance
        
        """
        out_dict = gp_pred_function(x_new,self.x_train,self.beta,self.hyp,self.kern_type,self.inv_K,True,compute_grads)
        mu_new = out_dict["pred_mu"]
        sigma_new = out_dict["pred_sigma"]
        if compute_grads:
            jac_mu = out_dict["jac_mu"]
            return mu_new, sigma_new, jac_mu
            
        return mu_new, sigma_new
             
    def init_casadi(self):
        """
        
        """
        raise NotImplementedError()
             
    def predictive_gradients(self,X_new,grad_sigma = False):
        """ Compute the gradients of the predictive mean/variance w.r.t. inputs
        
        """

        if grad_sigma:
            ## would be easy to implement since it is returend by GPys predictive gradient
            ## but we dont need it right now
            raise NotImplementedError("Gradient of sigma not implemented")
        
        T = np.shape(X_new)[0]
        
        grad_mu_pred = np.empty([T,self.n_s,self.n_s+self.n_u])
        
        for i in range(self.n_s):
            g_mu_pred,_ = self.gps[i].predictive_gradients(X_new)
            grad_mu_pred[:,i,:] = g_mu_pred[:,:,0]
            
        return grad_mu_pred
        
    def sample_from_gp(self, inp, size = 10):
        """ Sample from GP predictive distribution
        
        
        Args:
            inp (numpy.ndarray[float]): array of shape n x (n_s + n_u); the
                test inputs 
            size (int, optional): number of samples per test point
            
        Returns:
            S (numpy.ndarray[float]): array of shape n x size x n_s; array of size samples
            of the posterior distribution per test input
            
        """
        
        n = np.shape(inp)[0]
        S = np.empty((n,size,self.n_s))        
        
        for i in range(self.n_s):
            S[:,:,i] = self.gps[i].posterior_samples_f(inp,size=size,full_cov = False)
            
        return S