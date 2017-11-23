# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
"""
import sys
import numpy as np
import numpy.linalg as nLa
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
    
    def __init__(self,n_s,n_u,X=None,y=None,m=None,kern_types = None, hyp = None, train = False):
        """ Initialize GP Model ( possibly without training set)
        
        Parameters
        ----------
            X (np.ndarray[float], optional): Training inputs
            y (np.ndarray[float], optional): Training targets
            m (int, optional): number of datapoints to be selected (randomly) 
                    from the training set 
            
        """
        self.n_s = n_s
        self.n_u = n_u
        self.gp_trained = False
        self.m = m
        
        
        self._init_kernel_function(kern_types,hyp)
                
        
        if train:
            self.train(X,y,m)
                 
    @classmethod 
    def from_dict(cls,gp_dict):
        """ Initialize GP using data from a dict
        
        Initialized the SimpleGPModel from a dictionary containing
        the necessary information.
        
        Parameters
        ----------    
        gp_dict: dict
            The dictionary containing the following entries:
            
        """
        
        if "data_path" in gp_dict:
            data_path = gp_dict["data_path"]
            data = np.load(data_path)
            x = data["X"]
            y = data["y"]
        elif "x" in gp_dict and "y" in gp_dict:
            x = gp_dict["x"]
            y = gp_dict["y"]
        else:
            raise ValueError("""gp_dict either needs a data_path or 
            the data itself (key 'data_path' or keys 'x' and 'y')""")
        
        if "prior_model" in gp_dict:
            prior_model = gp_dict["prior_model"]
            y = y - prior_model(x)
            
        n_s = np.shape(y)[1]
        n_u = np.shape(x)[1]-n_s
        
        m = None
        if "m" in gp_dict:
            m = gp_dict["m"]
        
        kern_type = None
        if "kern_types" in gp_dict:
            kern_types = gp_dict["kern_types"]
        
        train = False
        if "train" in gp_dict:
            train = gp_dict["train"]
            
        hyp = None
        if "hyp" in gp_dict:
            hyp = gp_dict["hyp"]
            
        return cls(n_s,n_u,x,y,m,kern_types,hyp,train)
        
    def to_dict(self):
        """ return a dict summarizing the object """
        gp_dict = dict()
        gp_dict["x_train"] = self.x_train        
        gp_dict["y_train"] = self.y_train
        gp_dict["kern_type"] = self.kern_type
        gp_dict["hyp"] = self.hyp
        gp_dict["beta"] = self.beta
        gp_dict["inv_K"] = self.inv_K

        return gp_dict
        
    def train(self,X,y,m = None):
        """ Train a GP for each state dimension
        
        Args:
            X: Training inputs of size [N, n_s + n_u]
            y: Training targets of size [N, n_s]
        """
        n_data, _ = np.shape(X)

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
            kern = self.base_kerns[i]
            y_i = y_train[:,i].reshape(-1,1)
            model_gp = GPy.models.GPRegression(X_train,y_i,kernel = kern)
            model_gp.optimize(max_iters = 1000,messages=True)
            
            post = model_gp.posterior
            inv_K[i] = post.woodbury_inv
            beta[:,i] = post.woodbury_vector.reshape(-1,)
            process_noise[i] = model_gp.Gaussian_noise.variance
            gps[i] = model_gp
                       
        # create a dictionary of kernel paramters
        self.hyp = self._create_hyp_dict(gps,self.kern_types)
        
        #update the class attributes      
        self.inv_K = inv_K
        self.beta = beta
        self.gps = gps
        self.gp_trained = True
        self.x_train = X_train
        self.y_train = y_train
        
    def update_model(self, x, y, train = True, replace_old = True):
        """ Update the model based on the current settings and new data 
        
        Parameters
        ----------
        x: n x (n_s + n_u) array[float]
            The training set
        y: n x n_s
            The training targets
        train: bool, optional
            If this is set to TRUE the hyperparameters are re-optimized
        """
        if replace_old:
            x_new = x 
            y_new = y
        else:
            x_new = np.vstack((self.x_train,x))
            y_new = np.vstack((self.y_train,y))
            
        if train:
            self.train(x_new,y_new,self.m)
        else:
            n_data = np.shape(x_new)[0]
            inv_K = [None]*self.n_s
            beta = np.empty((n_data,self.n_s))
            for i in range(self.n_s):
                self.gps[i].set_XY(x_new,y_new[:,i].reshape(-1,1))
                post = self.gps[i].posterior
                inv_K[i] = post.woodbury_inv
                beta[:,i] = post.woodbury_vector.reshape(-1,)
            
            self.x_train = x_new
            self.y_train = y_new
            self.inv_K = inv_K
            self.beta = beta
            
    def _init_kernel_function(self,kern_types = None, hyp = None):
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
        kerns = [None]*self.n_s
        
        if hyp is None:
            hyp = [None]*self.n_s
            
        if kern_types is None:
            kern_types = [None]*self.n_s
            for i in range(self.n_s):
                kern_types[i] = "rbf"
                kerns[i] = RBF(input_dim, ARD = True)
                
        else:
            for i in range(self.n_s):
                if kern_types[i] == "rbf":
                    kern_i = RBF(input_dim, ARD = True)
                    hyp_i = hyp[i]
                    if not hyp_i is None:
                        if "rbf_lengthscales" in hyp_i:
                            kern_i.lengthscale = hyp_i["rbf_lengthscales"]
                            kern_i.lengthscale.fix()
                        
                        if "rbf_variance" in hyp_i:
                            kern_i.variance = hyp_i["rbf_variance"]
                            kern_i.variance.fix()
                    kerns[i] = kern_i        
                elif kern_type == "prod_lin_rbf":
                    raise NotImplementedError("")
                else:
                    raise ValueError("kernel type '{}' not supported".format(kern_type))
        self.base_kerns = kerns
        self.kern_types = kern_types
        
    def _create_hyp_dict(self,gps,kern_types):
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
       
        for i in range(self.n_s):
                if kern_types[i] == "rbf":          
                    hyp_i = dict()
                    hyp_i["rbf_lengthscales"] = np.reshape(gps[i].kern.lengthscale,(-1,))
                    hyp_i["rbf_variance"] = gps[i].kern.variance
                    hyp[i] = hyp_i
            
                elif kern_types[i] == "prod_lin_rbf":
                
                    hyp_i = dict()
                    hyp_i["rbf_lengthscales"] = np.reshape(gps[i].kern.rbf.lengthscale,(-1,))
                    hyp_i["rbf_variance"] = gps[i].kern.rbf.variance
                    hyp_i["lin_variances"] = np.array([gps[i].kern.linear.variances]*(self.n_s+self.n_u))
                    hyp[i] = hyp_i
                else:
                    raise ValueError("kernel type not supported")
        
        return hyp
                
    def predict(self,x_new,quantiles = None,compute_gradients = False):
        """ Compute the predictive mean and variance for a set of test inputs
        
        
        """
        assert self.gp_trained,"Cannot predict, need to train the GP first!" 
        
        T = np.shape(x_new)[0]
        y_mu_pred = np.empty((T,self.n_s))
        y_sigm_pred = np.empty((T,self.n_s))
        
        
        for i in range(self.n_s):
            
            if quantiles is None:
                y_mu_pred[:,i],y_sigm_pred[:,i] = self.gps[i].predict_noiseless(x_new)
            else:
                raise NotImplementedError()
                
        if compute_gradients:
            grad_mu = self.predictive_gradients(x_new)
            return y_mu_pred, y_sigm_pred, grad_mu
            
        return y_mu_pred,y_sigm_pred
        
    def predict_casadi_symbolic(self,x_new,compute_grads = False):
        """ Return a symbolic casadi function representing predictive mean/variance
        
        """
        out_dict = gp_pred_function(x_new,self.x_train,self.beta,self.hyp,self.kern_types,self.inv_K,True,compute_grads)
        mu_new = out_dict["pred_mu"]
        sigma_new = out_dict["pred_sigma"]
        if compute_grads:
            jac_mu = out_dict["jac_mu"]
            return mu_new, sigma_new, jac_mu
            
        return mu_new, sigma_new
             
    def predictive_gradients(self,x_new,grad_sigma = False):
        """ Compute the gradients of the predictive mean/variance w.r.t. inputs
        
        Parameters
        ----------
        x_new: T x (n_s + n_u) array[float]
            The test inputs to compute the gradients at
        grad_sigma: bool, optional
            Additionaly returns the gradients of the predictive variance w.r.t. the inputs if
            this is set to TRUE
        
        """

        if grad_sigma:
            ## would be easy to implement since it is returend by GPys predictive gradient
            ## but we dont need it right now
            raise NotImplementedError("Gradient of sigma not implemented")
        
        T = np.shape(x_new)[0]
        
        grad_mu_pred = np.empty([T,self.n_s,self.n_s+self.n_u])
        
        for i in range(self.n_s):
            g_mu_pred,_ = self.gps[i].predictive_gradients(x_new)
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
        
    def information_gain(self,x = None):
        """ """
        
        if x is None:
            x = self.x_train
            y = self.y_train
            
        n_data = np.shape(x)[0]
        inf_gain_x_f = [None]*self.n_s
        for i in range(self.n_s):
            noise_var_i = float(self.gps[i].Gaussian_noise.variance)
            inf_gain_x_f[i] = np.log(nLa.det(np.eye(n_data) + (1/noise_var_i)*self.gps[i].posterior._K))
        
        return inf_gain_x_f