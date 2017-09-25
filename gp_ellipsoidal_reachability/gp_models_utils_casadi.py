# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:54:53 2017

@author: tkoller

TODO: Untested (But reused)!!
TODO: Undocumented!

"""

from casadi import mtimes, exp, sum1, sum2, repmat, SX


def _k_rbf(x,lenscales,precision,y = None,diag_only = False):
    """ Evaluate the RBF kernel function in Casadi
    
    """
    raise NotImplementedError()
    
    T,_ = x_new.shape
    
    if diag_only:
        ret = SX(T,)
        ret[:] = precision
        return ret
    
    n_x,_ = np.shape(x)
    
    if Y is None:
        y = x
    n_y,_ = np.shape(y)
    

    lens_X = repmat(lenscales.reshape(1,-1),n_x)
    lens_Y = repmat(lenscales.reshape(1,-1),n_y)
    r = _unscaled_dist(X / lens_X,Y / lens_Y)
        
    return precision * exp(-0.5 * r**2)
    
    
def _unscaled_dist(x,y):
    """ calculate the squared distance between two sets of datapoints
    
    
    
    Source:
    https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/stationary.py
    """
    n_x,_ = np.shape(x)
    n_y,_ = np.shape(y)
    x1sq = sum2(x**2)
    x2sq = sum2(y**2)
    r2 = -2 * mtimes(x,y.T) + repmat(x1sq,1,n_y) + repmat(x2sq.T,n_x,1)
    
    
def gp_pred_rbf(x,beta,x_train,lenscales,precision,noise_var,k_inv_training = None,pred_var = True):
    """
    
    """
    n_pred, _ = np.shape(x)
    n_gps = np.shape(beta)[1]
    pred_mu = SX(n_pred,n_gps)
    
    if pred_var:
        if k_training is None:
                raise ValueError("""The inverted kernel matrix is required 
                for computing the predictive variance""") 
        pred_sigm = SX(n_pred,n_gps)
        
    for i in range(n_gps):
        k_star_i = _k_rbf(x,lenscales[i],precision[i],y = x_train)
        pred_mu[:,ii] = mtimes(k_star_i,beta[:,ii])
        
        if pred_var:
            k_expl_var_i = _k_rbf(x,lenscales[i],precision[i],diag_only = True)
            pred_sigm[:,i] = k_expl_var_i - sum2(mtimes(k_star_i,D_var[i])*k_star_i)
            
    if pred_var:
        return pred_mu, pred_var
    return pred_mu
            
def gp_pred_rbf_function(n_pred,n_gps,n_inp,x_train,beta,hyp,k_inv_training = None, pred_var = True, compute_grads = False):
    """
    
    """
    raise NotImplementedError("Still need to implement this!")
    x = SX.sym("input",(n_pred,n_gps))    
    
            
        
        
    