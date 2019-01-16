# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:12:30 2017

@author: tkoller
"""
import os.path

import pytest
import numpy as np
from casadi import SX, Function, vertcat
from casadi import reshape as cas_reshape

from .. import gp_models
from .. import uncertainty_propagation_casadi as prop_casadi


@pytest.fixture(params = [("InvPend",True,True),("InvPend",False,True),
                          ("InvPend",True,True),("InvPend",False,True)])
def before_test_onestep_reachability(request):

    env, init_uncertainty, lin_model = request.param
    if env == "InvPend":
        n_s = 2
        n_u = 1
        path = os.path.join(os.path.dirname(__file__), "invpend_data.npz")
        c_safety = 2
    a = None
    b = None
    if lin_model:
        a = np.random.rand(n_s,n_s)
        b = np.random.rand(n_s,n_u)

    train_data = np.load(path)
    X = train_data["X"]
    y = train_data["y"]
    m = 50
    gp = gp_models.SimpleGPModel(n_s,n_s,n_u,X,y,m,train = True)
    L_mu = np.array([0.1]*n_s)
    L_sigm = np.array([0.1]*n_s)
    k_fb = .1*np.random.rand(n_u,n_s) # need to choose this appropriately later
    k_ff = .1*np.random.rand(n_u,1)

    p = .1*np.random.randn(n_s,1)
    if init_uncertainty:
        q = .2 * np.array([[.5,.2],[.2,.65]]) # reachability based on previous uncertainty
    else:
        q = None # no initial uncertainty

    return p,q,gp,k_fb,k_ff,L_mu,L_sigm,c_safety,a,b


def test_multistep_trajectory(before_test_onestep_reachability):
    """ Compare multi-steps 'by hand' with the function """

    mu_0,_,gp,k_fb,k_ff,L_mu,L_sigm,c_safety,a,b = before_test_onestep_reachability
    T=3
    n_u,n_s = np.shape(k_fb)

    k_fb_cas_single = SX.sym("k_fb_single",(n_u,n_s))
    k_ff_cas_single  = SX.sym("k_ff_single",(n_u,1))
    k_ff_cas_all = SX.sym("k_ff_single",(T,n_u))

    k_fb_cas_all = SX.sym("k_fb_all",(T-1,n_s*n_u))
    k_fb_cas_all_inp = [k_fb_cas_all[i,:].reshape((n_u,n_s)) for i in range(T-1)]
    mu_0_cas = SX.sym("mu_0",(n_s,1))
    sigma_0_cas = SX.sym("sigma_0",(n_s,n_s))

    mu_onestep_no_var_in, sigm_onestep_no_var_in, _ = prop_casadi.one_step_taylor(mu_0_cas,gp,k_ff_cas_single, a = a, b = b)

    mu_one_step, sigm_onestep,_ = prop_casadi.one_step_taylor(mu_0_cas,gp,k_ff_cas_single, k_fb = k_fb_cas_single, sigma_x = sigma_0_cas,a = a, b = b)

    mu_multistep, sigma_multistep, _ = prop_casadi.multi_step_taylor_symbolic(mu_0_cas, gp, k_ff_cas_all, k_fb_cas_all_inp , a = a, b = b)

    on_step_no_var_in = Function("on_step_no_var_in",[mu_0_cas,k_ff_cas_single],[mu_onestep_no_var_in,sigm_onestep_no_var_in])
    one_step = Function("one_step",[mu_0_cas,sigma_0_cas,k_ff_cas_single,k_fb_cas_single],[mu_one_step,sigm_onestep])
    multi_step = Function("multi_step",[mu_0_cas,k_ff_cas_all,k_fb_cas_all],[mu_multistep,sigma_multistep])


    ## TODO: Need mu, sigma as input aswell
    mu_1, sigma_1 = on_step_no_var_in(mu_0,k_ff)

    mu_2, sigma_2 = one_step(mu_1,sigma_1,k_ff,k_fb)
    mu_3, sigma_3 = one_step(mu_2,sigma_2,k_ff,k_fb)

    ## TODO: stack k_ff and k_fb
    k_fb_mult = np.array(cas_reshape(k_fb,(1,n_u*n_s)))
    k_fb_mult = np.array(vertcat(*[k_fb_mult]*(T-1)))


    k_ff_mult = vertcat(*[k_ff]*T)
    mu_all, sigma_all = multi_step(mu_0,k_ff_mult,k_fb_mult)

    assert np.allclose(np.array(mu_all[0,:]),np.array(mu_1).T), "Are the centers of the first prediction the same?"
    assert np.allclose(np.array(mu_all[-1,:]),np.array(mu_3).T), "Are the centers of the final prediction the same?"
    assert np.allclose(cas_reshape(sigma_all[-1,:],(n_s,n_s)),sigma_3), "Are the last covariance matrices the same?"
