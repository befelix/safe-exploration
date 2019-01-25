# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:54:53 2017

@author: tkoller

TODO: Untested (But reused)!!
TODO: Undocumented!

"""

import numpy as np
from casadi import mtimes, exp, sum2, repmat, SX, Function, sqrt, vertcat, horzcat
from casadi import reshape as cas_reshape


def _k_rbf(x, y=None, variance=1., lengthscale=None, diag_only=False):
    """ Evaluate the RBF kernel function symbolically using Casadi

    """
    n_x, dim_x = x.shape

    if diag_only:
        ret = SX(n_x, )
        ret[:] = variance
        return ret

    if y is None:
        y = x
    n_y, _ = np.shape(y)

    if lengthscale is None:
        lengthscale = np.ones((dim_x,))

    lens_x = repmat(lengthscale.reshape(1, -1), n_x)
    lens_y = repmat(lengthscale.reshape(1, -1), n_y)

    r = _unscaled_dist(x / lens_x, y / lens_y)

    return variance * exp(-0.5 * r ** 2)


def _k_mat52(x, y=None, variance=1., lengthscale=None, diag_only=False, ARD=False):
    """ Evaluate the Matern52 kernel function symbolically using Casadi"""
    n_x, dim_x = x.shape

    if diag_only:
        ret = SX(n_x, )
        ret[:] = variance
        return ret

    if y is None:
        y = x
    n_y, _ = np.shape(y)

    if lengthscale is None:
        if ARD:
            lengthscale = np.ones((dim_x,))
        else:
            lengthscale = 1.

    if ARD is False:
        lengthscale = lengthscale * np.ones((dim_x,))

    lens_x = repmat(lengthscale.reshape(1, -1), n_x)
    lens_y = repmat(lengthscale.reshape(1, -1), n_y)

    r = _unscaled_dist(x / lens_x, y / lens_y)
    # GPY: self.variance*(1+np.sqrt(5.)*r+5./3*r**2)*np.exp(-np.sqrt(5.)*r)
    return variance * (1. + sqrt(5.) * r + 5. / 3 * r ** 2) * exp(-sqrt(5.) * r)


def _k_lin_rbf(x, hyp, y=None, diag_only=False):
    """ Evaluate the prdocut of linear and rbf kernel function symbolically using Casadi

    """
    prod_rbf_lengthscale = hyp["prod.rbf.lengthscale"]
    prod_rbf_variance = hyp["prod.rbf.variance"]
    prod_linear_variances = hyp["prod.linear.variances"]
    linear_variances = hyp["linear.variances"]

    x_rbf = cas_reshape(x[:, 1], (-1, 1))
    y_rbf = y
    if not y is None:
        y_rbf = cas_reshape(y[:, 1], (-1, 1))

    k_prod_rbf = _k_rbf(x_rbf, y_rbf, prod_rbf_variance, prod_rbf_lengthscale,
                        diag_only)

    x_prod_lin = cas_reshape(x[:, 1], (-1, 1))
    y_prod_lin = y
    if not y is None:
        y_prod_lin = cas_reshape(y[:, 1], (-1, 1))

    k_prod_lin = _k_lin(x_prod_lin, y_prod_lin, prod_linear_variances, diag_only)

    k_linear = _k_lin(x, y, linear_variances, diag_only)

    return k_prod_lin * k_prod_rbf + k_linear


def _k_lin_mat52(x, hyp, y=None, diag_only=False):
    """ Evaluate the custom kernel composed of linear and matern kernels

    Evaluate the kernel
        k_lin*k_mat52 + k_lin
    """

    prod_mat52_lengthscale = hyp["prod.mat52.lengthscale"]
    prod_mat52_variance = hyp["prod.mat52.variance"]
    prod_linear_variances = hyp["prod.linear.variances"]
    linear_variances = hyp["linear.variances"]

    x_m52 = cas_reshape(x[:, 1], (-1, 1))
    y_m52 = y
    if not y is None:
        y_m52 = cas_reshape(y[:, 1], (-1, 1))

    k_prod_mat52 = _k_mat52(x_m52, y_m52, prod_mat52_variance, prod_mat52_lengthscale,
                            diag_only)

    x_prod_lin = cas_reshape(x[:, 1], (-1, 1))
    y_prod_lin = y
    if not y is None:
        y_prod_lin = cas_reshape(y[:, 1], (-1, 1))

    k_prod_lin = _k_lin(x_prod_lin, y_prod_lin, prod_linear_variances, diag_only)

    k_linear = _k_lin(x, y, linear_variances, diag_only)

    return k_prod_lin * k_prod_mat52 + k_linear


def _k_lin(x, y=None, variances=None, diag_only=False):
    """ Evaluate the Linear kernel function symbolically using Casadi

    """
    n_x, dim_x = np.shape(x)

    if variances is None:
        variances = np.ones((dim_x,))

    if diag_only:
        var = repmat(variances.reshape(1, -1), n_x)
        ret = sum2(var * x ** 2)
        return ret

    var_x = sqrt(repmat(variances.reshape(1, -1), n_x))

    if y is None:
        var_y = var_x
        y = x
    else:
        n_y, _ = np.shape(y)
        var_y = sqrt(repmat(variances.reshape(1, -1), n_y))

    return mtimes(x * var_x, (y * var_y).T)


def _unscaled_dist(x, y):
    """ calculate the squared distance between two sets of datapoints



    Source:
    https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/stationary.py
    """
    n_x, _ = np.shape(x)
    n_y, _ = np.shape(y)
    x1sq = sum2(x ** 2)
    x2sq = sum2(y ** 2)
    r2 = -2 * mtimes(x, y.T) + repmat(x1sq, 1, n_y) + repmat(x2sq.T, n_x, 1)

    return sqrt(r2)


def gp_pred(x, kern, beta, x_train, k_inv_training=None, pred_var=True):
    """

    """
    n_pred, _ = np.shape(x)

    k_star = kern(x, y=x_train)
    pred_mu = mtimes(k_star, beta)

    if pred_var:
        if k_inv_training is None:
            raise ValueError("""The inverted kernel matrix is required 
                for computing the predictive variance""")

        k_expl_var = kern(x, y=x_train, diag_only=True)
        pred_sigm = k_expl_var - sum2(mtimes(k_star, k_inv_training) * k_star)

        return pred_mu, pred_sigm

    return pred_mu


def _get_kernel_function(kern_type, hyp):
    """ Return the casadi function for a specific kernel type

    Parameters
    ----------
    kern_type: str
        The identifier of the kernel
    hyp: dict
        The dictionary of hyperparameters with the k,v types
        specified in gp_models.py

    Returns
    -------
        f_pred: SX
            The python function containing the casadi.SX term representing
            the given kern_type kernel
    """

    if kern_type == "rbf":
        return lambda x, y=None, diag_only=False: _k_rbf(x, y=y, diag_only=diag_only,
                                                         **hyp)
    elif kern_type == "lin_rbf":
        return lambda x, y=None, diag_only=False: _k_lin_rbf(x, hyp, y=y,
                                                             diag_only=diag_only)
    elif kern_type == "lin_mat52":
        return lambda x, y=None, diag_only=False: _k_lin_mat52(x, hyp, y=y,
                                                               diag_only=diag_only)
    else:
        raise ValueError("Unknown kernel {}".format(kern_type))


def gp_pred_function(x, x_train, beta, hyp, kern_types, k_inv_training=None,
                     pred_var=True, compute_grads=False):
    """

    """
    n_gps = np.shape(beta)[1]
    inp = SX.sym("input", (x.shape))

    out_dict = dict()
    mu_all = []
    pred_sigma_all = []
    jac_mu_all = []

    for i in range(n_gps):
        kern_i = _get_kernel_function(kern_types[i], hyp[i])
        beta_i = beta[:, i]
        k_inv_i = None
        if not k_inv_training is None:
            k_inv_i = k_inv_training[i]
        if pred_var:
            mu_new, sigma_new = gp_pred(inp, kern_i, beta_i, x_train, k_inv_i, pred_var)
            pred_func = Function("pred_func", [inp], [mu_new, sigma_new], ["inp"],
                                 ["mu_1", "sigma_1"])
            F_1 = pred_func(inp=x)
            pred_sigma = F_1["sigma_1"]
            pred_sigma_all = horzcat(pred_sigma_all, pred_sigma)

        else:
            mu_new = gp_pred(inp, kern_i, beta_i, x_train, k_inv_i, pred_var)
            pred_func = Function("pred_func", [inp], [mu_new], ["inp"], ["mu_1"])
            F_1 = pred_func(inp=x)

        mu_1 = F_1["mu_1"]
        mu_all = horzcat(mu_all, mu_1)

        if compute_grads:
            # jac_func = pred_func.jacobian("inp","mu_1")

            jac_func = pred_func.factory('dmudinp', ['inp'], ['jac:mu_1:inp'])

            F_1_jac = jac_func(inp=x)
            # print(F_1_jac)
            jac_mu = F_1_jac['jac_mu_1_inp']
            jac_mu_all = vertcat(jac_mu_all, jac_mu)

    out_dict["pred_mu"] = mu_all
    if pred_var:
        out_dict["pred_sigma"] = pred_sigma_all
    if compute_grads:
        out_dict["jac_mu"] = jac_mu_all

    return out_dict
