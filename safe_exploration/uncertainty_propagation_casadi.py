# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:18:58 2017

@author: tkoller
"""

from casadi import *


def one_step_taylor(mu_x, ssm, k_ff, sigma_x=None, k_fb=None, a=None, b=None,
                    a_gp_inp_x=None):
    """ One-step uncertainty propagation via first-order taylor approximation

    Parameters
    ----------
    mu: n_s x 1 ndarray[casadi.sym]
            Mean of the gaussian input
    ssm: StateSpaceModel
            The statistical model
    k_ff: n_u x 1 array[float]
            The additive term of the controls
    sigma: n_s x n_s ndarray[casadi.sym]
            The covariance matrix of the gaussian input



    Returns
    -------
    mu_new: n_s x 1 ndarray[casadi.sym]
            Mean of the gaussian output of the uncertainty propagation    sigma_new: n_s x n_s ndarray[casadi.sym]
            The covariance matrix of the gaussian output of the uncertainty propagation

    """
    n_s = np.shape(mu_x)[0]
    n_u = np.shape(k_ff)[0]

    u_p = k_ff
    if a_gp_inp_x is None:
        a_gp_inp_x = MX.eye(n_s)

    n_gp_in, _ = np.shape(a_gp_inp_x)

    x_bar = mtimes(a_gp_inp_x, mu_x)
    z_bar = vertcat(x_bar, u_p)
    if a is None:
        a = MX.eye(n_s)
        b = MX.zeros(n_s, n_u)

    if sigma_x is None:
        pred_mu, pred_var, _ = ssm(x_bar.T, u_p.T)
        lin_prior = mtimes(a, mu_x) + mtimes(b, u_p)
        mu_new = lin_prior + pred_mu

        return mu_new, diag(pred_var), pred_var.T

    mu_g, sigma_g, jac_mu = ssm(x_bar.T, u_p.T)

    jac_mu = horzcat(mtimes(jac_mu[:, :n_gp_in], a_gp_inp_x), jac_mu[:, n_gp_in:])
    # Compute taylor approximation of the posterior
    sigma_u = mtimes(k_fb, mtimes(sigma_x, k_fb.T))  # covariance of control input
    sigma_xu = mtimes(sigma_x, k_fb.T)  # cross-covariance between state and controls

    sigma_z_0 = horzcat(sigma_x, sigma_xu)
    sigma_z_1 = horzcat(sigma_xu.T, sigma_u)

    sigma_z = vertcat(sigma_z_0,
                      sigma_z_1)  # covariance matrix of combined state-control input z

    sigma_zg = mtimes(sigma_z, jac_mu.T)  # cross-covariance between g and z

    sigma_g = diag(sigma_g) + mtimes(jac_mu, mtimes(sigma_z,
                                                    jac_mu.T))  # The addtitional term stemming from the taylor approxiamtion

    sigma_all_0 = horzcat(sigma_z, sigma_zg)
    sigma_all_1 = horzcat(sigma_zg.T, sigma_g.T)

    sigma_all = vertcat(sigma_all_0, sigma_all_1)  # covariance of combined z and g

    lin_trafo_mat = horzcat(a, b, MX.eye(n_s))  # linear trafo matrix

    mu_zg = vertcat(mu_x, k_ff, mu_g)
    mu_new = mtimes(lin_trafo_mat, mu_zg)

    sigma_new = mtimes(lin_trafo_mat, mtimes(sigma_all, lin_trafo_mat.T))

    return mu_new, sigma_new, sigma_g.T


def multi_step_taylor_symbolic(mu_0, ssm, k_ff, k_fb, sigma_0=None, a=None, b=None,
                               a_gp_inp_x=None):
    """ Multi step ahead predictions of the taylor uncertainty propagation



    Parameters
    ----------
    mu_0: n_s x 1 ndarray[casadi.sym | float]
        Initial state
    ssm: StateSpaceModel
            The statistical model
    k_ff: T x n_u ndarray[casadi.sym]
        The feed forward terms to optimize over
    k_fb: n_s x n_u ndarray[casadi.SX]
        The feedback gain (same for each timestep)
    sigma_0: n_s x n_s ndarray[casadi.sym | float]
        The initial uncertainty
    a: n_s x n_s ndarray[float]
        The A matrix of the linear model Ax + Bu
    b: n_s x n_u ndarray[float]
        The B matrix of the linear model Ax + Bu


    Returns
    -------
    mu_all: T x n_s ndarray[casadi.sym | float]
        The means of the uncertainty propagation trajectory
    sigma_all: T x (n_s*n_s) ndarray[casadi.sym | float]
        The variances of the uncertainty propagation trajectory

    """

    if not sigma_0 is None:
        raise NotImplementedError("Still need  to do this")

    n_s = np.shape(mu_0)[0]
    T, n_u = np.shape(k_ff)

    mu_new, sigma_new, gp_sigma_pred = one_step_taylor(mu_0, ssm,
                                                       k_ff[0, :].reshape((n_u, 1)),
                                                       None, None, a, b, a_gp_inp_x)
    mu_all = mu_new.T
    sigma_all = sigma_new.reshape((1, n_s * n_s))
    gp_sigma_pred_all = gp_sigma_pred

    for i in range(T - 1):
        mu_old = mu_new
        sigma_old = sigma_new
        k_ff_i = k_ff[i + 1, :].reshape((n_u, 1))

        mu_new, sigma_new, gp_sigma_pred = one_step_taylor(mu_old, ssm, k_ff_i,
                                                           sigma_old, k_fb[i], a, b,
                                                           a_gp_inp_x)

        mu_all = vertcat(mu_all, mu_new.T)
        sigma_all = vertcat(sigma_all, sigma_new.reshape((1, n_s * n_s)))
        gp_sigma_pred_all = vertcat(gp_sigma_pred_all, gp_sigma_pred)

    return mu_all, sigma_all, gp_sigma_pred_all


def mean_equivalent_multistep(mu_0, ssm, k_ff, k_fb, sigma_0=None, a=None, b=None,
                              a_gp_inp_x=None):
    """ Compute the simple 'mean-equivalent' uncertainty propagation with GPs

    Parameters
    ----------
    mu_0: n_s x 1 ndarray[casadi.sym | float]
        Initial state
    ssm: StateSpaceModel
            The statistical model
    k_ff: T x n_u ndarray[casadi.sym]
        The feed forward terms to optimize over
    a: n_s x n_s ndarray[float]
        The A matrix of the linear model Ax + Bu
    b: n_s x n_u ndarray[float]
        The B matrix of the linear model Ax + Bu


    Returns
    -------
    mu_all: T x n_s ndarray[casadi.sym | float]
        The means of the uncertainty propagation trajectory
    sigma_all: T x (n_s*n_s) ndarray[casadi.sym | float]
        The variances of the uncertainty propagation trajectory



    """

    if not sigma_0 is None:
        raise NotImplementedError("Still need to do this!")

    n_s = np.shape(mu_0)[0]
    T, n_u = np.shape(k_ff)

    mu_new, sigma_new, gp_sigma_pred = one_step_mean_equivalent(mu_0, ssm,
                                                                k_ff[0, :].reshape(
                                                                    (n_u, 1)), None,
                                                                None, a, b, a_gp_inp_x)
    mu_all = mu_new.T
    sigma_all = sigma_new.reshape((1, n_s * n_s))
    gp_sigma_pred_all = gp_sigma_pred
    for i in range(T - 1):
        mu_old = mu_new
        sigma_old = sigma_new
        k_ff_i = k_ff[i + 1, :].reshape((n_u, 1))

        mu_new, sigma_new, gp_sigma_pred = one_step_mean_equivalent(mu_old, ssm, k_ff_i,
                                                                    sigma_old, k_fb[i],
                                                                    a, b, a_gp_inp_x)

        mu_all = vertcat(mu_all, mu_new.T)
        sigma_all = vertcat(sigma_all, sigma_new.reshape((1, n_s * n_s)))
        gp_sigma_pred_all = vertcat(gp_sigma_pred_all, gp_sigma_pred)

    return mu_all, sigma_all, gp_sigma_pred_all


def one_step_mean_equivalent(mu_x, ssm, k_ff, sigma_x=None, k_fb=None, a=None, b=None,
                             a_gp_inp_x=None):
    """ One-step uncertainty propagation via first-order taylor approximation

    Parameters
    ----------
    mu: n_s x 1 ndarray[casadi.sym]
            Mean of the gaussian input
    ssm: StateSpaceModel
            The statistical model
    k_ff: n_u x 1 array[float]
            The additive term of the controls
    sigma: n_s x n_s ndarray[casadi.sym]
            The covariance matrix of the gaussian input




    Returns
    -------
    mu_new: n_s x 1 ndarray[casadi.sym]
            Mean of the gaussian output of the uncertainty propagation
    sigma_new: n_s x n_s ndarray[casadi.sym]
            The covariance matrix of the gaussian output of the uncertainty propagation

    """

    n_s = np.shape(mu_x)[0]
    n_u = np.shape(k_ff)[0]

    u_p = k_ff

    if a_gp_inp_x is None:
        a_gp_inp_x = MX.eye(n_s)
    x_bar = mtimes(a_gp_inp_x, mu_x)
    z_bar = vertcat(x_bar, u_p)
    if a is None:
        a = MX.eye(n_s)
        b = MX.zeros(n_s, n_u)

    if sigma_x is None:
        pred_mu, pred_var, _ = ssm(x_bar.T, u_p.T)
        lin_prior = mtimes(a, mu_x) + mtimes(b, u_p)
        mu_new = lin_prior + pred_mu

        return mu_new, diag(pred_var), pred_var.T

    mu_g, sigma_g, _ = ssm(x_bar.T, u_p.T)

    # Compute taylor approximation of the posterior
    sigma_u = mtimes(k_fb, mtimes(sigma_x, k_fb.T))  # covariance of control input
    sigma_xu = mtimes(sigma_x, k_fb.T)  # cross-covariance between state and controls

    sigma_z_0 = horzcat(sigma_x, sigma_xu)
    sigma_z_1 = horzcat(sigma_xu.T, sigma_u)

    sigma_z = vertcat(sigma_z_0,
                      sigma_z_1)  # covariance matrix of combined state-control input z

    sigma_zg = MX.zeros(n_s + n_u, n_s)  # cross-covariance between g and z

    sigma_all_0 = horzcat(sigma_z, sigma_zg)
    sigma_all_1 = horzcat(sigma_zg.T, diag(sigma_g))

    sigma_all = vertcat(sigma_all_0, sigma_all_1)  # covariance of combined z and g

    lin_trafo_mat = horzcat(a, b, MX.eye(n_s))  # linear trafo matrix

    mu_zg = vertcat(mu_x, k_ff, mu_g)
    mu_new = mtimes(lin_trafo_mat, mu_zg)

    sigma_new = mtimes(lin_trafo_mat, mtimes(sigma_all, lin_trafo_mat.T))

    return mu_new, sigma_new, sigma_g.T
