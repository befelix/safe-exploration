# -*- coding: utf-8 -*-
"""
Test the gp_models_utils_casadi.py module by comparing it to the
corresponding GPy implementation. Most tests will pass if they
show the same behaviour as the gpy implementation

@author: tkoller
"""

import numpy as np
import pytest
from GPy.kern import RBF, Linear, Matern52
from casadi import Function, SX

from ..gp_models_utils_casadi import _unscaled_dist, _k_rbf, _k_lin, _k_lin_rbf, \
    _k_mat52


@pytest.fixture(params=[(1, 10, 3), (5, 1, 3), (10, 20, 5), (10, 0, 5)])
def before_gp_utils_casadi_test_rbf(request):
    """ """
    n_x, n_y, n_dim = request.param
    x = np.random.rand(n_x, n_dim)
    y = None
    if n_y > 0:
        y = np.random.rand(n_y, n_dim)

    return x, y, n_dim


def test_unscaled_dist(before_gp_utils_casadi_test_rbf):
    """ Does _unscaled_dist show the same behaviour as the GPy implementation"""
    tol = 1e-6
    x_inp, y_inp, n_dim = before_gp_utils_casadi_test_rbf

    kern_rbf = RBF(n_dim)
    x = SX.sym("x", np.shape(x_inp))
    if y_inp is None:
        f = Function("f", [x], [_unscaled_dist(x, x)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y", np.shape(y_inp))
        f = Function("f", [x, y], [_unscaled_dist(x, y)])
        f_out_casadi = f(x_inp, y_inp)
    f_out_gpy = kern_rbf._unscaled_dist(x_inp, y_inp)

    assert np.all(np.isclose(f_out_casadi, f_out_gpy))


def test_k_rbf(before_gp_utils_casadi_test_rbf):
    """ Does _k_rbf_ show the same behaviours as the GPy implementation?"""

    x_inp, y_inp, n_dim = before_gp_utils_casadi_test_rbf
    ls = np.random.rand(n_dim, ) + 1
    rbf_var = np.random.rand() + 1
    kern_rbf = RBF(n_dim, rbf_var, ls, True)
    x = SX.sym("x", np.shape(x_inp))

    if y_inp is None:
        f = Function("f", [x], [_k_rbf(x, None, rbf_var, ls)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y", np.shape(y_inp))
        f = Function("f", [x, y], [_k_rbf(x, y, rbf_var, ls)])
        f_out_casadi = f(x_inp, y_inp)

    f_out_gpy = kern_rbf.K(x_inp, y_inp)
    assert np.all(np.isclose(f_out_casadi, f_out_gpy))


def test_k_lin(before_gp_utils_casadi_test_rbf):
    """ Does _k_lin show the same behaviours as the GPy implementation?"""

    x_inp, y_inp, n_dim = before_gp_utils_casadi_test_rbf
    ls = np.random.rand(n_dim, ) + 1

    kern_lin = Linear(n_dim, ls, True)
    x = SX.sym("x", np.shape(x_inp))

    if y_inp is None:
        f = Function("f", [x], [_k_lin(x, None, ls)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y", np.shape(y_inp))
        f = Function("f", [x, y], [_k_lin(x, y, ls)])
        f_out_casadi = f(x_inp, y_inp)

    f_out_gpy = kern_lin.K(x_inp, y_inp)
    assert np.all(np.isclose(f_out_casadi, f_out_gpy))


def test_k_mat52(before_gp_utils_casadi_test_rbf):
    """ Does _k_mat52 show the same behaviours as the GPy implementation? """
    x_inp, y_inp, n_dim = before_gp_utils_casadi_test_rbf
    ls = np.random.rand(n_dim, ) + 1

    variance = np.random.rand()

    kern = Matern52(n_dim, variance, ls, True)
    x = SX.sym("x", np.shape(x_inp))

    if y_inp is None:
        f = Function("f", [x], [_k_mat52(x, None, variance, ls)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y", np.shape(y_inp))
        f = Function("f", [x, y], [_k_mat52(x, y, variance, ls)])
        f_out_casadi = f(x_inp, y_inp)

    f_out_gpy = kern.K(x_inp, y_inp)
    assert np.all(np.isclose(f_out_casadi, f_out_gpy))


def test_k_lin_rbf(before_gp_utils_casadi_test_rbf):
    """ Does _k_rbf_ show the same behaviours as the GPy implementation?"""

    x_inp, y_inp, n_dim = before_gp_utils_casadi_test_rbf
    ls_lin = np.random.rand(n_dim, ) + 1
    ls_prod_lin = np.random.rand(1, ) + 1
    ls_rbf = np.random.rand(1, ) + 1

    rbf_var = np.random.rand() + 1
    hyp = dict()
    hyp["linear.variances"] = ls_lin
    hyp["prod.rbf.lengthscale"] = ls_rbf
    hyp["prod.rbf.variance"] = rbf_var
    hyp["prod.linear.variances"] = ls_prod_lin

    kern_lin = Linear(1, ls_prod_lin, True, active_dims=[1]) * RBF(1, rbf_var, ls_rbf,
                                                                   True, active_dims=[
            1]) + Linear(n_dim, ls_lin, True)

    x = SX.sym("x", np.shape(x_inp))

    if y_inp is None:
        f = Function("f", [x], [_k_lin_rbf(x, hyp)])
        f_out_casadi = f(x_inp)
    else:
        y = SX.sym("y", np.shape(y_inp))
        f = Function("f", [x, y], [_k_lin_rbf(x, hyp, y)])
        f_out_casadi = f(x_inp, y_inp)

    f_out_gpy = kern_lin.K(x_inp, y_inp)
    assert np.all(np.isclose(f_out_casadi, f_out_gpy))
