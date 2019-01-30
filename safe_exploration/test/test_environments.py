# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:39 2017

@author: tkoller
"""
import builtins
import numpy as np
import pytest
import sys
from scipy.optimize import approx_fprime

from ..environments import InvertedPendulum, CartPole
from ..utils import sample_inside_polytope

np.random.seed(0)


@pytest.fixture(params=[InvertedPendulum(), CartPole()])
def before_test_inv_pend(request):
    env = request.param

    return env

@pytest.fixture
def no_matplotlib(monkeypatch):
    """ Mock an import error for matplotlib"""
    import_orig = builtins.__import__
    def mocked_import(name, globals, locals, fromlist, level):

        if name == 'matplotlib':
            raise ImportError("This is a mocked import error")
        return import_orig(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', mocked_import)


@pytest.mark.usefixtures('no_matplotlib')
def test_plotting_invpend_without_matplotlib_throws_error():
    """ """

    sys.modules.pop('safe_exploration.environments', None)
    import safe_exploration.environments as env_plot

    assert not env_plot._has_matplotlib

    env = env_plot.InvertedPendulum()
    with pytest.raises(ImportError) as e_info:
        env.plot_safety_bounds()

    with pytest.raises(ImportError) as e_info:
        env.plot_state(None)

    with pytest.raises(ImportError) as e_info:
        env.plot_ellipsoid_trajectory(None,None)


def test_normalization(before_test_inv_pend):
    """ """
    env = before_test_inv_pend
    state = np.random.rand(env.n_s)
    action = np.random.rand(env.n_u)
    s_1, a_1 = env.normalize(*env.unnormalize(state, action))
    s_2, a_2 = env.unnormalize(*env.normalize(state, action))

    assert np.all(s_1 == state)
    assert np.all(a_1 == action)
    assert np.all(s_2 == state)
    assert np.all(a_2 == action)


def test_safety_bounds_normalization(before_test_inv_pend):
    """ """
    env = before_test_inv_pend

    n_samples = 50
    x = np.random.randn(n_samples, env.n_s)
    h_mat_safe, h_safe, _, _ = env.get_safety_constraints(normalize=False)
    in_unnorm = sample_inside_polytope(x, h_mat_safe, h_safe)

    x_norm, _ = env.normalize(x)
    h_mat_safe_norm, h_safe_norm, _, _ = env.get_safety_constraints(normalize=True)
    in_norm = sample_inside_polytope(x_norm, h_mat_safe_norm, h_safe_norm)

    assert np.all(
        in_unnorm == in_norm), "do the normalized constraint correspond to the unnormalized?"


def test_gradients(before_test_inv_pend):
    env = before_test_inv_pend
    n_s = env.n_s
    n_u = env.n_u

    for i in range(n_s):
        f = lambda z: env._dynamics(0, z[:env.n_s], z[env.n_s:])[i]
        f_grad = env._jac_dynamics()[i, :]
        grad_finite_diff = approx_fprime(np.zeros((n_s + n_u,)), f, 1e-8)

        # err = check_grad(f,f_grad,np.zeros((n_s+n_u,)))

        assert np.allclose(f_grad,
                           grad_finite_diff), 'Is the gradient of the {}-th dynamics dimension correct?'.format(
            i)
