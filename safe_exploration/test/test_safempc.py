# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:03:39 2017

@author: tkoller
"""
import os.path

import numpy as np
import pytest
from casadi import reshape as cas_reshape
from casadi import MX
from numpy.testing import assert_allclose

from ..environments import CartPole
from ..safempc_simple import SimpleSafeMPC
from ..state_space_models import CasadiSSMEvaluator

import safe_exploration.ssm_gpy
from safe_exploration.state_space_models import StateSpaceModel
from safe_exploration.ssm_gpy import SimpleGPModel
from safe_exploration.gp_reachability_casadi import lin_ellipsoid_safety_distance
from GPy.kern import RBF

try:
    import safe_exploration.ssm_gpy
    from safe_exploration.ssm_gpy import SimpleGPModel
    from GPy.kern import RBF
    _has_ssm_gpy = True
except:
    _has_ssm_gpy = False

try:
    from safe_exploration.ssm_pytorch import GPyTorchSSM, BatchKernel
    import gpytorch
    import torch

    _has_ssm_gpytorch = True
except:
    _has_ssm_gpytorch = False


a_tol = 1e-5
r_tol = 1e-4

# custom opts. To make tests fast, it's important to have max_iter == 1
OPTS_SOLVER = {'error_on_fail': False,
                   'ipopt': {'hessian_approximation': 'limited-memory', "max_iter": 1,
                             "expect_infeasible_problem": "no", \
                             'acceptable_tol': 1e-4, "acceptable_constr_viol_tol": 1e-5,
                             "bound_frac": 0.5, "start_with_resto": "no",
                             "required_infeasibility_reduction": 0.85,
                             "acceptable_iter": 8}}


def get_gpy_ssm(path,n_s,n_u):

    train_data = dict(list(np.load(path).items()))
    X = train_data["X"]
    X = X[:20, :]
    y = train_data["y"]
    y = y[:20, :]

    kerns = ["rbf"]*n_s
    m = None
    gp = SimpleGPModel(n_s, n_s, n_u, kerns, X, y, m)
    gp.train(X, y, m, opt_hyp=False, choose_data=False)

    return gp


class DummySSM(StateSpaceModel):
    """


    """

    def __init__(self, n_s, n_u):
        super(DummySSM, self).__init__(n_s, n_u)

    def predict(self, states, actions, jacobians=False, full_cov=False):
        """
        """

        if jacobians:
            return np.random.randn(self.num_states, 1), np.ones(
                (self.num_states, 1)), np.zeros(
                (self.num_states, self.num_states + self.num_actions)), np.zeros(
                (self.num_states, self.num_states + self.num_actions))
        return np.random.randn(self.num_states, 1), np.ones((self.num_states, 1))

    def linearize_predict(self, states, actions, jacobians=False, full_cov=True):
        if jacobians:
            return np.random.randn(self.num_states, 1), np.ones(
                (self.num_states, 1)), np.zeros(
                (self.num_states, self.num_states + self.num_actions)), np.zeros(
                (self.num_states, self.num_states + self.num_actions)), np.random.randn(
                self.num_states, self.num_actions + self.num_states,
                self.num_states + self.num_actions)
        return np.random.randn(self.num_states, 1), np.ones((self.num_states, 1))


def get_gpytorch_ssm(path,n_s,n_u):
    kernel = BatchKernel([gpytorch.kernels.RBFKernel()]*n_s)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=n_s)

    train_data = dict(list(np.load(path).items()))
    X = train_data["X"]
    train_x = torch.from_numpy(np.array(X[:20, :],dtype=np.float32))
    y = train_data["y"]
    train_y = torch.from_numpy(np.array(y[:20, :],dtype=np.float32)).t()

    ssm = GPyTorchSSM(n_s, n_u, train_x, train_y, kernel, likelihood)

    return ssm


@pytest.fixture(params=[("CartPole", True,"gpytorch", False, False), ("CartPole", True,"GPy", False, True),
                        ("CartPole", True,"gpytorch", True, False), ("CartPole", True,"GPy", True, True)])#,("CartPole", True,"dummy")])
def before_test_safempc(request):
    np.random.seed(12345)
    env, lin_model, ssm, opt_x0, init_uncertainty = request.param

    if env == "CartPole":
        env = CartPole()
        n_s = env.n_s
        n_u = env.n_u
        path = os.path.join(os.path.dirname(__file__), "data_cartpole.npz")
        c_safety = 0.5
    q_0 = None
    if init_uncertainty:
        q_0 = 0.1*np.eye(n_s)
    if ssm == "GPy":
        if not _has_ssm_gpy:
            pytest.skip("Test requires optional dependencies 'ssm_gpy'")

        ssm = get_gpy_ssm(path, env.n_s, env.n_u)

    elif ssm == "gpytorch":
        if not _has_ssm_gpytorch:
            pytest.skip("Test requires optional dependencies 'ssm_gpytorch'")
        ssm = get_gpytorch_ssm(path, env.n_s, env.n_u)
    elif ssm == "dummy":
        ssm = DummySSM(n_s,n_u)

    else:
        pytest.fail("unknown ssm")

    a = None
    b = None
    lin_model_param = None
    if lin_model:
        a, b = env.linearize_discretize()
        lin_model_param = (a, b)

    n_safe = 3
    n_perf = 1

    l_mu = np.array([0.01] * n_s)
    l_sigma = np.array([0.01] * n_s)

    h_mat_safe = np.hstack((np.eye(n_s, 1), -np.eye(n_s, 1))).T
    h_safe = np.array([3000, 3000]).reshape((2, 1))
    h_mat_obs = np.copy(h_mat_safe)
    h_obs = np.array([3000, 3000]).reshape((2, 1))

    wx_cost = 10 * np.eye(n_s)
    wu_cost = 0.1

    dt = 0.1
    ctrl_bounds = np.hstack((np.reshape(-1000, (-1, 1)), np.reshape(1000, (-1, 1))))

    opt_perf = {'type_perf_traj': 'mean_equivalent', 'n_perf': n_perf, 'r': 1,
                    'perf_has_fb': True}

    safe_policy = None
    env_opts_safempc = dict()
    env_opts_safempc["h_mat_safe"] = h_mat_safe
    env_opts_safempc["h_safe"] = h_safe
    env_opts_safempc["lin_model"] = lin_model
    env_opts_safempc["ctrl_bounds"] = ctrl_bounds
    env_opts_safempc["safe_policy"] = safe_policy
    env_opts_safempc["h_mat_obs"] = h_mat_obs
    env_opts_safempc["h_obs"] = h_obs
    env_opts_safempc["dt"] = dt
    env_opts_safempc["lin_trafo_gp_input"] = None
    env_opts_safempc["l_mu"] = l_mu
    env_opts_safempc["l_sigma"] = l_sigma
    env_opts_safempc["lin_model"] = lin_model_param

    safe_mpc = SimpleSafeMPC(n_safe, ssm, env_opts_safempc, wx_cost, wu_cost,
                             beta_safety = c_safety, opt_perf_trajectory = opt_perf, opts_solver=OPTS_SOLVER)

    if n_perf > 1:
        cost = lambda p_0, u_0, p_all, q_all, k_ff_safe, k_fb_safe, \
                                        sigma_safe, mu_perf, sigma_perf, \
                                        gp_pred_sigma_perf, k_fb_perf, k_ff_perf: MX(1)
    else:
        cost = lambda p_0, u_0, p_all, q_all, k_ff_safe, k_fb_safe, \
                                        sigma_safe: MX(1)

    safe_mpc.init_solver(cost, opt_x0=opt_x0, init_uncertainty=init_uncertainty)

    x_0 = 0.2 * np.random.randn(n_s, 1)

    k_fb_0 = safe_mpc.get_lqr_feedback()
    _, _, _, _, k_fb_apply, k_ff_apply, p_all, q_all, sol = safe_mpc.solve(x_0,
                                                                        sol_verbose=True, q_0=q_0, k_fb_0=k_fb_0)

    return env, safe_mpc, None, None, k_fb_apply, k_ff_apply, p_all, q_all, sol, q_0, k_fb_0


#def test_run_safempc(before_test_safempc):
#    """ """
#    env, safe_mpc, k_ff_perf_traj, k_fb_perf_traj, k_fb_apply, k_ff_apply, p_all, q_all, sol, _, _ = before_test_safempc
#    safe_mpc.solve(np.random.randn(env.n_s,1))


def test_mpc_casadi_same_constraint_values_as_numeric_eval(before_test_safempc):
    """check if the returned open loop (numerical) ellipsoids are the same as in internal planning"""

    env, safe_mpc, k_ff_perf_traj, k_fb_perf_traj, k_fb_apply, k_ff_apply, p_all, q_all, sol, q_0, k_fb_0 = before_test_safempc

    n_s = env.n_s
    n_u = env.n_u

    g_0 = lin_ellipsoid_safety_distance(p_all[0, :].reshape(n_s, 1),
                                        q_all[0, :].reshape(n_s, n_s),
                                        safe_mpc.h_mat_obs, safe_mpc.h_obs)
    g_1 = lin_ellipsoid_safety_distance(p_all[1, :].reshape(n_s, 1),
                                        cas_reshape(q_all[1, :], (n_s, n_s)),
                                        safe_mpc.h_mat_obs, safe_mpc.h_obs)
    g_safe = lin_ellipsoid_safety_distance(p_all[-1, :].reshape(n_s, 1),
                                           q_all[-1, :].reshape(n_s, n_s),
                                           safe_mpc.h_mat_safe, safe_mpc.h_safe)

    idx_state_constraints = env.n_u * safe_mpc.n_safe * 2 - 1
    if q_0 is not None:
        idx_state_constraints += 1 ## not sure why this is

    constr_values = sol["g"]

    assert_allclose(g_0, constr_values[
                         idx_state_constraints:idx_state_constraints + safe_mpc.m_obs],
                    r_tol, a_tol)
    # Are the distances to the obstacle the same after two steps?
    assert_allclose(g_1, constr_values[
                         idx_state_constraints + safe_mpc.m_obs:idx_state_constraints + 2 * safe_mpc.m_obs],
                    r_tol, a_tol)
    # Are the distances to the safe set the same after the last step?
    assert_allclose(g_safe, constr_values[
                            idx_state_constraints + 2 * safe_mpc.m_obs:idx_state_constraints + 2 * safe_mpc.m_obs + safe_mpc.m_safe],
                    r_tol, a_tol)

@pytest.mark.xfail(reason="There is a bug in custom cost right now, need to write test before debugging.")
def test_safempc_custom_cost(before_test_safempc):
    raise NotImplementedError("Need to test this")