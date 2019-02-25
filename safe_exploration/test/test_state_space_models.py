# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:39 2017

@author: tkoller
"""
import re
import casadi as cas
import numpy as np
import pytest
import os.path
from casadi.tools import capture_stdout
from casadi import MX,vertcat,sum1,sum2

from ..gp_reachability_casadi import lin_ellipsoid_safety_distance
from ..state_space_models import StateSpaceModel, CasadiSSMEvaluator
from .. import gp_reachability_casadi as reach_cas
from .. import uncertainty_propagation_casadi as prop_casadi

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


def get_gpy_ssm(path,n_s,n_u):

    train_data = dict(list(np.load(path).items()))
    X = train_data["X"]
    X = X[:80, :]
    y = train_data["y"]
    y = y[:80, :]

    kerns = ["rbf"]*n_s
    m = None
    gp = SimpleGPModel(n_s, n_s, n_u, kerns, X, y, m)
    gp.train(X, y, m, opt_hyp=False, choose_data=False)

    return gp


def get_gpytorch_ssm(path,n_s,n_u):

    kernel = BatchKernel([gpytorch.kernels.RBFKernel()]*n_s)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=n_s)

    train_data = dict(list(np.load(path).items()))
    X = np.array(train_data["X"],dtype=np.float32)
    train_x = torch.from_numpy(X[:80, :])
    y = np.array(train_data["y"],dtype=np.float32)
    train_y = torch.from_numpy(y[:80, :])
    print(train_x.dtype)
    print(train_y.dtype)

    ssm = GPyTorchSSM(n_s,n_u,train_x,train_y,kernel,likelihood)

    return ssm


@pytest.fixture(params=[("CartPole", True,"gpytorch",True),("CartPole", True,"GPy",True)])
def before_test_casadissm(request):
    np.random.seed(12345)
    env, lin_model, ssm, init_uncertainty = request.param

    if env == "CartPole":
        n_s = 4
        n_u = 1
        path = os.path.join(os.path.dirname(__file__), "data_cartpole.npz")
        c_safety = 0.5

    if ssm == "GPy":
        if not _has_ssm_gpy:
            pytest.skip("Test requires optional dependencies 'ssm_gp'")

        ssm = get_gpy_ssm(path,n_s,n_u)

    elif ssm == "gpytorch":
        pytest.xfail(reason="Requires multi-input multi-output GP fix!")
        if not _has_ssm_gpytorch:
            pytest.skip("Test requires optional dependencies 'ssm_gp'")
        ssm = get_gpytorch_ssm(path,n_s,n_u)
    else:
        pytest.fail("unknown ssm")

    a = None
    b = None
    lin_model_param = None
    if lin_model:
        #a, b = env.linearize_discretize()
        a = np.eye(n_s)
        b = np.zeros((n_s,1))
        lin_model_param = (a, b)

    n_safe = 1
    n_perf = 2

    L_mu = np.array([0.001] * n_s)
    L_sigm = np.array([0.001] * n_s)
    k_fb = np.random.rand(n_u, n_s)  # need to choose this appropriately later
    k_ff = np.random.rand(n_u, 1)

    p = .1 * np.random.randn(n_s, 1)
    if init_uncertainty:
        q = .2 * np.array(
            [[.5, .2], [.2, .65]])  # reachability based on previous uncertainty
    else:
        q = None  # no initial uncertainty

    return p, q, ssm, k_fb, k_ff, L_mu, L_sigm, c_safety, a, b


def pytest_namespace():
    return {"ipopt_output": []}


@pytest.fixture(params=[(2, 3, False),
                    (2, 3, True)])
def gpy_torch_ssm_init(request,check_has_ssm_pytorch):

    pytest.xfail(reason= "Requires multi-input multi-output GP fix!")
    n_s, n_u, linearize_mean = request.param

    n_data = 10
    kernel = BatchKernel([gpytorch.kernels.RBFKernel()]*n_s)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=n_s)
    train_x = torch.randn((n_data,n_s+n_u))
    train_y = torch.randn((n_data,n_s))
    ssm = GPyTorchSSM(n_s,n_u,train_x,train_y,kernel,likelihood)


    return ssm, n_s, n_u, linearize_mean

@pytest.fixture(params = [ (1, 3, True),(1, 3, False)])
def dummy_ssm_init(request):
    n_s,n_u,linearize_mean = request.param
    ssm = DummySSM(n_s,n_u)

    return ssm, n_s,n_u, linearize_mean

def test_create_casadissmevaluator_no_error_thrown(dummy_ssm_init):
    dummy_ssm, n_s, n_u, linearize_mean = dummy_ssm_init

    try:
        casadi_ssm = CasadiSSMEvaluator(dummy_ssm, linearize_mean)
    except MyError:
        pytest.fail("Creation of CasadiSSMEvaluator fails for linearize_mean = ".format(
            linearize_mean))


def test_jacobians_no_error_thrown(dummy_ssm_init):
    """ """
    dummy_ssm, n_s, n_u, linearize_mean = dummy_ssm_init

    casadi_ssm = CasadiSSMEvaluator(dummy_ssm, linearize_mean)

    x_in_dummy = np.random.randn(n_s, 1)
    u_in_dummy = np.random.randn(n_u, 1)

    n_in = casadi_ssm.get_n_in()
    n_out = casadi_ssm.get_n_out()
    for i in range(n_in):
        for j in range(n_out):
            f_jac = casadi_ssm.jacobian_old(i, j)
            f_jac(x_in_dummy, u_in_dummy)


class TestDerivativesCasadiSSMEvaluator(object):

    ipopt_output = []

    def test_jacobians_no_error_thrown_dummy_ssm(self,dummy_ssm_init):
        """ """
        dummy_ssm, n_s, n_u, linearize_mean = dummy_ssm_init

        casadi_ssm = CasadiSSMEvaluator(dummy_ssm, linearize_mean)

        self.compute_jacobians(dummy_ssm,casadi_ssm,n_s,n_u)

    #@pytest.mark.skip(reason = "Still need to fully implement the GPytorchSSM to fit the CasadiSSMEvaluator")
    def test_jacobians_no_error_thrown_gpytorch_ssm(self,gpy_torch_ssm_init):
        """ """
        ssm, n_s, n_u, linearize_mean = gpy_torch_ssm_init

        casadi_ssm = CasadiSSMEvaluator(ssm, linearize_mean)

        self.compute_jacobians(ssm,casadi_ssm,n_s,n_u)

    def compute_jacobians(self, ssm, casadi_ssm ,n_s ,n_u):

        x_in_dummy = np.random.randn(n_s, 1)
        u_in_dummy = np.random.randn(n_u, 1)

        n_in = casadi_ssm.get_n_in()
        n_out = casadi_ssm.get_n_out()

        for i in range(n_in):
            for j in range(n_out):
                f_jac = casadi_ssm.jacobian_old(i, j)
                f_jac(x_in_dummy, u_in_dummy)

    #@pytest.mark.skip(reason = "Still need to fully implement the GPytorchSSM to fit the CasadiSSMEvaluator")
    def test_integration_gpytorch_ssm_casadissm_evaluator_casadi_no_error_thrown(self,gpy_torch_ssm_init):
        ssm, n_s, n_u, linearize_mean = gpy_torch_ssm_init
        self.ipopt_output += [tuple(run_ipopt_ssmevaluator(ssm,n_s,n_u,linearize_mean))]

    @pytest.mark.dependency(depends=['test_integration_dummy_ssm_casadissm_evaluator_casadi_no_error_thrown',
                              'test_integration_gpytorch_ssm_casadissm_evaluator_casadi_no_error_thrown'])
    def test_ssm_evaluator_derivatives_passed_correctly(self):
        """ Check if jacobians are passed correctly to casadi with SSMEvaluator

        This is NOT a derivative checker for the SSM. This needs to be done in a different test
        specific to the SSM. We only check here, if we correctly pass the jacobians to casadi.

        """

        ipopt_output = self.ipopt_output
        for i in range(len(ipopt_output)):
            model_name, lin_mean, out = ipopt_output[i]
            n_errors = _parse_derivative_checker_output(out)

            assert n_errors == 0, ("Did the derivative checker fail for"
                                   f" model {model_name} with"
                                   f" linearize_mean = {lin_mean}?")

@pytest.mark.skip(reason="Not sure how to implement yet and not sure if we need this")
def test_jacobian_ssm_evaluator_same_as_ssm(dummy_ssm_init):
    """
    Test if calling jacobian() on the ssm_casadi function results
    in the same derivative information as calling predict(..,jacobian = True,..) or linearize_predict(..,jacobian = True,..)

    """
    dummy_ssm, n_s, n_u, linearize_mean = dummy_ssm_init

    casadi_ssm = CasadiSSMEvaluator(dummy_ssm, linearize_mean)

    raise NotImplementedError("Still need to implement this")


def run_ipopt_ssmevaluator(ssm,n_s,n_u,linearize_mean):
    casadi_ssm = CasadiSSMEvaluator(ssm, linearize_mean)

    x = cas.MX.sym("x", (n_s, 1))
    y = cas.MX.sym("y", (n_u, 1))

    if linearize_mean:
        mu, sigma, mu_jac = casadi_ssm(x, y)
        f = cas.sum1(cas.sum2(mu)) + cas.sum1(cas.sum2(sigma)) + cas.sum1(
            cas.sum2(mu_jac))
    else:
        mu, sigma = casadi_ssm(x, y)
        f = cas.sum1(cas.sum2(mu)) + cas.sum1(cas.sum2(sigma))

    x = cas.vertcat(x, y)

    options = {"ipopt": {"hessian_approximation": "limited-memory", "max_iter": 2,
                         "derivative_test": "first-order"}}
    solver = cas.nlpsol("solver", "ipopt", {"x": x, "f": f}, options)

    with capture_stdout() as out:
        res = solver(x0=np.random.randn(5, 1))

    return str(type(ssm)),linearize_mean,out[0]




#def test_

def test_ipopt_ssmevaluator_multistep_ahead(before_test_casadissm):

    p_0, q_0, ssm, k_fb, k_ff, L_mu, L_sigm, c_safety, a, b = before_test_casadissm
    T = 3

    n_u, n_s = np.shape(k_fb)

    u_0 = .2 * np.random.randn(n_u, 1)
    k_fb_0 = np.random.randn(T - 1,
                             n_s * n_u)  # np.zeros((T-1,n_s*n_u))# np.random.randn(T-1,n_s*n_u)
    k_ff = np.random.randn(T - 1, n_u)
    # k_fb_ctrl = np.zeros((n_u,n_s))#np.random.randn(n_u,n_s)

    u_0_cas = MX.sym("u_0", (n_u, 1))
    k_fb_cas_0 = MX.sym("k_fb", (T - 1, n_u * n_s))
    k_ff_cas = MX.sym("k_ff", (T - 1, n_u))

    ssm_forward = ssm.get_forward_model_casadi(True)

    p_new_cas, q_new_cas, pred_sigm_all = reach_cas.multi_step_reachability(p_0, u_0, k_fb_cas_0,
                                                                k_ff_cas, ssm_forward, L_mu,
                                                                L_sigm, c_safety, a, b)

    h_mat_safe = np.hstack((np.eye(n_s, 1), -np.eye(n_s, 1))).T
    h_safe = np.array([300, 300]).reshape((2, 1))
    h_mat_obs = np.copy(h_mat_safe)
    h_obs = np.array([300, 300]).reshape((2, 1))

    g = []
    lbg = []
    ubg = []
    for i in range(T):
        p_i = p_new_cas[i, :].T
        q_i = q_new_cas[i, :].reshape((n_s, n_s))
        g_state = lin_ellipsoid_safety_distance(p_i, q_i, h_mat_obs,
                                                h_obs,
                                                c_safety=2.0)
        g = vertcat(g, g_state)
        lbg += [-cas.inf] * 2
        ubg += [0] * 2



    x_safe = vertcat(u_0_cas,k_ff_cas.reshape((-1,1)))
    params_safe = vertcat(k_fb_cas_0.reshape((-1,1)))
    f_safe = sum1(sum2(p_new_cas)) + sum1(sum2(q_new_cas)) + sum1(sum2(pred_sigm_all))

    k_ff_cas_all = MX.sym("k_ff_single", (T, n_u))

    k_fb_cas_all = MX.sym("k_fb_all", (T - 1, n_s * n_u))
    k_fb_cas_all_inp = [k_fb_cas_all[i, :].reshape((n_u, n_s)) for i in range(T - 1)]


    ssm_forward1 = ssm.get_forward_model_casadi(True)
    mu_multistep, sigma_multistep, sigma_pred_perf = prop_casadi.multi_step_taylor_symbolic(p_0,
                                                                              ssm_forward1,
                                                                              k_ff_cas_all,
                                                                              k_fb_cas_all_inp,
                                                                              a=a, b=b)
    x_perf = vertcat(k_ff_cas_all.reshape((-1,1)))
    params_perf = vertcat(k_fb_cas_all.reshape((-1,1)))
    f_perf = sum1(sum2(mu_multistep)) + sum1(sum2(sigma_multistep)) + sum1(sum2(sigma_pred_perf))

    f_both = f_perf + f_safe
    x_both =  vertcat(x_safe,x_perf)
    params_both = vertcat(params_safe,params_perf)


    options = {"ipopt": {"hessian_approximation": "limited-memory", "max_iter": 1},'error_on_fail': False}



    #raise NotImplementedError("""Need to parse output to get fail/pass signal!
    #                         Either 'Maximun Number of Iterations..' or 'Optimal solution found' are result of
    #                         a successful run""")
    #safe only
    n_x = np.shape(x_safe)[0]
    n_p = np.shape(params_safe)[0]
    solver = cas.nlpsol("solver", "ipopt", {"x": x_safe, "f": f_safe, "p":params_safe,"g":g}, options)
    with capture_stdout() as out:
        solver(x0=np.random.randn(n_x,1),p = np.random.randn(n_p,1),lbg=lbg,ubg=ubg)

    opt_sol_found = solver.stats()
    if not opt_sol_found:
        max_numb_exceeded = parse_solver_output_pass(out)
        if not max_numb_exceeded:
            pytest.fail("Neither optimal solution found, nor maximum number of iterations exceeded. Sth. is wrong")


    n_x = np.shape(x_perf)[0]
    n_p = np.shape(params_perf)[0]
    solver = cas.nlpsol("solver", "ipopt", {"x": x_perf, "f": f_perf, "p":params_perf}, options)
    with capture_stdout() as out:
        solver(x0=np.random.randn(n_x,1),p = np.random.randn(n_p,1))
    opt_sol_found = solver.stats()
    if not opt_sol_found:
        max_numb_exceeded = parse_solver_output_pass(out)
        if not max_numb_exceeded:
            pytest.fail("Neither optimal solution found, nor maximum number of iterations exceeded. Sth. is wrong")

    #both
    n_x = np.shape(x_both)[0]
    n_p = np.shape(params_both)[0]
    solver = cas.nlpsol("solver", "ipopt", {"x": x_both, "f": f_both, "p":params_both}, options)
    with capture_stdout() as out:
        solver(x0=np.random.randn(n_x,1),p = np.random.randn(n_p,1))
    opt_sol_found = solver.stats()
    if not opt_sol_found:
        max_numb_exceeded = parse_solver_output_pass(out)
        if not max_numb_exceeded:
            pytest.fail("Neither optimal solution found, nor maximum number of iterations exceeded. Sth. is wrong")


def parse_solver_output_pass(out):
    """ Check if the solver exited without crash

    We define a run to be successful if either of the following messages appear:
        * "Maximum number of iterations exceeded"

    """

    # Check for max number of iterations message
    exp_max_number = r"Maximum Number of Iterations Exceeded"
    m_max_number = re.earch(exp_max_number,out)

    m_max_numb_found = False
    if m_max_number:
        m_max_numb_found = True

    return m_max_numb_found


def _parse_derivative_checker_output(out):
    """ Check the output of the derivative checker

    This check is very sensitive to changes in the ipopt version (and hence possible changes in the output of the derivative checker).
    However, this is probably the only way to make sure that ipopt gets the right derivatives

    Parameters
    ----------
    out: String
        The caught output of casadi that includes the derivative checker output

    Returns
    -------
    n_errors: int
        The number of errors thrown by the derivative checker
    """

    exp_n_err = r'Derivative checker detected ([0-9]+)'  # error(s)'
    exp_no_err_detected = r'No errors detected by derivative checker'
    m_n_err = re.search(exp_n_err, out)
    m_no_err_detected = re.search(exp_n_err, out)

    n_err_found = False
    if m_n_err:
        n_fails = m_n_err.group(1)
    else:
        n_fails = 0
        n_err_found = True

    no_err_detected_found = False
    if m_no_err_detected:
        no_err_detected_found = True

    if not n_err_found and not no_err_detected_found:
        pytest.fail("""Neither the number of errors nor the message 'No errors detected..'
         was found in output. Test seems to be broken""")


    return int(n_fails)


class DummySSM(StateSpaceModel):
    """


    """

    def __init__(self, n_s, n_u):
        super(DummySSM, self).__init__(n_s, n_u)

    def predict(self, states, actions, jacobians=False, full_cov=False):
        """
        """

        if jacobians:
            return np.random.randn(self.num_states, 1), np.zeros(
                (self.num_states, 1)), np.zeros(
                (self.num_states, self.num_states + self.num_actions)), np.zeros(
                (self.num_states, self.num_states + self.num_actions))
        return np.random.randn(self.num_states, 1), np.zeros((self.num_states, 1))

    def linearize_predict(self, states, actions, jacobians=False, full_cov=True):
        if jacobians:
            return np.random.randn(self.num_states, 1), np.zeros(
                (self.num_states, 1)), np.zeros(
                (self.num_states, self.num_states + self.num_actions)), np.zeros(
                (self.num_states, self.num_states + self.num_actions)), np.random.randn(
                self.num_states, self.num_actions + self.num_states,
                self.num_states + self.num_actions)
        return np.random.randn(self.num_states, 1), np.zeros((self.num_states, 1))
