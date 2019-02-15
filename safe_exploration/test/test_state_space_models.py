# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:39 2017

@author: tkoller
"""
import re
import casadi as cas
import numpy as np
import pytest
from casadi.tools import capture_stdout

from ..state_space_models import StateSpaceModel, CasadiSSMEvaluator

try:
    from safe_exploration.ssm_pytorch import GPyTorchSSM, BatchKernel
    import gpytorch
    import torch
except:
    pass

def pytest_namespace():
    return {"ipopt_output": []}

@pytest.fixture(params=[(2, 3, False),
                    (2, 3, True)])
def gpy_torch_ssm_init(request,check_has_ssm_pytorch):
    n_s, n_u, linearize_mean = request.param

    n_data = 10
    kernel = [gpytorch.kernels.RBFKernel()]*n_s

    likelihood = [gpytorch.likelihoods.GaussianLikelihood()]*n_s
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
        print(type(ssm))
        for i in range(n_in):
            for j in range(n_out):
                f_jac = casadi_ssm.jacobian_old(i, j)
                f_jac(x_in_dummy, u_in_dummy)



    def test_integration_dummy_ssm_casadissm_evaluator_casadi_no_error_thrown(self,dummy_ssm_init):
        ssm, n_s, n_u, linearize_mean = dummy_ssm_init

        #self.ipopt_output += [tuple(run_ipopt_ssmevaluator(ssm,n_s,n_u,linearize_mean))]
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
        #print(self.ipopt_output)

        ipopt_output = self.ipopt_output
        for i in range(len(ipopt_output)):
            print(len(self.ipopt_output[i]))
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
    res = solver(x0=np.random.randn(5, 1))

    return str(type(ssm)),linearize_mean,out[0]





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
