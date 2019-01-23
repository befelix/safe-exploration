# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:39 2017

@author: tkoller
"""
import numpy as np
import pytest
import re

from ..state_space_models import StateSpaceModel, CasadiSSMEvaluator
import casadi as cas
from casadi.tools import capture_stdout

def pytest_namespace():
    return {"ipopt_output": None}

@pytest.fixture(params=[(2,3,False),(2,3,True)])
def before_test_ssm(request):
    n_s,n_u,linearize_mean = request.param
    dummy_ssm = DummySSM(n_s,n_u)

    return dummy_ssm, n_s ,n_u ,linearize_mean


def test_create_casadissmevaluator_no_error_thrown(before_test_ssm):
    dummy_ssm, n_s ,n_u ,linearize_mean = before_test_ssm

    try:
        casadi_ssm = CasadiSSMEvaluator(dummy_ssm,linearize_mean)
    except MyError:
        pytest.fail("Creation of CasadiSSMEvaluator fails for linearize_mean = ".format(linearize_mean))

def test_jacobians_no_error_thrown(before_test_ssm):
    """ """
    dummy_ssm, n_s ,n_u ,linearize_mean = before_test_ssm

    casadi_ssm = CasadiSSMEvaluator(dummy_ssm,linearize_mean)

    x_in_dummy = np.random.randn(n_s,1)
    u_in_dummy = np.random.randn(n_u,1)

    n_in = casadi_ssm.get_n_in()
    n_out = casadi_ssm.get_n_out()
    for i in range(n_in):
        for j in range(n_out):
            f_jac = casadi_ssm.jacobian_old(i,j)
            f_jac(x_in_dummy,u_in_dummy)

@pytest.mark.skip(reason = "Not sure how to implement yet")
def test_jacobian_ssm_evaluator_same_as_ssm(before_test_ssm):
    """
    Test if calling jacobian() on the ssm_casadi function results
    in the same derivative information as calling predict(..,jacobian = True,..) or linearize_predict(..,jacobian = True,..)

    """
    dummy_ssm, n_s ,n_u ,linearize_mean = before_test_ssm

    casadi_ssm = CasadiSSMEvaluator(dummy_ssm,linearize_mean)

    raise NotImplementedError("Still need to implement this")

@pytest.mark.dependency()
def test_integration_casadi_SSM_casadi_no_error_thrown(before_test_ssm):

    dummy_ssm, n_s ,n_u ,linearize_mean = before_test_ssm

    casadi_ssm = CasadiSSMEvaluator(dummy_ssm,linearize_mean)

    x = cas.MX.sym("x",(n_s,1))
    y = cas.MX.sym("y",(n_u,1))

    if linearize_mean:
        mu,sigma,mu_jac = casadi_ssm(x,y)
        f = cas.sum1(cas.sum2(mu)) + cas.sum1(cas.sum2(sigma)) + cas.sum1(cas.sum2(mu_jac))
    else:
        mu, sigma = casadi_ssm(x,y)
        f = cas.sum1(cas.sum2(mu)) + cas.sum1(cas.sum2(sigma))

    x = cas.vertcat(x,y)

    options = {"ipopt": {"hessian_approximation": "limited-memory","max_iter":2, "derivative_test": "first-order"}}
    solver = cas.nlpsol("solver","ipopt",{"x":x,"f":f},options)

    with capture_stdout() as out:
        res = solver(x0=np.random.randn(5,1))

    pytest.ipopt_output = out[0]

@pytest.mark.xfail(reason = "Need to fix the DummySSM to have correct and meaningful (i.e. varying values) derivatives ")
@pytest.mark.dependency(depends=['test_integration_casadi_SSM_casadi_no_error_thrown'])
def test_ssm_evaluator_derivatives_passed_correctly():
    """ Check if jacobians are passed correctly to casadi with SSMEvaluator

    This is NOT a derivative checker for the SSM. This needs to be done in a different test
    specific to the SSM. We only check here, if we correctly pass the jacobians to casadi.

    """
    n_errors = _parse_derivative_checker_output(pytest.ipopt_output)

    assert n_errors == 0, "Did we get errors in the derivative check?"

def _parse_derivative_checker_output(out):
    """ Check the output of the derivative checker

    This check is very sensitive to changes in the ipopt version (and hence possible changes in the output of the derivative checkrer).
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

    exp = r'Derivative checker detected ([0-9]+)'# error(s)'

    m = re.search(exp, out)
    if m:
        n_fails = m.group(1)
    else:
        pytest.fail("what is the output if there is no fail?. Or did the interface change?")

    return int(n_fails)

class DummySSM(StateSpaceModel):
    """


    """
    def __init__(self,n_s,n_u):
        super(DummySSM,self).__init__(n_s,n_u)

    def predict(self, states, actions, jacobians=False, full_cov=False):
        """
        """

        if jacobians:
            return np.random.randn(self.num_states,1), np.zeros((self.num_states,1)), np.zeros((self.num_states,self.num_states+self.num_actions)), np.zeros((self.num_states,self.num_states+self.num_actions))
        return np.random.randn(self.num_states,1), np.zeros((self.num_states,1))

    def linearize_predict(self,states,actions,jacobians = False, full_cov = True):
        if jacobians:
            return np.random.randn(self.num_states,1), np.zeros((self.num_states,1)), np.zeros((self.num_states,self.num_states+self.num_actions)), np.zeros((self.num_states,self.num_states+self.num_actions)), np.random.randn(self.num_states,self.num_actions+self.num_states,self.num_states+self.num_actions)
        return np.random.randn(self.num_states,1), np.zeros((self.num_states,1))