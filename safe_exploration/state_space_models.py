# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
"""
import warnings

import numpy as np
import casadi as cas


from .utils import reshape_derivatives_3d_to_2d


class StateSpaceModel(object):
    """A state space model including uncertainty information.

    x_{t+1} = f(x_t, u_t)

    with x in (1 x n) and u in (1 x m).

    Attributes
    ----------
    num_states : int
        The state dimension (n).
    num_actions : int
        The input dimension (m).
    """

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

    def predict(self, states, actions, jacobians=False, full_cov=False):
        """Predict the next states and uncertainty.

        Parameters
        ----------
        states : np.ndarray
            A (N x n) array of states.
        actions : np.ndarray
            A (N x m) array of actions.
        jacobians : bool, optional
            If true, return two additional outputs corresponding to the jacobians.
        full_cov : bool, optional
            Whether to return the full covariance.

        Returns
        -------
        mean : np.ndarray
            A (N x n) mean prediction for the next states.
        variance : np.ndarray
            A (N x n) variance prediction for the next states. If full_cov is True,
            then instead returns the (n x N x N) covariance matrix for each independent
            output of the GP model.
        jacobian_mean : np.ndarray
            A (N x n x n + m) array with the jacobians for each datapoint on the axes.
        jacobian_variance : np.ndarray
            Only supported without the full_cov flag.
        """
        raise NotImplementedError("Need to implement this in a subclass!")

        if jacobians and full_cov:
            raise NotImplementedError('Jacobians of full covariance not supported.')

    def linearize_predict(self, states, actions, jacobians=False, full_cov=False):
        """Predict the next states and uncertainty.

        Parameters
        ----------
        states : np.ndarray
            A (N x n) array of states.
        actions : np.ndarray
            A (N x m) array of actions.
        jacobians : bool, optional
            If true, return two additional outputs corresponding to the jacobians of the predictive
            mean, the linearized predictive mean and variance.
        full_cov : bool, optional
            Whether to return the full covariance.

        Returns
        -------
        mean : np.ndarray
            A (N x n) mean prediction for the next states.
        variance : np.ndarray
            A (N x n) variance prediction for the next states. If full_cov is True,
            then instead returns the (n x N x N) covariance matrix for each independent
            output of the GP model.
        jacobian_mean : np.ndarray
            A (N x n x (n + m) array with the jacobians for each datapoint on the axes.
        hessian_mean: np.ndarray
            A (N x n*(n+m) x (n+m)) Array with the derivatives of each entry in the jacobian for each input
        jacobian_variance : np.ndarray
            Only supported without the full_cov flag.
        """
        raise NotImplementedError(""" Need to implement this in a subclass when using the predefined
                                    get_forward_model_casadi() method that requires this method for the
                                    CasadiSSMEvaluator! Otherwise it is not strictly necessary. """)

        if jacobians and full_cov:
            raise NotImplementedError('Jacobians of full covariance not supported.')

    def get_forward_model_casadi(self, linearize_mu=True):
        """ Returns a forward model that can be used in Casadi

        Generate functions representing predictive mean and variance as well as
        the jacobian of the predictive mean (if linearize_mu is True) that admit the
        casadi function structure (either native casadi.Function types or implemented through
        the casadi.Callback interface)

        Parameters
        ----------
        linearize_mu: Bool
            If true, output the jacobian of the predictive mean addtionally.

        Returns
        -------
        pred_mean: Casadi function
            Function with vector-valued inputs x (px1 array), u (qx1 array)
            that returns the predictive mean of the SSM evaluated at (x,u).
        pred_var: Casadi function
            Function with vector-valued inputs x (px1 array), u (qx1 array)
            that returns the predictive variance of the SSM evaluated at (x,u).
        jac_pred_mu: Casadi function
            Function with vector-valued inputs x (px1 array), u (qx1 array)
            that returns the jacobian of te predictive mean of the SSM evaluated at (x,u).

        """
        return CasadiSSMEvaluator(self, True)


class CasadiSSMEvaluator(cas.Callback):
    """ Base class for a casadi evalauator of a state space model

    Attributes
    ----------
    ssm: StateSpaceModel
            The underlying state space model

    """

    def __init__(self, ssm, linearize_mu=True, differentiation_mode="jacobian",
                 opts={}):
        """

          Parameters
          ----------
          ssm: StateSpaceModel
            The underlying state space model

          linearize_mu: Bool, optional
            True, if we want to linearize the predictive mean

        """
        cas.Callback.__init__(self)
        if not differentiation_mode is "jacobian":
            raise NotImplementedError(
                "For now we only allow the 'jacobian' differentation mode, may implement reverse/forward in future ")
        self.ssm = ssm
        self.linearize_mu = linearize_mu
        self.construct("CasadiModelEvaluator", opts)

        warnings.warn("Need to test this!")

    def get_n_in(self):
        """ """
        return 2  # state and action

    def get_n_out(self):
        """ """
        if self.linearize_mu:
            return 3  # mean, variance and mean_jacobian
        else:
            return 2  # mean and variance

    def get_sparsity_in(self, i):
        """ Input dimensionality """
        return [cas.Sparsity.dense(self.ssm.num_states, 1),
                cas.Sparsity.dense(self.ssm.num_actions, 1)][i]

    def get_sparsity_out(self, i):
        """ Output dimensionality"""
        if self.linearize_mu:
            # mean, variance and mean_jacobian
            return [cas.Sparsity.dense(self.ssm.num_states, 1),
                    cas.Sparsity.dense(self.ssm.num_states, 1),
                    cas.Sparsity.dense(self.ssm.num_states,
                                       self.ssm.num_states + self.ssm.num_actions)][i]
        else:
            return [cas.Sparsity.dense(self.ssm.num_states, 1),
                    cas.Sparsity.dense(self.ssm.num_states, 1)][i]

    def eval(self, arg):
        """ Evaluate the statistical model

        Parameters
        ----------
        arg: list
            List of length n_in containing the inputs to the function (state,action)

        Returns
        -------
        mu: np.ndarray
            A (num_states x 1 ) vector containing the mean prediction of the ssm for each output dimensions
        sigma: np.ndarray
            A (num_states x 1 ) vector containing the variance prediction of the ssm for each output dimensions

        """
        state = arg[0]
        action = arg[1]

        if self.linearize_mu:
            mu, sigma, jac_mu, _ = self.ssm.predict(state.T, action.T, True, False)
            return [mu, sigma, jac_mu]
        else:
            mu, sigma = self.ssm.predict(state.T, action.T)
            return [mu, sigma]

    def get_jacobian(self, name, inames, onames, opts):
        """ Return the Callback function for the jacobians

        Parameters
        ----------
        name:
        inames:
        onames:

        Returns
        -------
        jac_callback: Casadi.Callback
            A Callback-type function returning the gradient function for
            the predictive mean and variance
        """

        class JacFun(cas.Callback):
            """ Nested class representing the Jacobian

            Parameters
            ----------
            ssm: StateSpaceModel
                The underlying state space model.
            linearize_mu: Bool
                If True, we linearize the predictive mean of the SSM. Hence, we need to
                provide the derivatives of the predictive jacobian.

            """

            def __init__(self, ssm, linearize_mu, opts={}):
                self.ssm = ssm
                self.linearize_mu = linearize_mu

                cas.Callback.__init__(self)
                self.construct(name, opts)

                warnings.warn("Need to test this!")

            def get_n_in(self):
                """ """
                if self.linearize_mu:
                    return 5
                else:
                    return 4

            def get_n_out(self):
                """ """
                return 1

            def get_sparsity_in(self, i):
                """ """
                return [cas.Sparsity.dense(self.ssm.num_states, 1),
                        cas.Sparsity.dense(self.ssm.num_actions, 1),
                        cas.Sparsity.dense(self.ssm.num_states, 1),
                        cas.Sparsity.dense(self.ssm.num_states, 1),
                        cas.Sparsity.dense(self.ssm.num_states,
                                           self.ssm.num_states + self.ssm.num_actions)][i]

            def get_sparsity_out(self, i):
                """ """
                if self.linearize_mu:
                    return cas.Sparsity.dense(2 * self.ssm.num_states + (
                            self.ssm.num_states * (
                            self.ssm.num_states + self.ssm.num_actions)),
                                              self.ssm.num_states + self.ssm.num_actions)
                else:
                    return cas.Sparsity.dense(2 * self.ssm.num_states,
                                              self.ssm.num_states + self.ssm.num_actions)

            def eval(self, arg):
                """ Evaluate the Jacobian of the ssm predictive mean/variance

                Parameters
                ----------
                arg: list
                    List of length n_in containing the inputs to the function (state,action)

                Returns
                -------
                jac_pred: np.ndarray
                    A (2*n x (n+m)) array containing the stacked jacobians of the predictive mean
                    and predictive variance of the ssm
                """
                state = arg[0]
                action = arg[1]

                if self.linearize_mu:
                    mu, sigma, jac_mu, jac_sigma, gradients_jac_mu = self.ssm.linearize_predict(
                        state.T, action.T, True, False)

                    gradient_jac_mu_compressed = reshape_derivatives_3d_to_2d(
                        gradients_jac_mu)
                    jac_pred = np.vstack(
                        (jac_mu, jac_sigma, gradient_jac_mu_compressed))
                else:
                    mu, sigma, jac_mu, jac_sigma = self.ssm.predict(state.T, action.T, True,
                                                                    False)
                    jac_pred = np.vstack((jac_mu, jac_sigma))
                return [jac_pred]

        self.jac_callback = JacFun(self.ssm, self.linearize_mu)

        return self.jac_callback

    def has_reverse(self, nadj):
        """ """
        return False

    def has_forward(self, nfwd):
        """ """
        return False

    def has_jacobian(self):
        """ """
        return True

    def get_reverse(self, name, inames, onames, opts):
        """ """
        raise NotImplementedError(
            "Need to implement this if you set has_reverse = True")

    def get_forward(self, name, inames, onames, opts):
        """ """
        raise NotImplementedError(
            "Need to implement this if you set has_forward = True")
