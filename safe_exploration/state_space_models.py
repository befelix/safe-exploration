# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
"""
import numpy as np
import casadi as cas
import copy

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
    forward_cache : ??
        Stores the forward pass information when
        calling predict(state, action). Can
        then be used for reverse directional derivatives
        (e.g forward_cache.backward(v), where v in (n+m)x1.
        See get_reverse method.
    _linearize_forward_cache : ??
        Stores the forward pass information when
        calling predict(state, action). Can
        then be used for reverse directional derivatives
        (e.g forward_cache.backward(v), where v in (n+m)x1.
        See get_reverse method.
    """

    def __init__(self, num_states, num_actions, has_jacobian=True, has_reverse=False):
        self.num_states = num_states
        self.num_actions = num_actions

        self._forward_cache = None
        self._linearize_forward_cache = None
        self.has_jacobian = has_jacobian
        self.has_reverse = has_reverse

    def __call__(self, states, actions):
        """

        Parameters
        ----------
        states : np.ndarray
            A (N x n) array of states.
        actions : np.ndarray
            A (N x m) array of actions.

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

        """
        return self.predict(states, actions, True, False)

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
            Function with vector-valued inputs x (nx1 array), u (mx1 array)
            that returns the predictive mean of the SSM evaluated at (x,u).
        pred_var: Casadi function
            Function with vector-valued inputs x (nx1 array), u (mx1 array)
            that returns the predictive variance of the SSM evaluated at (x,u).
        jac_pred_mu: Casadi function
            Function with vector-valued inputs x (nx1 array), u (mx1 array)
            that returns the jacobian of te predictive mean of the SSM evaluated at (x,u).

        """
        return CasadiSSMEvaluator(copy.deepcopy(self), linearize_mu, self.has_jacobian, self.has_reverse)

    def get_reverse(self, seed):
        """ Get directional derivative through reverse automatic differentiation

        Compute the derivative v^T \cdot J(w), where
        v \in \R^{n+n} is the seed vector and J(w) is the Jacobian
        of the output mu(w), var(w) = ssm(w) w.r.t some input w \in \R^{n+m}.

        In reverse automatic differentiation, we have mu(w), var(w) already computed in the
        forward pass and stored.

        """
        raise NotImplementedError("Need to implement this in a sublass when providing reverse AD for forward model")

    def get_linearize_reverse(self, seed):
        """ Get directional derivative through reverse automatic differentiation

            Compute the directional derivative v^T \cdot J(w), where
            v \in \R^{n + n + n*(n+m)} is the seed and J(w) is the stacked Jacobian
            of the output mu(w), var(w), jac_mu(w) = ssm(w), w \in \R^{n+m}.

            In reverse automatic differentiation, we have mu(w), var(w), jac_mu(w) already computed in the
            forward pass and stored.

        """
        raise NotImplementedError("Need to implement this in a sublass when providing reverse AD for linearized forward")

    def update_model(self, train_x, train_y, opt_hyp=False, replace_old=False):
        """ Update the state space model

        Parameters
        ----------
        train_x: np.ndarray
            A (N x (n+m)) array of state-action pairs as training inputs
        train_y: np.ndarray
            A (N x n) array of states as training targets

        opt_hyp: Bool
            True, if we want to reoptimize hyperparameters
        replace_old: Bool
            True, if we want to replace previous training data with
            the current set of train_x/train_y data

        """
        raise NotImplementedError("Need to implement this in subclass")


class CasadiSSMEvaluator(cas.Callback):
    """ Base class for a casadi evalauator of a state space model

    Attributes
    ----------
    ssm: StateSpaceModel
            The underlying state space model

    """

    def __init__(self, ssm, linearize_mu=True, has_jacobian=True,
                 has_reverse=False, opts={}):
        """

          Parameters
          ----------
          ssm: StateSpaceModel
            The underlying state space model

          linearize_mu: Bool, optional
            True, if we want to linearize the predictive mean

        """
        cas.Callback.__init__(self)

        self.v_has_jacobian = has_jacobian
        self.v_has_reverse = has_reverse
        self.v_has_forward = False  # option not implemented yet
        any_diff = self.v_has_jacobian or self.v_has_reverse or self.v_has_forward
        if not any_diff:
            raise ValueError("Need to specify either the ")

        self.ssm = ssm
        self.linearize_mu = linearize_mu
        self.construct("CasadiModelEvaluator", opts)

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
            mu, sigma, jac_mu = self.ssm.linearize_predict(state.T, action.T, False, False)

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

            def get_n_in(self):
                """ """
                if self.linearize_mu:
                    return 5
                else:
                    return 4

            def get_n_out(self):
                """ """
                return 1

            def has_reverse(self, nadj):
                """ """
                return False

            def has_forward(self, nfwd):
                """ """
                return False

            def has_jacobian(self):
                """ """
                return False

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
        return self.v_has_reverse and nadj == 1

    def has_forward(self, nfwd):
        """ """
        return self.v_has_forward

    def has_jacobian(self):
        """ """

        return self.v_has_jacobian

    def get_reverse(self, nadj, name, inames, onames, opts):
        """ Return the Callback function for the reverse

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
        if not self.v_has_reverse:
            raise ValueError("Calling reverse even though it is not provided! This should not happen.")

        class BackFun(cas.Callback):
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

            def get_n_in(self):
                """ """
                if self.linearize_mu:
                    return 2 + 3 + 3  # n_in + n_out + n_out
                else:
                    return 2 + 2 + 2  # n_in + n_out + n_out

            def get_n_out(self):
                """ """
                return 2  # n_in

            def has_reverse(self, nadj):
                """ """
                return False

            def has_forward(self, nfwd):
                """ """
                return False

            def has_jacobian(self):
                """ """
                return False

            def get_sparsity_in(self, i):
                """

                1. input dimensionalities
                2. output dimensionalities
                3. gradient dimensionalities
                """
                if self.linearize_mu:
                    return [cas.Sparsity.dense(self.ssm.num_states, 1),
                            cas.Sparsity.dense(self.ssm.num_actions, 1),
                            cas.Sparsity.dense(self.ssm.num_states, 1),
                            cas.Sparsity.dense(self.ssm.num_states, 1),
                            cas.Sparsity.dense(self.ssm.num_states,
                                               self.ssm.num_states + self.ssm.num_actions),
                            cas.Sparsity.dense(self.ssm.num_states,
                                               1),
                            cas.Sparsity.dense(self.ssm.num_states,
                                               1),
                            cas.Sparsity.dense(self.ssm.num_states,
                                               self.ssm.num_states + self.ssm.num_actions)
                            ][i]
                else:
                    return [cas.Sparsity.dense(self.ssm.num_states, 1),
                            cas.Sparsity.dense(self.ssm.num_actions, 1),
                            cas.Sparsity.dense(self.ssm.num_states, 1),
                            cas.Sparsity.dense(self.ssm.num_states, 1),
                            cas.Sparsity.dense(self.ssm.num_states,
                                               self.ssm.num_states + self.ssm.num_actions),
                            cas.Sparsity.dense(self.ssm.num_states,
                                               self.ssm.num_states + self.ssm.num_actions)
                            ][i]

            def get_sparsity_out(self, i):
                """ """

                return [cas.Sparsity.dense(self.ssm.num_states, 1), cas.Sparsity.dense(self.ssm.num_actions, 1)][i]

            def eval(self, arg):
                """ Evaluate the Jacobian of the ssm predictive mean/variance

                Parameters
                ----------
                arg: list
                    List of length n_in containing the inputs to the function (state,action)

                Returns
                -------
                jac_pred: np.ndarray
                    A (nx1) array containing the stacked jacobians of the predictive mean
                    and predictive variance of the ssm
                """

                mean_seed = arg[5]
                cov_seed = arg[6]

                seed = cas.vertcat(mean_seed, cov_seed)

                if self.linearize_mu:
                    jac_mean_seed = arg[7]
                    seed = cas.vertcat(seed, jac_mean_seed.reshape((-1, 1)))
                    adj_state, adj_action = self.ssm.get_linearize_reverse(np.array(seed, dtype=np.float32))

                    return cas.DM(adj_state), cas.DM(adj_action)
                else:
                    adj_state, adj_action = self.ssm.get_reverse(np.array(seed, dtype=np.float32))
                    return cas.DM(adj_state), cas.DM(adj_action)

        self.reverse_callback = BackFun(self.ssm, self.linearize_mu)

        return self.reverse_callback

    def get_forward(self, name, inames, onames, opts):
        """ """
        if not self.v_has_forward:
            raise ValueError("Calling forward even though it is not provided! This should not happen.")
        raise NotImplementedError("Not implemented yet")
