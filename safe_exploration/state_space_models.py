# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
"""
import warnings

import GPy
import casadi as cas
import numpy as np
import numpy.linalg as nLa
from sklearn import cluster

from utils import reshape_derivatives_3d_to_2d


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
            return [cas.Sparsity.dense(self.ssm.num_states, 1),
                    cas.Sparsity.dense(self.ssm.num_states, 1),
                    cas.Sparsity.dense(self.ssm.num_states,
                                       self.ssm.num_states + self.ssm.num_actions)][
                i]  # mean, mean_jacobian and variance
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
            mu, sigma, jac_mu, _ = self.ssm.predict(state, action, True, False)
            return [mu, sigma, jac_mu]
        else:
            mu, sigma = self.ssm.predict(state, action)
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
                                           self.ssm.num_states + self.ssm.num_actions)][
                    i]

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
                        state, action, True, False)

                    gradient_jac_mu_compressed = reshape_derivatives_3d_to_2d(
                        gradients_jac_mu)
                    jac_pred = np.vstack(
                        (jac_mu, jac_sigma, gradient_jac_mu_compressed))
                else:
                    mu, sigma, jac_mu, jac_sigma = self.ssm.predict(state, action, True,
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


class CasadiGaussianProcess(StateSpaceModel):
    """ Simple Wrapper around GPy

    Wrapper around the GPy library
    to get predictions per dimension.

    Attributes:
        gp_trained (bool): Is set to TRUE once the train() method
            was called.
        n_s (int): number of state dimensions of the dynamic system
        n_u (int): number of action/control dimensions of the dynamic system
        gps (list[GPy.core.GP]): A list of length n_s with the trained GPs

    """

    def __init__(self, n_s_out, n_s_in, n_u, kerns, X=None, y=None, m=None, train=False,
                 Z=None):
        """ Initialize GP Model ( possibly without training set)

        Parameters
        ----------
            X (np.ndarray[float], optional): Training inputs
            y (np.ndarray[float], optional): Training targets
            m (int, optional): number of datapoints to be selected (randomly)
                    from the training set
            kern_types (list[str]): a list of pre-specified covariance function types
            Z (tuple): Inducing points

        """
        self.n_s_out = n_s_out
        self.n_s_in = n_s_in
        self.n_u = n_u
        self.gp_trained = False
        self.m = m
        self.Z = Z
        self.kerns = kerns
        self.do_sparse_gp = False

        self.z_fixed = False
        if not Z is None:
            self.z_fixed = True

        if X is None or y is None:  # initialize without training (no data available)
            train = False
        if train:
            self.train(X, y, m, Z=Z)

    def train(self, X, y, m=None, opt_hyp=True, noise_diag=1e-5, Z=None,
              choose_data=True):
        """ Train a GP for each state dimension

        Args:
            X: Training inputs of size [N, n_s + n_u]
            y: Training targets of size [N, n_s]
        """
        n_data, _ = np.shape(X)

        X_train = X
        y_train = y

        if not m is None:
            if self.do_sparse_gp and not Z is None:
                pass
            else:
                if n_data < m:
                    warnings.warn("""The desired number of datapoints is not available. Dataset consist of {}
                           Datapoints! """.format(n_data))
                    X_train = X
                    y_train = y
                    Z = X
                    y_z = y_train
                else:

                    if choose_data:
                        Z, y_z = self.choose_datapoints_maxvar(X_train, y_train, self.m)

                    else:
                        idx = np.random.choice(n_data, size=m, replace=False)
                        Z = X[idx, :]
                        y_z = y[idx, :]
        else:
            if self.do_sparse_gp:
                raise ValueError(
                    "Number of inducing points m needs to be specified for sparse gp regression")

            Z = X_train
            y_z = y_train

        gps = [None] * self.n_s_out

        for i in range(self.n_s_out):
            kern = self.kerns[i]

            if self.do_sparse_gp:
                y_i = y_train[:, i].reshape(-1, 1)
                model_gp = GPy.models.SparseGPRegression(X_train, y_i, kernel=kern, Z=Z)
            else:
                y_i = y_z[:, i].reshape(-1, 1)
                model_gp = GPy.models.GPRegression(Z, y_i, kernel=kern)

            if opt_hyp:
                model_gp.optimize(max_iters=1000, messages=True)

            post = model_gp.posterior
            gps[i] = model_gp

        # update the class attributes
        if self.z_fixed:
            self.z = self.Z
        else:
            self.z = Z

        self.gps = gps
        self.gp_trained = True
        self.x_train = X_train
        self.y_train = y_train

    def choose_datapoints_maxvar(self, x, y, m, k=10, min_ratio_k=0.25, n_reopt_gp=1):
        """ Choose datapoints for the GP based on the maximum predicted variance criterion

        Parameters
        ----------
        x: n x (n_s + n_u) array[float]
            The training set
        y: n x n_s
            The training targets

        m: The number of datapoints to select

        """
        n_data = np.shape(x)[0]

        if n_data <= m:  # we have less data than the subset m -> use the whole dataset
            return x, y

        if not self.gp_trained:
            self.train(x, y, m=None)

        k = np.minimum(int(n_data * min_ratio_k), k)

        # get initial set of datapoints using k-means
        km = cluster.KMeans(n_clusters=k)
        clust_ids = km.fit_predict(x)

        sort_idx = np.argsort(clust_ids)
        clust_ids_sorted = clust_ids[sort_idx]

        unq_first = np.concatenate(
            ([True], clust_ids_sorted[1:] != clust_ids_sorted[:-1]))
        unq_items = clust_ids_sorted[unq_first]
        unq_count = np.diff(np.nonzero(unq_first)[0])
        unq_idx = np.split(sort_idx, np.cumsum(
            unq_count))  # list containing k arrays -> the indices of the samples in the corresponding classes

        unq_per_cluster_idx = [np.random.choice(clust_idx) for clust_idx in unq_idx]
        idx_not_selected = np.setdiff1d(np.arange(n_data), unq_per_cluster_idx)

        x_chosen = x[np.array(unq_per_cluster_idx), :]
        y_chosen = y[np.array(unq_per_cluster_idx), :]

        x_pool = x[idx_not_selected, :]
        y_pool = y[idx_not_selected, :]

        chunks = int((m - k) / (n_reopt_gp + 1))

        for i in range(m - k):
            if i % chunks == 0:
                self.train(x_pool, y_pool, m=None)

            _, pred_var_pool = self.predict(x_pool)
            idx_max_sigma = np.argmax(np.sum(pred_var_pool, axis=1))

            x_chosen = np.vstack((x_chosen, x_pool[None, idx_max_sigma, :]))
            y_chosen = np.vstack((y_chosen, y_pool[None, idx_max_sigma, :]))

            x_pool = np.delete(x_pool, (idx_max_sigma), axis=0)
            y_pool = np.delete(y_pool, (idx_max_sigma), axis=0)

            for j in range(self.n_s_out):
                self.gps[j].set_XY(x_chosen, y_chosen[:, j, None])

        return x_chosen, y_chosen

    def update_model(self, x, y, opt_hyp=False, replace_old=True, noise_diag=1e-5,
                     choose_data=True):
        """ Update the model based on the current settings and new data

        Parameters
        ----------
        x: n x (n_s + n_u) array[float]
            The training set
        y: n x n_s
            The training targets
        train: bool, optional
            If this is set to TRUE the hyperparameters are re-optimized
        """
        if replace_old:
            x_new = x
            y_new = y
        else:
            x_new = np.vstack((self.x_train, x))
            y_new = np.vstack((self.y_train, y))

        if opt_hyp or not self.gp_trained:
            self.train(x_new, y_new, self.m, opt_hyp=opt_hyp, Z=self.Z)
        else:
            n_data = np.shape(x_new)[0]
            inv_K = [None] * self.n_s_out
            if self.m is None:
                n_beta = n_data
                Z = x_new
                y_z = y_new
            else:

                if n_data < self.m:
                    warnings.warn("""The desired number of datapoints is not available. Dataset consist of {}
                           Datapoints! """.format(n_data))
                    Z = x_new
                    y_z = y_new
                    n_beta = n_data
                else:
                    if choose_data:
                        Z, y_z = self.choose_datapoints_maxvar(x_new, y_new, self.m)

                    else:
                        idx = np.random.choice(n_data, size=self.m, replace=False)
                        Z = x_new[idx, :]
                        y_z = y_new[idx, :]
                    n_beta = self.m

            beta = np.empty((n_beta, self.n_s_out))

            for i in range(self.n_s_out):
                if self.do_sparse_gp:
                    self.gps[i].set_XY(x_new, y_new[:, i].reshape(-1, 1))
                    if not self.z_fixed:
                        self.gps[i].set_Z(Z)
                else:
                    self.gps[i].set_XY(Z, y_z[:, i].reshape(-1, 1))

            self.x_train = x_new
            self.y_train = y_new
            self.z = Z

    def predict(self, states, actions, jacobians=False, full_cov=False):
        """ Compute the predictive mean and variance for a set of test inputs


        """
        assert self.gp_trained, "Cannot predict, need to train the GP first"

        T = np.shape(states)[0]
        y_mu_pred = np.empty((T, self.n_s_out))
        y_sigm_pred = np.empty((T, self.n_s_out))

        z_new = np.hstack((states, actions))
        for i in range(self.n_s_out):
            y_mu_pred[:, None, i], y_sigm_pred[:, None, i] = self.gps[
                i].predict_noiseless(z_new)

        if jacobians:
            grad_mu, grad_sigma = self.predictive_gradients(z_new, True)
            return y_mu_pred, y_sigm_pred, grad_mu, grad_sigma

        return y_mu_pred, y_sigm_pred

    def linearize_predict(self, states, actions, jacobians=False, linearize_cov=False):
        """ """
        raise NotImplementedError("Need to implement this")

        if jacobians and full_cov:
            raise NotImplementedError('Jacobians of full covariance not supported.')

    def predictive_gradients(self, x_new, grad_sigma=False):
        """ Compute the gradients of the predictive mean/variance w.r.t. inputs

        Parameters
        ----------
        x_new: T x (n_s + n_u) array[float]
            The test inputs to compute the gradients at
        grad_sigma: bool, optional
            Additionaly returns the gradients of the predictive variance w.r.t. the inputs if
            this is set to TRUE

        """
        T = np.shape(x_new)[0]

        grad_mu_pred = np.empty([T, self.n_s_out, self.n_s_in + self.n_u])
        grad_var_pred = np.empty([T, self.n_s_out, self.n_s_in + self.n_u])

        for i in range(self.n_s_out):
            g_mu_pred, g_var_pred_ = self.gps[i].predictive_gradients(x_new)
            grad_mu_pred[:, i, :] = g_mu_pred[:, :, 0]
            grad_var_pred[:, i, :] = grad_var_pred[:, :, 0]

        if grad_sigma:
            return grad_mu_pred, grad_var_pred

        return grad_mu_pred

    def sample_from_gp(self, inp, size=10):
        """ Sample from GP predictive distribution


        Args:
            inp (numpy.ndarray[float]): array of shape n x (n_s + n_u); the
                test inputs
            size (int, optional): number of samples per test point

        Returns:
            S (numpy.ndarray[float]): array of shape n x size x n_s; array of size samples
            of the posterior distribution per test input

        """
        n = np.shape(inp)[0]
        S = np.empty((n, size, self.n_s_out))

        for i in range(self.n_s_out):
            S[:, :, i] = self.gps[i].posterior_samples_f(inp, size=size, full_cov=False)

        return S

    def information_gain(self, x=None):
        """ Mutual information between samples and system """

        if x is None:
            x = self.z

        n_data = np.shape(x)[0]
        inf_gain_x_f = [None] * self.n_s_out
        for i in range(self.n_s_out):
            noise_var_i = float(self.gps[i].Gaussian_noise.variance)
            inf_gain_x_f[i] = np.log(
                nLa.det(np.eye(n_data) + (1 / noise_var_i) * self.gps[i].posterior._K))

        return inf_gain_x_f
