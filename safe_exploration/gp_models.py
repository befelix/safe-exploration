# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
"""
import warnings

import GPy
import numpy as np
import numpy.linalg as nLa
from GPy.kern import RBF, Linear, Matern52
from GPy.util.linalg import pdinv
from sklearn import cluster

from .gp_models_utils_casadi import gp_pred_function
from .utils import rgetattr, rsetattr


class SimpleGPModel():
    """ Simple Wrapper around GPy

    Wrapper around the GPy library
    to get predictions per dimension.

    TODO: Shouldnt be allowed to initialize model without training data!

    Attributes:
        gp_trained (bool): Is set to TRUE once the train() method
            was called.
        n_s (int): number of state dimensions of the dynamic system
        n_u (int): number of action/control dimensions of the dynamic system
        gps (list[GPy.core.GP]): A list of length n_s with the trained GPs

    """

    def __init__(self, n_s_out, n_s_in, n_u, X=None, y=None, m=None, kern_types=None,
                 hyp=None, train=False, Z=None):
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
        self._init_kernel_function(kern_types, hyp)
        self.do_sparse_gp = False

        self.z_fixed = False
        if not Z is None:
            self.z_fixed = True

        if X is None or y is None:  # initialize without training (no data available)
            train = False

        if train:
            self.train(X, y, m, Z=Z)

    @classmethod
    def from_dict(cls, gp_dict):
        """ Initialize GP using data from a dict

        Initialized the SimpleGPModel from a dictionary containing
        the necessary information.

        Parameters
        ----------
        gp_dict: dict
            The dictionary containing the following entries:

        """
        data_available = False
        y = None
        x = None

        if "data_path" in gp_dict and not gp_dict["data_path"] is None:
            data_path = gp_dict["data_path"]
            data = np.load(data_path)
            x = data["S"]
            y = data["y"]
            data_available = True
        elif "x" in gp_dict and "y" in gp_dict:
            x = gp_dict["x"]
            y = gp_dict["y"]
            data_available = True
        else:
            warnings.warn("""In order to be trained, GP either needs a data_path or
            the data itself (key 'data_path' or keys 'x' and 'y') -> we just instantiate the GP class, no training""")

        if "prior_model" in gp_dict:
            prior_model = gp_dict["prior_model"]
            if data_available:
                y = y - prior_model(x)

        n_s_in = gp_dict["n_s_in"]
        n_s_out = gp_dict["n_s_out"]
        n_u = gp_dict["n_u"]

        m = None
        if "m" in gp_dict:
            m = gp_dict["m"]

        kern_types = None
        if "kern_types" in gp_dict:
            kern_types = gp_dict["kern_types"]

        train = False
        if "train" in gp_dict:
            train = gp_dict["train"]
        train = train and data_available

        hyp = None
        if "hyp" in gp_dict:
            hyp = gp_dict["hyp"]

        Z = None
        if "Z" in gp_dict:
            Z = gp_dict["Z"]

        return cls(n_s_out, n_s_in, n_u, x, y, m, kern_types, hyp, train, Z)

    def to_dict(self):
        """ return a dict summarizing the object """
        gp_dict = dict()
        gp_dict["x"] = self.x_train
        gp_dict["y"] = self.y_train
        gp_dict["kern_types"] = self.kern_types
        gp_dict["hyp"] = self.hyp
        gp_dict["beta"] = self.beta
        gp_dict["inv_K"] = self.inv_K

        return gp_dict

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

        n_beta = np.shape(Z)[0]
        beta = np.empty((n_beta, self.n_s_out))

        inv_K = [None] * self.n_s_out
        process_noise = np.empty((self.n_s_out,))
        gps = [None] * self.n_s_out

        for i in range(self.n_s_out):
            kern = self.base_kerns[i]

            if self.do_sparse_gp:
                y_i = y_train[:, i].reshape(-1, 1)
                model_gp = GPy.models.SparseGPRegression(X_train, y_i, kernel=kern, Z=Z)

            else:
                y_i = y_z[:, i].reshape(-1, 1)
                model_gp = GPy.models.GPRegression(Z, y_i, kernel=kern)

            if opt_hyp:
                model_gp.optimize(max_iters=1000, messages=True)

            post = model_gp.posterior

            if noise_diag > 0.:
                model_gp.likelihood.variance.fix(noise_diag)
                # inv_K[i] = pdinv(post._K+float(model_gp.Gaussian_noise.variance+noise_diag)*np.eye(n_beta))[0]
            # else:
            #    inv_K[i] = post.woodbury_inv

            post = model_gp.posterior
            inv_K[i] = post.woodbury_inv

            beta[:, i] = post.woodbury_vector.reshape(-1, )
            process_noise[i] = model_gp.Gaussian_noise.variance
            gps[i] = model_gp

        # create a dictionary of kernel paramters
        self.hyp = self._create_hyp_dict(gps, self.kern_types)

        # update the class attributes
        if self.z_fixed:
            self.z = self.Z
        else:
            self.z = Z
        self.inv_K = inv_K
        self.beta = beta
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

                post = self.gps[i].posterior

                if noise_diag > 0.0:
                    inv_K[i] = pdinv(post._K + float(
                        self.gps[i].Gaussian_noise.variance + noise_diag) * np.eye(
                        n_beta))[0]
                else:
                    inv_K[i] = post.woodbury_inv

                beta[:, i] = post.woodbury_vector.reshape(-1, )

            self.x_train = x_new
            self.y_train = y_new
            self.z = Z
            self.inv_K = inv_K
            self.beta = beta

    def _init_kernel_function(self, kern_types=None, hyp=None):
        """ Initialize GPy kernel functions based on name. Check if supported.

        Utility function to return a kernel based on its type name.
        Checks if the kernel type is supported.

        Parameters
        ----------
        kern_types: n_s x 0 array_like[str]
            The names of the kernels for each dimension

        Returns
        -------
        kern: GPy.Kern
            The Gpy kernel function
        """

        input_dim = self.n_s_in + self.n_u
        kerns = [None] * self.n_s_out

        if hyp is None:
            hyp = [None] * self.n_s_out
        warnings.warn(
            "Changed the kernel structure from the cdc paper implementation, see old structure commented out")

        """
        if kern_types[i] == "rbf":
                    kern_i = RBF(input_dim, ARD = True)
                elif kern_types[i] == "lin_rbf":
                    kern_i = Linear(1,active_dims = [1])*RBF(1,active_dims=[1]) + Linear(input_dim,ARD=True)
                elif kern_types[i] == "lin_mat52":
                    kern_i = Linear(1,active_dims = [1])*Matern52(1,active_dims=[1]) + Linear(input_dim,ARD=True)
                else:
        """

        if kern_types is None:
            kern_types = [None] * self.n_s_out
            for i in range(self.n_s_out):
                kern_types[i] = "rbf"
                kerns[i] = RBF(input_dim, ARD=True)

        else:
            for i in range(self.n_s_out):
                hyp_i = hyp[i]
                if kern_types[i] == "rbf":
                    kern_i = RBF(input_dim, ARD=True)
                elif kern_types[i] == "lin_rbf":
                    kern_i = Linear(input_dim) * RBF(input_dim) + Linear(input_dim,
                                                                         ARD=True)
                elif kern_types[i] == "lin_mat52":
                    kern_i = Linear(input_dim) * Matern52(input_dim) + Linear(input_dim,
                                                                              ARD=True)
                else:
                    raise ValueError(
                        "kernel type '{}' not supported".format(kern_types[i]))

                if not hyp_i is None:
                    for k, v in list(hyp_i.items()):
                        try:
                            rsetattr(kern_i, k, v)
                            kern_hyp = rgetattr(kern_i, k)
                            kern_hyp.fix()

                        except:
                            warnings.warn(
                                "Cannot set and fix hyperparameter: {}".format(k))
                kerns[i] = kern_i

        self.base_kerns = kerns
        self.kern_types = kern_types

    def _create_hyp_dict(self, gps, kern_types):
        """ Create a hyperparameter dict from the individual supported kernels

        Parameters
        ----------
        gps: n_s x 0 array_like[GPy.GP]
            The list of trained GPs
        kern_type:
            The kernel identifier

        Returns
        -------
        hyp: list[dict]
            A list of dictionaries containing the hyperparameters of the kernel type
            for each dimension.
        """

        hyp = [None] * self.n_s_out

        for i in range(self.n_s_out):
            hyp_i = dict()
            if kern_types[i] == "rbf":

                hyp_i["lengthscale"] = np.reshape(gps[i].kern.lengthscale, (-1,))
                hyp_i["variance"] = gps[i].kern.variance

            elif kern_types[i] == "lin_rbf":

                hyp_i["prod.rbf.lengthscale"] = np.array(
                    [gps[i].kern.mul.rbf.lengthscale])
                hyp_i["prod.rbf.variance"] = gps[i].kern.mul.rbf.variance
                hyp_i["prod.linear.variances"] = np.array(
                    gps[i].kern.mul.linear.variances)
                hyp_i["linear.variances"] = np.array([gps[i].kern.linear.variances])

            elif kern_types[i] == "lin_mat52":
                hyp_i["prod.mat52.lengthscale"] = np.array(
                    [gps[i].kern.mul.Mat52.lengthscale])
                hyp_i["prod.mat52.variance"] = gps[i].kern.mul.Mat52.variance
                hyp_i["prod.linear.variances"] = np.array(
                    gps[i].kern.mul.linear.variances)
                hyp_i["linear.variances"] = np.array([gps[i].kern.linear.variances])
            else:
                raise ValueError("kernel type not supported")
            hyp[i] = hyp_i
        return hyp

    def predict(self, x_new, quantiles=None, compute_gradients=False):
        """ Compute the predictive mean and variance for a set of test inputs


        """
        assert self.gp_trained, "Cannot predict, need to train the GP first!"

        T = np.shape(x_new)[0]
        y_mu_pred = np.empty((T, self.n_s_out))
        y_sigm_pred = np.empty((T, self.n_s_out))

        for i in range(self.n_s_out):

            if quantiles is None:
                y_mu_pred[:, None, i], y_sigm_pred[:, None, i] = self.gps[
                    i].predict_noiseless(x_new)
            else:
                raise NotImplementedError()

        if compute_gradients:
            grad_mu = self.predictive_gradients(x_new)
            return y_mu_pred, y_sigm_pred, grad_mu

        return y_mu_pred, y_sigm_pred

    def predict_casadi_symbolic(self, x_new, compute_grads=False):
        """ Return a symbolic casadi function representing predictive mean/variance

        """
        assert self.gp_trained, "Cannot predict, need to train the GP first!"

        out_dict = gp_pred_function(x_new, self.z, self.beta, self.hyp, self.kern_types,
                                    self.inv_K, True, compute_grads)
        mu_new = out_dict["pred_mu"]
        sigma_new = out_dict["pred_sigma"]
        if compute_grads:
            jac_mu = out_dict["jac_mu"]
            return mu_new, sigma_new, jac_mu

        return mu_new, sigma_new

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

        if grad_sigma:
            # would be easy to implement since it is returend by GPys predictive gradient
            # but we dont need it right now
            raise NotImplementedError("Gradient of sigma not implemented")

        T = np.shape(x_new)[0]

        grad_mu_pred = np.empty([T, self.n_s_out, self.n_s_in + self.n_u])

        for i in range(self.n_s_out):
            g_mu_pred, _ = self.gps[i].predictive_gradients(x_new)
            grad_mu_pred[:, i, :] = g_mu_pred[:, :, 0]

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
