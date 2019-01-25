# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
"""

import numpy as np
from numpy.matlib import repmat

from utils_ellipsoid import sample_inside_ellipsoid


class MonteCarloSafetyVerification:
    """ Verify probabilistic state bounds of GP dynamic system through sampling



    """

    def __init__(self, GP):
        """ Initialize Sampler with the GP dynamic system

        Args:
            GP (SimpleGPModel):

        """

        self.GP = GP
        self.n_s = GP.n_s
        self.n_u = GP.n_u

    def sample_n_step(self, x0, K, k, n=1, n_samples=1000):
        """ Sample from the distribution of the GP system of n steps



        Args:
            x0 (numpy.ndarray[float]): vector of shape n_s x 1 representing the
                initial (deterministic) state of the system
            K (numpy.ndarray[float]): array of shape n x n_u x n_s; the state feedback
                linear controller per time step.
            k (numpy.ndarray[float]) array of shape n x n_u; the feed-forward
                control modifications per time step.
            n (int, optional): number of time steps to propagate forward
            n_samples (int, optional): number of samples to propagate through
                the system.
        Returns:
            S (numpy.ndarray[float]): n_samples x n_s samples from the pdf of
                the n-step ahead propagation of the system
            S_all (numpy.ndarray[float]): n x n_samples x n_s samples for all
                intermediate states of the system in the n-step ahead propagation
                of the system.

        """

        assert n > 0, "The time horizon n for the multi-step sampling must be positive!"
        assert np.shape(K) == (
        n, self.n_u, self.n_s), "Required shape of K is ({},{},{})".format(n, self.n_u,
                                                                           self.n_s)
        assert np.shape(k) == (n, self.n_u), "Required shape of k is ({},{})".format(n,
                                                                                     self.n_u)

        S_all = np.empty((n, n_samples, self.n_s))  # The samples for each time step

        # Get samples from the predictive distribution for the first time step
        u0 = np.dot(K[0], x0) + k[0, :, None]
        inp0 = np.vstack((x0, u0)).T

        S = self.GP.sample_from_gp(inp0, size=n_samples).reshape((n_samples, self.n_s))
        S_all[0] = S

        # Sample from other time steps (if n > 1).
        for i in range(1, n):
            U = np.dot(S, K[i].T) + repmat(k[i, :, None], n_samples, 1)
            inp = np.hstack((S, U))
            S = self.GP.sample_from_gp(inp, size=1).reshape((n_samples, self.n_s))
            S_all[i] = S

        return S.squeeze(), S_all

    def inside_ellipsoid_ratio(self, S, Q, p):
        """ Get ratio of samples inside ellipsoid for multiple sample sets / ellipsoids


        Args:
            S (numpy.ndarray[float]): array of shape n x n_samples x n_s;
            Q (numpy.ndarray[float]): array of shape n x n_s x n_s;
            p (numpy.ndarray[float]): array of shape n x n_s;

        Returns:
            Ratio (numpy.array[float]): 1darray of length n; the ratio of samples being
                inside the given ellipsoids for each time step
            R_bool (numpy.ndarray[bool]): array of shape n x n_samples; R_bool(i,j) is
                TRUE if sample j at time step i is inside the ellipsoid, otherwise False
        """
        n, n_samples, _ = np.shape(S)

        R_bool = np.empty((n, n_samples))

        for i in range(n):
            R_bool[i] = sample_inside_ellipsoid(S[i], p[i, :, None], Q[i])

        Ratio = float(np.sum(R_bool, axis=1)) / n_samples

        return Ratio, R_bool

    def _sample(self, X, K, k, n_samples=1):
        """ Sample from the predictive distribution of a GP


        Args:
            X (numpy.ndarray[float]): array of shape n_inputs x n_s representing the
                samples of the system
            K (numpy.ndarray[float]): array of shape n x n_u x n_s; the state feedback
                linear controller per time step.
            k (numpy.ndarray[float]) array of shape n x n_u; the feed-forward
                control modifications per time step.

            n_samples (int, optional): number of samples to draw from predictive
                distribution PER INPUT.

        """

        raise NotImplementedError("Not sure if necessary!")
