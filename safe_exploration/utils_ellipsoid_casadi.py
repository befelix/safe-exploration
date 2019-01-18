# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:08:49 2017

@author: tkoller

"""
from casadi import sqrt, trace, diag
from casadi.tools import *

def sum_two_ellipsoids(p_1,q_1,p_2,q_2,c=None):
    """ overapproximation of sum of two ellipsoids

    Computes the ellipsoidal overapproximation of the sum of two n-dimensional
    ellipsoids.
    from:
    "A Kurzhanski, I Valyi - Ellipsoidal Calculus for Estimation and Control"

    Parameters
    ----------
    p_1,p_2: n x 1 array
        The centers of the ellipsoids to sum
    q_1,q_2: n x n array
        The shape matrices of the two ellipsoids
    c: float, optional
        The
    Returns
    -------
    p_new: n x 1 array
        The center of the resulting ellipsoid
    q_new: n x n array
        The shape matrix of the resulting ellipsoid
    """

    ## choose p s.t. the trace of the new shape matrix is minimized
    if c is None:
        c = sqrt(trace(q_1)/trace(q_2))

    p_new = p_1+p_2
    q_new = (1+(1./c))*q_1 + (1+c)*q_2

    return p_new, q_new


def ellipsoid_from_rectangle(u_b):
    """ Compute ellipsoid covering box

    Given a box defined by

        B = [l_b[0],u_b[0]] x ... x [l_b[-1],u_b[-1]],
    where l_b = -u_b (element-wise),
    we compute the minimum enclosing axis-aligned ellipsoid in closed-form
    as the solution to a linear least squares problem.

    Method is described in:
        [1] :


    Parameters
    ----------
        u_b: n x 1 array
            array containing upper bounds of intervals defining box (see above)
    Returns
    -------
        q: np.ndarray[float, n_dim = 2], array of size n x n
            The (diagonal) shape matrix of covering ellipsoid

    """
    n,_ = u_b.shape
    d = n*u_b**2
    q = diag(d)

    return q




