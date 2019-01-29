# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:49:49 2017

@author: tkoller
"""

import numpy as np
import numpy.linalg as nLa

from ..utils import unavailable
try:
    import matplotlib.pyplot as plt
    _has_matplotlib = True
except:
    _has_matplotlib = False

@unavailable(not _has_matplotlib,"matplotlib")
def plot_ellipsoid_3D(p, q, ax, n_points=100):
    """ Plot an ellipsoid in 3D

    Based on
    https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib

    TODO: Untested!

    Parameters
    ----------
    p: 3x1 array[float]
        Center of the ellipsoid
    q: 3x3 array[float]
        Shape matrix of the ellipsoid
    ax: matplotlib.Axes object
        Ax on which to plot the ellipsoid

    Returns
    -------
    ax: matplotlib.Axes object
        The Ax containing the ellipsoid

    """

    assert np.shape(p) == (3, 1), "p needs to be a 3x1 vector"
    assert np.shape(q) == (3, 3), "q needs to be a spd 3x3 matrix"
    assert np.allclose(q, 0.5 * (q + q.T), "q needs to be spd")
    # transform to radius/center parametrization
    U, s, rotation = linalg.svd(q)
    assert np.all(s > 0), "q needs to be positive definite"
    radii = 1.0 / np.sqrt(s)

    # get x,y,z of sphere and transform
    u = np.linspace(0.0, 2.0 * np.pi, n_points)
    v = np.linspace(0.0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]],
                                                 rotation) + center

    # plot the result
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='b', alpha=0.2)
    return ax

@unavailable(not _has_matplotlib,"matplotlib")
def plot_ellipsoid_2D(p, q, ax, n_points=100, color="r"):
    """ Plot an ellipsoid in 2D

    TODO: Untested!

    Parameters
    ----------
    p: 3x1 array[float]
        Center of the ellipsoid
    q: 3x3 array[float]
        Shape matrix of the ellipsoid
    ax: matplotlib.Axes object
        Ax on which to plot the ellipsoid

    Returns
    -------
    ax: matplotlib.Axes object
        The Ax containing the ellipsoid
    """
    plt.sca(ax)
    r = nLa.cholesky(q).T;  # checks spd inside the function
    t = np.linspace(0, 2 * np.pi, n_points);
    z = [np.cos(t), np.sin(t)];
    ellipse = np.dot(r, z) + p;
    handle, = ax.plot(ellipse[0, :], ellipse[1, :], color)

    return ax, handle
