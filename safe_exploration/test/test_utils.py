# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:20:28 2017

@author: tkoller
"""
import numpy as np

from ..utils import sample_inside_polytope


def test_sample_inside_polytope():
    """

    polytope:
        -.3 < x_1 < .4
        -.2 < x_2 < .2
    """
    x = np.array([[0.1, 0.15], [0.0, 0.0], [.5, .15]])

    a = np.vstack((np.eye(2), -np.eye(2), -np.eye(2)))
    b = np.array([.4, .2, .3, .2, .3, .2])[:, None]

    res = sample_inside_polytope(x, a, b)

    res_expect = np.array([True, True, False])  # should be: inside, inside, not inside

    assert np.all(
        res == res_expect), "Are the right samples inside/outside the polyope?"
