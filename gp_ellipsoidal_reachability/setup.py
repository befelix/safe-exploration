# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:18:35 2017

@author: Torsten
"""

from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["ilqr_cython.pyx"]),
    include_dirs=[numpy.get_include()]
)
