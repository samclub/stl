#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:33:23 2019

@author: vaibhav
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(name='stl_func',
      ext_modules=cythonize('stl_func.pyx'))