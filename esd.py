#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:47:25 2019

@author: vaibhav
"""

from itertools import groupby
from math import trunc, sqrt
from scipy.stats import t as student_t
from statsmodels.robust.scale import mad as mad_func
#import statsmodels.api as sm
import sys
import numpy as np
import pandas as pd

# Remove the seasonal component, and the median of the data to create the univariate remainder
d = {
    'timestamp': list(range(0, 10)),
    'value': [0, 1, 1, 0, 4, 6, 2, 5, 0, 1]
}

data = pd.DataFrame(d)

# Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
#'max_outliers = int(num_obs * k)
max_outliers = 3
alpha = 0.05

## Define values and vectors.
n = len(data.timestamp)
R_idx = list(range(max_outliers))

num_anoms = 0

# Compute test statistic until r=max_outliers values have been
# removed from the sample.
for i in range(1, max_outliers + 1):

    ares = (data.value - data.value.median()).abs()

    # protect against constant time series
    data_sigma = mad_func(data.value)
    if data_sigma == 0:
        break

    ares = ares / float(data_sigma)

    R = ares.max()

    temp_max_idx = ares[ares == R].index.tolist()[0]

    R_idx[i - 1] = temp_max_idx

    data = data[data.index != R_idx[i - 1]]

    p = 1 - alpha / float(2 * (n - i + 1))

    t = student_t.ppf(p, (n - i - 1))
    lam = t * (n - i) / float(sqrt((n - i - 1 + t**2) * (n - i + 1)))

    if R > lam:
        num_anoms = i

if num_anoms > 0:
    R_idx = R_idx[:num_anoms]
else:
    R_idx = None