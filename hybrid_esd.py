#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 18:37:40 2019

@author: vaibhav
"""


from scipy.stats import t as student_t
import numpy as np
import pandas as pd
from statsmodels.robust.scale import mad as mad_func

x = np.array([0, 1, 1, 0, 4, 6, 2, 5, 0, 1])

max_outliers = 3
alpha = 0.05

n = x.shape[0]

all_idx = list(range(0, n))
outlier_idxs = []

n_anomalies = 0

for j in range(0, max_outliers):
    
    i = j + 1
    
    median = np.median(x)
    #mad = np.median(np.abs(x - median))
    mad = mad_func(x)
    # mad_func gives adjusted mad - multiplied by ~1.48
    # this is the same as mad function in R
    
    if mad == 0:
        break
    
    # calculate test stat values
    test_stats = np.abs((x - median)/mad)
    max_test_stat_idx = np.argmax(test_stats)
    max_test_stat = test_stats[max_test_stat_idx] 
    outlier_idx = all_idx[max_test_stat_idx]
    
    # calculate critical values
    p = 1 - alpha / float(2 * (n - i + 1))
    t = student_t.ppf(p, (n - i - 1))
    crit_val = t * (n - i) / float(np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
    
    outlier_idxs.append(outlier_idx)
    x = np.delete(x, max_test_stat_idx)
    _ = all_idx.pop(max_test_stat_idx)
    
    if max_test_stat > crit_val:
        n_anomalies = i
        
outlier_idxs = outlier_idxs[0:n_anomalies]
    

    
    