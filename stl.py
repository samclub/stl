#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:08:11 2019

@author: vaibhav
"""

import numpy as np
import math
from stl_func import stl_func

def nextodd(x):
    x = int(round(x))
    if x%2 == 0:
        x += 1
    return x

#-     PURPOSE
#-     STL decomposes a time series into seasonal and trend  components.
#-     It returns the components and robustness weights.
#-     SYNOPSIS
#-     stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump,
#-             nljump, ni, no, rw, season, trend, work)
#-     integer n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump,
#-             nljump, ni, no
#-     real y(n), rw(n), season(n), trend(n), work(n+2*np,5)
#-     ARGUMENTS
#-     y       input, time series to be decomposed.
#-     n       input, number of values in y.
#-     np      input, the period of the seasonal component. For example,
#-             if  the  time series is monthly with a yearly cycle, then
#-             np=12.
#-     ns      input, length of the seasonal smoother.  The value of  ns
#-             should be an odd integer greater than or equal to 3; ns>6
#-             is recommended.   As  ns  increases  the  values  of  the
#-             seasonal component at a given point in the seasonal cycle
#-             (e.g., January values of a monthly series with  a  yearly
#-             cycle) become smoother.
#-     nt      input, length of the trend smoother.   The  value  of  nt
#-             should  be  an  odd integer greater than or equal to 3; a
#-             value of nt between 1.5*np and 2*np is  recommended.   As
#-             nt  increases  the  values  of the trend component become
#-             smoother.
#-     nl      input, length of the low-pass filter.  The  value  of  nl
#-             should  be an odd integer greater than or equal to 3; the
#-             smallest odd integer greater  than  or  equal  to  np  is
#-             recommended.
#-     isdeg   input, degree of locally-fitted  polynomial  in  seasonal
#-             smoothing.  The value is 0 or 1.
#-     itdeg   input,  degree  of  locally-fitted  polynomial  in  trend
#-             smoothing.  The value is 0 or 1.
#-     ildeg   input, degree of locally-fitted  polynomial  in  low-pass
#-             smoothing.  The value is 0 or 1.
#-     nsjump  input,  skipping  value  for  seasonal  smoothing.    The
#-             seasonal  smoother  skips  ahead  nsjump  points and then
#-             linearly interpolates in between.  The  value  of  nsjump
#-             should  be  a  positive  integer; if nsjump=1, a seasonal
#-             smooth is calculated  at  all  n  points.   To  make  the
#-             procedure  run  faster, a reasonable choice for nsjump is
#-             10%-20% of ns.
#-     ntjump  input, skipping value for trend smoothing.
#-     nljump  input, skipping value for the low-pass filter.
#-     ni      input, number of loops  for  updating  the  seasonal  and
#-             trend  components.   The value of ni should be a positive
#-             integer.  See the next argument for advice on the  choice
#-             of ni.
#-     no      input, number of iterations of robust fitting.  The value
#-             of  no  should be a nonnegative integer.  If the data are
#-             well behaved without outliers, then robustness iterations
#-             are not needed.  In this case set no=0, and set ni=2 to 5
#-             depending  on  how  much  security  you  want  that   the
#-             seasonal-trend   looping   converges.   If  outliers  are
#-             present then no=3 is  a  very  secure  value  unless  the
#-             outliers are radical, in which case no=5 or even 10 might
#-             be better.  If no>0 then set ni to 1 or 2.
#-     rw      output, final robustness weights. All rw are 1 if no=0.
#-     season  output, seasonal component.
#-     trend   output, trend component.
#-     work    workspace of (n+2*np)*5 locations.

def stl(x, period, s_window, s_degree=0,
        t_window=None, t_degree=1,
        l_window=None, l_degree=None,
        s_jump=None, t_jump=None,
        l_jump=None, robust=False,
        inner=2, outer=0, na_action='error'):

#stl <- function(x, s.window,
#		s.degree = 0,
#		t.window = NULL, t.degree = 1,
#		l.window = nextodd(period), l.degree = t.degree,
#		s.jump = ceiling(s.window/10),
#		t.jump = ceiling(t.window/10),
#		l.jump = ceiling(l.window/10),
#		robust = FALSE,
#		inner = if(robust)  1 else 2,
#		outer = if(robust) 15 else 0,
#		na.action = na.fail)
    
    n = len(x)
    
    if (period < 2) or (n <= 2 * period):
        raise ValueError("series is not periodic or has less than two periods")
        
    periodic = False
    
    if s_window == 'periodic':
        periodic = True
        s_window = 10*n + 1
        s_degree = 0
        
    if t_window is None:
        t_window = nextodd(math.ceil(1.5 * period / (1 - 1.5/s_window)))
    
    if l_window is None:
        l_window = nextodd(period)
        
    if l_degree is None:
        l_degree = t_degree
        
    if s_jump is None:
        s_jump = int(math.ceil(s_window/10))
        
    if t_jump is None:
        t_jump = int(math.ceil(t_window/10))
    
    if l_jump is None:
        l_jump = int(math.ceil(l_window/10))
    
    if robust:
        inner = 1
        outer = 15
        
    if s_degree not in [0, 1]:
        raise ValueError("s_degree must be 0 or 1")
        
    if t_degree not in [0, 1]:
        raise ValueError("t_degree must be 0 or 1")
        
    if l_degree not in [0, 1]:
        raise ValueError("l_degree must be 0 or 1") 
        
#    z <- .Fortran(C_stl, x, n,
#		  as.integer(period),
#		  as.integer(s.window),
#		  as.integer(t.window),
#		  as.integer(l.window),
#		  s.degree, t.degree, l.degree,
#		  nsjump = as.integer(s.jump),
#		  ntjump = as.integer(t.jump),
#		  nljump = as.integer(l.jump),
#		  ni = as.integer(inner),
#		  no = as.integer(outer),
#		  weights = double(n),
#		  seasonal = double(n),
#		  trend = double(n),
#		  double((n+2*period)*5))
    
    weights = np.zeros((n, ), dtype='float')
    seasonal = np.zeros((n, ), dtype='float')
    trend = np.zeros((n, ), dtype='float')
    work = np.zeros((n + 2*period, 5), dtype='float')
    
#    print('x shape', x.shape)
#    print('n', n)
#    print('period', period)
#    print('s_window', s_window)
#    print('t_window', t_window)
#    print('l_window', l_window)
#    print('s_degree', s_degree)
#    print('t_degree', t_degree)
#    print('l_degree', l_degree)
#    print('s_jump', s_jump)
#    print('t_jump', t_jump)
#    print('l_jump', l_jump)
#    print('inner', inner)
#    print('outer', outer)
    
    stl_func(x, n, period,
             s_window, t_window, l_window,
             s_degree, t_degree, l_degree,
             s_jump, t_jump, l_jump,
             inner, outer,
             weights, seasonal, trend, work)
       
#    if(periodic) {
#	## make seasonal part exactly periodic
#	which.cycle <- cycle(x)
#	z$seasonal <- tapply(z$seasonal, which.cycle, mean)[which.cycle]
#    }

#myts <- ts(c(0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3), frequency=5)
#
#which.cycle <- cycle(myts)
#
#seasonal <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.11, 0.22, 0.33, 0.44, 0.55, 0.12, 0.23)
#
#seasonal <- tapply(seasonal, which.cycle, mean)[which.cycle]
#
#print(seasonal)
#
#        1         2         3         4         5         1         2         3
#0.1100000 0.2166667 0.3150000 0.4200000 0.5250000 0.1100000 0.2166667 0.3150000
#        4         5         1         2
#0.4200000 0.5250000 0.1100000 0.2166667

#seasonal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.11, 
#                     0.22, 0.33, 0.44, 0.55, 0.12, 0.23])
#period = 5
#n = len(seasonal)
#
#for p in range(0, period):
#    period_mask = [i%period == p for i in range(0, n)]
#    seasonal_periodic = np.mean(np.compress(period_mask, seasonal))
#    np.copyto(seasonal, seasonal_periodic, where=period_mask)

    if periodic:
        # make seasonal part exactly periodic
        for p in range(0, period):
            period_mask = [i%period == p for i in range(0, n)]
            seasonal_periodic = np.mean(np.compress(period_mask, seasonal))
            np.copyto(seasonal, seasonal_periodic, where=period_mask)

    remainder = x - seasonal - trend
    
    return seasonal, trend, remainder


             
             
    
    
    