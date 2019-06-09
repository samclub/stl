#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:57:23 2019

@author: vaibhav
"""

cimport cython
from cython cimport view
import numpy as np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def stl_func(double [:] y, int n, int np, int ns, int nt,
        int nl, int isdeg, int itdeg,
        int ildeg, int nsjump, int ntjump,
        int nljump, int ni, int no,
        double [:] rw, double [:] season,
        double [:] trend, double [:, :] work):
    
    cdef int i, k, newns, newnt, newnl, newnp
    cdef int userw
    
    userw = 0
    
    for i in range(1, n+1):
        trend[i-1] = 0.
             
    # the three spans must be at least three and odd
    newns = max(3, ns)
    newnt = max(3, nt)
    newnl = max(3, nl)
    
    if newns%2 == 0:
        newns += 1
        
    if newnt%2 == 0:
        newnt += 1
        
    if newnl%2 == 0:
        newnl += 1
        
    # periodicity at least 2:
    newnp = max(2, np)

    #print('y shape', y.shape)
    #print('n', n)
    #print('period', newnp)
    #print('s_window', newns)
    #print('t_window', newnt)
    #print('l_window', newnl)
    #print('s_degree', isdeg)
    #print('t_degree', itdeg)
    #print('l_degree', ildeg)
    #print('s_jump', nsjump)
    #print('t_jump', ntjump)
    #print('l_jump', nljump)
    #print('inner', ni)
    #print('userw', userw)
    #print('rw shape', rw.shape)
    #print('season shape', season.shape)
    #print('trend shape', trend.shape)
    #print('work shape', work.shape)
    
    # outer loop -- robustnes iterations
    stlstp(y, n, newnp, newns, newnt, newnl,
           isdeg, itdeg, ildeg, nsjump,
           ntjump, nljump, ni, userw, rw,
           season, trend, work)
    
    for k in range(0, no):
        
        for i in range(1, n+1):
            work[i-1, 0] = trend[i-1] + season[i-1]
            
        stlrwt(y, n, work[0:n, 0], rw)
        userw = 1
        
        stlstp(y, n, newnp, newns, newnt, newnl,
               isdeg, itdeg, ildeg, nsjump,
               ntjump, nljump, ni, userw, rw,
               season, trend, work)

    # robustness weights when there were no robustness iterations:
    if no <= 0:
        for i in range(1, n+1):
            rw[i-1] = 1.
                  
    return

def stless(double [:] y, int n, int len_, 
           int ideg, int njump, int userw, 
           double [:] rw, double [:] ys, double [:] res):
    
    cdef int newnj, nleft, nright, nsh, k, i, j
    cdef double delta, ys_
    cdef int ok

    ok = 0
    
    if n < 2:
        ys[0] = y[0]
        return
    
    newnj = min(njump, n - 1)
    if len_ >= n:
        nleft = 1
        nright = n
        for i in range(1, n+1, newnj):
            ok, ys_ = stlest(y, n, len_, ideg, 
                             <double>i, ys[i-1],
                             nleft, nright, 
                             res, userw,
                             rw)

            if ok == 1:
                ys[i-1] = ys_
            
            if ok == 0:
                ys[i-1] = y[i-1]
    else:
        if newnj == 1:
            nsh = (len_ + 1)/2
            nleft = 1
            nright = len_
            for i in range(1, n+1):
                if (i > nsh) and (nright != n):
                    nleft += 1
                    nright += 1
                ok, ys_ = stlest(y, n, len_, ideg,
                       <double>i, ys[i-1],
                       nleft, nright,
                       res, userw,
                       rw)
                if ok == 1:
                    ys[i-1] = ys_
                if ok == 0:
                    ys[i-1] = y[i-1]
        else:
            nsh = (len_ + 1)/2
            for i in range(1, n+1, newnj):
                if i < nsh:
                    nleft = 1
                    nright = len_
                elif i >= (n - nsh + 1):
                    nleft = n - len_ + 1
                    nright = n
                else:
                    nleft = i - nsh + 1
                    nright = len_ + i - nsh
                    
                ok, ys_ = stlest(y, n, len_, ideg,
                       <double>i, ys[i-1],
                       nleft, nright,
                       res, userw,
                       rw)

                if ok == 1:
                    ys[i-1] = ys_
                
                if ok == 0:
                    ys[i-1] = y[i-1]
                    
    if newnj != 1:
        for i in range(1, n-newnj+1, newnj):
            delta = (ys[i+newnj-1] - ys[i-1])/newnj
            for j in range(i+1, i+newnj):
                ys[j-1] = ys[i-1] + delta*<double>(j-i)
                
        k = int(((n - 1)//newnj)*newnj + 1)
            
        if k != n:
            ok, ys_ = stlest(y, n, len_, ideg,
                   <double>n, ys[n-1],
                   nleft, nright,
                   res, userw,
                   rw)

            if ok == 1:
                ys[n-1] = ys_
            
            if ok == 0:
                ys[n-1] = y[n-1]
                
            if k != (n - 1):
                delta = (ys[n-1] - ys[k-1])/<double>(n - k)
                for j in range(k + 1, n - 1 + 1):
                    ys[j-1] = ys[k-1] + delta*<double>(j - k)
                    
    return

def stlest(double [:] y, int n, int len_,
           int ideg, double xs, double ys,
           int nleft, int nright, double [:] w,
           int userw, double [:] rw):
    
    cdef double range_, h, h1, h9, a, b, c, r
    cdef int j, ok
    
    range_ = <double>n - <double>1
    
    h = max(xs - <double>nleft, <double>nright - xs)
    if len_ > n:
        h += (len_ - n)//2
             
    h9 = 0.999*h
    h1 = 0.001*h
    a = 0.
    
    for j in range(nleft, nright+1):
        r = abs(<double>j - xs)
        if r <= h9:
            if r <= h1:
                w[j-1] = 1.
            else:
                w[j-1] = (1. - (r/h)**3)**3
            
            if userw == 1:
                w[j-1] = rw[j-1]*w[j-1]
            
            a += w[j-1]
        else:
            w[j-1] = 0.
             
    if a <= 0.:
        ok = 0
    else:
        ok = 1
        for j in range(nleft, nright+1):
            w[j-1] = w[j-1]/a

        if (h > 0.) and (ideg > 0):
            a = 0.
            for j in range(nleft, nright+1):
                a += w[j-1]*<double>j
            b = xs - a
            c = 0.
            for j in range(nleft, nright+1):
                c += w[j-1]*(<double>j - a)**2
            if np.sqrt(c) > 0.001*range_:
                b = b/c
                for j in range(nleft, nright+1):
                    w[j-1] = w[j-1]*(b*(<double>j - a) + 1.)
        
        ys = 0.
        for j in range(nleft, nright+1):
            ys += w[j-1]*y[j-1]
            
    return ok, ys

def stlfts(double [:] x, int n, int np,
           double [:] trend, double [:] work):
    
    stlma(x, n, np, trend)
    stlma(trend, n-np+1, np, work)
    stlma(work, n-2*np+2, 3, trend)
    
    return

def stlma(double [:] x, int n, int len_, double [:] ave):
    
    cdef double flen, v
    cdef int k, m, newn
    cdef Py_ssize_t i, j
    
    newn = n - len_ + 1
    flen = <double>len_
    v = 0.
    
    for i in range(1, len_+1):
        v += x[i-1]
        
    ave[0] = v/flen
    if newn > 1:
        k = len_
        m = 0
        for j in range(2, newn+1):
            k += 1
            m += 1
            v += (x[k-1] - x[m-1])
            ave[j-1] = v/flen
               
    return

def stlstp(double [:] y, int n, int np, int ns,
           int nt, int nl, int isdeg,
           int itdeg, int ildeg, int nsjump,
           int ntjump, int nljump, int ni,
           int userw, double [:] rw,
           double [:] season, double [:] trend,
           double [:, :] work):

    cdef int i, j

    for j in range(1, ni+1):
        for i in range(1, n+1):
            work[i-1, 0] = y[i-1] - trend[i-1]
            
        stlss(work[:, 0], n, np, ns, isdeg, nsjump, userw, rw, work[:, 1],
              work[:, 2], work[:, 3], work[:, 4], season)
        stlfts(work[:, 1], n+2*np, np, work[:, 2], work[:, 0])
        stless(work[:, 2], n, nl, ildeg, nljump, 0, 
               work[:, 3], work[:, 0], work[:, 4])
        
        for i in range(1, n+1):
            season[i-1] = work[np+i-1, 1] - work[i-1, 0]
            
        for i in range(1, n+1):
            work[i-1, 0] = y[i-1] - season[i-1]
            
        stless(work[:, 0], n, nt, itdeg, ntjump, userw,
               rw, trend, work[:, 2])
        
    return

def stlrwt(double [:] y, int n, double [:] fit, double [:] rw):
    
    cdef int i
    cdef double cmad, c9, c1, r

    for i in range(1, n+1):
        rw[i-1] = abs(y[i-1] - fit[i-1])
        
    cmad = 6.*np.median(rw)
    
    c9 = 0.999*cmad
    c1 = 0.001*cmad
    
    for i in range(1, n+1):
        r = abs(y[i-1] - fit[i-1])
        if r <= c1:
            rw[i-1] = 1.
        elif r <= c9:
            rw[i-1] = (1. - (r/cmad)**2)**2
        else:
            rw[i-1] = 0.
    
    return

def stlss(double [:] y, int n, int np,
          int ns, int isdeg, int nsjump, int userw,
          double [:] rw, double [:] season,
          double [:] work1, double [:] work2,
          double [:] work3, double [:] work4):
    
    cdef int nright, nleft, i, j, k, m
    cdef int ok
    cdef double xs, ys_

    ok = 0

    if np < 1:
        return
    
    for j in range(1, np+1):
        k = (n - j)//np + 1
        for i in range(1, k+1):
            work1[i-1] = y[(i - 1)*np + j - 1]
            
        if userw == 1:
            for i in range(1, k+1):
                work3[i-1] = rw[(i - 1)*np + j -1]
                
        stless(work1, k, ns, isdeg, nsjump, userw, work3, work2[1:], work4)
        xs = 0
        nright = min(ns, k)
        ok, ys_ = stlest(work1, k, ns, isdeg, xs, work2[0], 1, nright, work4,
               userw, work3)

        if ok == 1:
            work2[0] = ys_
        
        if ok == 0:
            work2[0] = work2[1]
            
        xs = k + 1
        nleft = max(1, k - ns + 1)
        ok, ys_ = stlest(work1, k, ns, isdeg, xs, work2[k + 1], nleft, k,
               work4, userw, work3)

        if ok == 1:
            work2[k + 1] = ys_
        
        if ok == 0:
            work2[k + 1] = work2[k]
            
        for m in range(1, k+2+1):
            season[(m - 1)*np + j - 1] = work2[m - 1]
            
    return