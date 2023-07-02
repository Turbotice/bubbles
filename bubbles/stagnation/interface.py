#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:58:43 2022

@author: riviere
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.special import sph_harm

def clean_interface(x, y, window=3, degree=1):
    #not yet mono-valued
    nx = np.unique(x)#also sort x values by increasing order
    ny = np.zeros_like(nx)
    for i in range(len(nx)):
        ny[i] = np.mean(y[np.where(x==nx[i])])
    ny = savgol_filter(ny, window, degree)
    return nx, ny

def Yl0(l, theta):
  '''
  Spherical decomposition associated to the principal number l
  and secondary number m=0.
  '''
  return np.real(sph_harm(0, l, 0, theta))
