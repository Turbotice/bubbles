#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:56:24 2022

@author: riviere

Useful functions and parameters for the stagnation point's study,
"""
import numpy as np

def We(gam, rho=1, d=2, k=1):
    return rho*k**2*d**3/gam

def Re(k=1, d=2, nu=0.01):
    return k*d**2/nu

def Ca(gam, rho=1, d=2, k=1, nu=0.01):
    #We/Re
    return k*d*nu*rho/gam

def omega(l, gamma, r=1, rho=1):
    return np.sqrt((l-1)*(l+1)*(l+2)*gamma/(rho*r**3))

def Oh(gamma, rho=1, d=2, mu=0.01):
    return mu/np.sqrt(rho*gamma*d)


rho = 1
d = 2
r = d/2.
nu = 0.01