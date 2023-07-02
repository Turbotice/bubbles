#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:03:40 2023

@author: riviere
"""
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt

folder = 'Re55t200-l9-400/'

pos = ['x', 'y', 'z']
nfiles = ['fi' + coord + '_*.dat' for coord in pos ]
print(nfiles)

n = len(glob.glob(folder + nfiles[0]))
print(n)