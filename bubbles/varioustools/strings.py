#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:54:02 2020

@author: alienor
"""

import numpy as np

def sort_names(list_names):
    lens = np.array([len(list_names[i]) for i in range(len(list_names))])
    files_name = []
    for n in np.unique(lens):
        l = list(list_names[np.where(lens==n)])
        l.sort()
        files_name += l
    return files_name
