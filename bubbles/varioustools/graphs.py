#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:10:21 2020

@author: alienor

Plot functions and preferences
"""
import numpy as np
from matplotlib import rc

rc('font',family='serif', size='15')
rc('lines', linewidth=2)
rc('legend', frameon=False)
rc('axes', linewidth=1, labelsize='17')
rc('mathtext', rm = 'serif', fontset='cm')

def align_yaxis(axes):
    y_lims = np.array([ax.get_ylim() for ax in axes])

    # force 0 to appear on all axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize all axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lims = y_new_lims_normalized * y_mags
    for i, ax in enumerate(axes):
        ax.set_ylim(new_lims[i])
