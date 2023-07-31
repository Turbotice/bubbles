% Calcul des gradients pour le paramètres R0


import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.optimize import fsolve
import scipy
import time
import fonction_bulle.py

L = 120 #box size
dx = 120/2**9 #minimum grid size
print(dx)
folderfig='/home/turbots/Documents/final_fig/R0/grad/'
eps = 7.8 #mean dissipation without the bubble
rho  = 1 #fluid density (taken as reference)
sig=dx/2

folds = glob.glob('Re*') # we pick up all the files that start with Re and put them in folds
len(folds)# lenght of folds

folds0 = 'Re55t140-l9-16-2540-2-dump_63405'
filesdissint = glob.glob(folds0+ '/localstats_[!f]*')
filesdissext = glob.glob(folds0 + '/localstats_fluid*')
filesinterf = glob.glob(folds0 + '/inter*.dat')
        
    

import warnings

with warnings.catch_warnings():#to remove warnings associated to the fact that some files are empty.
    warnings.simplefilter("ignore") 
#     dataint = np.zeros([0, 7], dtype=float)
#     dataext = np.zeros([0, 6], dtype=float)
    interf = np.zeros([0, 10], dtype=float)
    bubble = np.zeros([0, 12], dtype=float)
    stat = np.zeros([0, 8], dtype=float)
    vit_in = np.zeros([0,17], dtype=float)
    vit_interf = np.zeros([0,17], dtype=float)
    vit_out = np.zeros([0,16], dtype=float)

    for file in filesinterf:
        datatemp = np.loadtxt(file, skiprows=1)
        if datatemp.shape !=(0,):
            interf = np.vstack((interf, datatemp))

try:
    datatemp = np.loadtxt(folds0 +'/bubbles.dat', skiprows=1)
except:
    datatemp = np.loadtxt(folds0 +'/bubbles.dat', skiprows=0)
if datatemp.shape !=(0,):
    bubble = np.vstack((bubble,datatemp))

    
datatemp=np.loadtxt(folds0 + '/vitessein_0.dat',skiprows=1)
if datatemp.shape !=(0,):
        vit_in = np.vstack((vit_in,datatemp))

datatemp=np.loadtxt(folds0 + '/vitesseinterf_0.dat',skiprows=1)
if datatemp.shape !=(0,):
        vit_interf = np.vstack((vit_interf,datatemp))

datatemp=np.loadtxt(folds0 + '/vitesseout_0.dat',skiprows=1)
if datatemp.shape !=(0,):
        vit_out = np.vstack((vit_out,datatemp))
        

R0= float(folds0.split('-')[2])

mask_vit_in=(vit_in[:,0]<600)
maskinterf=(interf[:,0]<600)

print(30*dx)
for bulle in bubble:
    if bulle[3]<30*dx: #la bulle est plus petite que 30 cellules ie pas une "bulle"
        j=bulle[2]
        mask_vit_in=np.logical_and(mask_vit_in,vit_in[:,0][:]!=j)
        maskinterf=np.logical_and(maskinterf,interf[:,0][:]!=j)





