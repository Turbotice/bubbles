# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Alienor

Package to analyse the data from files stats.dat
Include the following functions:
    -import_one_file(dir)
    -import_from_folder(dir, rescale_data = False, rescale_time = False, R0 = 8.)
    -Weber(data,R0=8.0,sigma=18.0,Ceps=2)
    -Taylor_length(data)

"""

import numpy as np
import glob
from . import dropbub as db

def import_one_file(dir, rescale_data = False, rescale_time = False, shift_time = False):
    '''
    Download data from file stats.dat given the directory in the form 'directory/'
    
    Optionnal arguments: 
        -rescale_data = False : If True, divide all the data by its initial value
        -rescale_time = False : If True, divide time by the turbulent time given by R0^(2/3)*eps^(-1/3)

    Return an 2d array of the form:
    time dissipation kinetic_energy Reynolds
    
    '''

    #try:    
        #data = np.loadtxt(dir +"stats.dat")
    #except:
    data = np.genfromtxt(dir + 'stats.dat', skip_header=1, skip_footer=1)
    
    if shift_time:
        data[:,0] -= data[0,0]
    
    if rescale_time:
        bubbles = db.import_bubbles(dir+'/')
        R0 = (3/(4*np.pi)*np.max(bubbles[:][3]))**(1/3)
        data[:,0] /= R0**(2/3)*data[0,1]**(-1/3)
    if rescale_data:
        data[:,1:] /= data[0,1:] 
    
    return data

def import_from_folder(dir, rescale_data = False, rescale_time = False):
    '''
    Take in argument the direction of a directory where are all the results from the simulations.
    
    Optionnal arguments: 
        -rescale_data = False : If True, divide all the data by its initial value
        -rescale_time = False : If True, divide time by the turbulent time given by R0^(2/3)*eps^(-1/3)
    
    Return names,data
    -names: a list of the names of all the folders analysed
    -data: a list of all the data
    data[i]: 2d array given by the function import_one_file
    '''
    folders = sorted(glob.glob(dir+'*'))
    n = len(dir)
    data, names = [], []
    for i in range(len(folders)):
        try:
            data.append(import_one_file(folders[i], rescale_data, rescale_time))
            names.append(folders[i][n:])
        except:
            pass
        
    return names, data

def Weber(data,R0=8.0,sigma=18.0,Ceps=2):
    '''
    Takes in argument the data of one experiment given by import_one_file or import_from_folders()[i]
    Optionnal arguments:
        -Ro = 8.0 the radius of the initial bubble
        -Sigma = 18.0 the surface tension
        -Ceps = 2  the numerical coefficient that turbulent dissipation rate the mean root squared of the velocity
    
    Return a 1d array containing the Weber number for every time step.
    '''
    We = data[:,1]**(2/3)*(2*R0)**(5/3)/(sigma*Ceps**(2/3))
    return We

def Taylor_length(data):
    '''
    Takes in argument the data of one experiment given by import_one_file or import_from_folders()[i]
    
    Return a 1d array containing the Taylor microscale for every time step.
    '''
    
    Taylor = np.sqrt(200/3)*data[:,2]**(3/2)/(data[:,1]*data[:,3])
    return Taylor

