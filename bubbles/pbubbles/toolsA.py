#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:46:14 2019

@author: Alienor

This file contains functions that can be used in any context:
    -cartesian_to_polar(t,x,y)
    -cartesian_to_polar2(t,x,y)
    -cartesian_to_spherical(t,x,y,z)
    -polar(x,y)
    -spherical(x,y,z)
    -moyenne_glissante(liste)
    -norm(point1,point2)
    -cossin2ampphi(a,b)
    -loadtoarrayoflist(file_name)
    -save_dictionary(dictionary, name)
    -load_dictionary

"""
import pickle
import numpy as np


def cartesian_to_polar(t,x,y): 
    '''
    Takes in argument a np array of the values of a field in cartesian 
    coordinates and computes them in polar coordinates.
    t[i,j] i=>r; j=>theta
    '''
    r, theta = polar(x,y)
    values_r = np.unique(r)
    values_theta = np.unique(theta)

    t_polar = np.NaN*np.ones([len(values_r), len(values_theta)])
    
    for i in range(len(values_r)):
        coordinate = np.where(r == values_r[i])

        for k in range(len(coordinate[0])):
            #print(k)
            j = list(values_theta).index(theta[coordinate[0][k], coordinate[1][k]])
            #print(i,j)
            t_polar[i,j] = t[coordinate[0][k],coordinate[1][k]]
            
    return t_polar, values_r, values_theta

def cartesian_to_polar2(t,x,y):
    '''
    Given an array of the value of a scalar field in 2D, on a cartesian grid,
    return the list of its values on polar coordinate.
    
    t is an array of the values of the field on a grid (x,y)
    x, y are 2 one-dimensionnal arrays given the values of x and y
    
    Return:
        -t_polar, a list such that np.shape(t_polar)=(len(x)*len(y),3)
        t_polar[0]: value of t at coordinates r=t_polar[1], theta = t_polar[2]
        -values_r: 1D array, all the values taken by r
        -values_theta: 1D array, all the values taken by theta
    '''
    r, theta = polar(x,y)
    values_r = np.unique(r)
    values_theta = np.unique(theta)
    
    t_polar = []
    for i in range (np.shape(t)[0]):
        for j in range(np.shape(t)[1]):
            t_polar.append([t[i,j], r[i,j], theta[i,j]])
    return t_polar, values_r, values_theta
    


def cartesian_to_spherical(t,x,y,z):
    r, theta, phi = spherical(x, y, z)
    values_r = np.unique(r)
    values_theta = np.unique(theta)
    values_phi = np.unique(phi)
    
    t_spherical = []
    for i in range (np.shape(t)[0]):
        for j in range(np.shape(t)[1]):
            for k in range(np.shape(t)[2]):
                t_spherical.append([t[i,j,k],r[i,j,k], theta[i,j,k], phi[i,j,k]])
    return t_spherical, values_r, values_theta, values_phi

def polar(x,y):
    return np.sqrt(x**2+y**2), np.arctan2(y,x)

def spherical(x,y,z):
    '''
    Compute the spherical coordinate from the cartesian coordinates 
    using the physical conventions (theta is the colatitude and phi the longitude).
    '''
    r = np.sqrt(x**2+ y**2 + z**2)
    theta = np.pi/2 - np.arctan2(z,np.sqrt(x**2+y**2))
    phi = np.arctan2(y,x)
    return r, theta, phi

def moyenne_glissante(liste):
    '''
    Average the value of liste to smooth the data.
    
    '''
    l = np.zeros(len(liste))
    l[0], l[-1] = liste[0], liste[-1]
    if type(liste) == type(np.array([])):
        l[1:-1] = (liste[1:-1] + liste[:-2] + liste[2:])/3
        return l
    else:
        listeI = np.array(liste)
        l[1:-1] = (listeI[1:-1] + listeI[:-2] + listeI[2:])/3
        return list(listeI)
#    l[]
#    for i in range(1,len(liste)-1):
#        l.append((liste[i-1]+liste[i]+liste[i+1])/3)
#    l.append(liste[-1])
#    return l

def norm(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2+(point1[2])-point2[2])

def cossin2ampphi(a,b):
    '''
    Given the coefficient a and b in acos(x)+bsin(x) = c cos(x+phi) return c and phi
    
    '''
    
    phi = np.arctan2(b,a)
    
    c = np.sqrt(a**2+b**2)
    
    if abs(phi)>=3*np.pi/4 or abs(phi)<np.pi/4:
        return c*np.sign(np.cos(phi)*a), phi
    elif abs(phi)>=np.pi/4 and abs(phi)<3*np.pi/4:
        return c*np.sign(np.sin(phi)*b), phi

def loadtoarrayoflist(file_name, skip_header=0):
    '''
    To load data from a text file when the length of the lines is not constant.
    Returns an array containing the list of the lines.
    
    Skip_header is the number of lines at the beginning that should be skiped.
    '''
    
    file = open(file_name, 'r')
    lines = file.readlines()[skip_header:]
    data = np.zeros(len(lines), dtype=object)
    
    for i in range(len(lines)):
        
        line = (lines[i].split('\n')[0]).split(',')
        data[i] = [float(line[0])]
        for j in range(1,len(line)):
            data[i].append(float(line[j]))
    
    file.close()
    
    return data
            
    

def save_dictionary(dictionary, name):
    with open(name +'.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    
def load_dictionary(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
