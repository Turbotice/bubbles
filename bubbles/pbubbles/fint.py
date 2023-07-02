#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:04:19 2019

@author: Alienor
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from matplotlib import animation
import glob



def extract_data(file):
    data = np.array([line.split() for line in open(file)])
    for line in data:
        for i in range(len(line)):
            line[i] = float(line[i])
            
    return data

def interface(data):
    i = 0
    t = []
    Bulles = []
    while i<len(data):
        t.append(data[i][0])
        
        if len(data[i+1])!=0:
            i = i+1
            Volume = []
            while i<len(data) and len(data[i])!=1:
                j = i
                Face = []
                while len(data[j])!=0:
                    Face.append(data[j])
                    j = j+1            
                Volume.append(Face)
                i = j+2    
            Bulles.append(Volume)
        else:
            i += 2
            Bulles.append([])
    return t, Bulles

#To much data=>impossible to have all of them in memory at the same time
#def extract_all_data(dir):
#        names = sorted(glob.glob(dir+'/fint*.dat'))
#        data0 = extract_data(names[0])
#        t0,Bulles0 = interface(data0)
#        all_data = np.array(Bulles0)
#        
#        for name in names[1:]:
#            data = extract_data(name)
#            t, Bulles = interface(data)
#            all_data += np.array(Bulles)
#            print('{} OK'.format(name))
#        return t,all_data
            
def interface_i(dir,i):
    names = sorted(glob.glob(dir+'/fint*.dat'))
    data0 = extract_data(names[0])
    t0,Bulles0 = interface(data0)
    Interface = Bulles0[i]
    
    for name in names[1:]:
        data = extract_data(name)
        t, Bulles = interface(data)
        Interface += Bulles[i]
        print('{} OK'.format(name))
    
    return t0[i], Interface


def ind_at_time(t):
    data0 = extract_data(dir+'fint_0.dat')
    t0, Bulles0 = interface(data0)
    i = 0
    while i<len(t0) and t0[i]<t:
        i += 1
    return i





####################################
###########Analysis#################    
####################################
dir = 'bubbles_sigma18/turb_810/'

t = 160  #Determine this time by analysing the fi files
ind = ind_at_time(t)

#t,Interface = interface_i(dir,ind) #Put in comment when data is download because takes a long time
SHOW_VIEW = True
SHOW_ANIMATION = False





#####################################
#########Representation##############
#####################################

if SHOW_VIEW:
    
    fig = plt.figure ()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,120)
    ax.set_ylim(0,120)
    ax.set_zlim(0,120)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_title('t = {}'.format(t),loc='left')
    ax.add_collection3d (Poly3DCollection (Interface, facecolor=None, edgecolor = 'b', linewidths=1, alpha=1))
    ax.add_collection3d(Line3DCollection(Interface, colors='b', linewidths=0.5, linestyles=':'))


if SHOW_ANIMATION:
 
    fig = plt.figure('Animation')
    ax = fig.add_subplot(111, projection='3d')
    
    ax.grid()
    
    ind_init = 0
    ind_final = len(t)-1
    pas = 10
    Number_frames = (ind_final-ind_init)//pas
    
    
    def animate(i):
        """perform animation step"""   
        ax.clear()
        ax.set_xlim(0,120)
        ax.set_ylim(0,120)
        ax.set_zlim(0,120)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.add_collection3d (Poly3DCollection (Bulles[pas*i], facecolor=None, edgecolor = 'b', linewidths=1, alpha=1))
        ax.add_collection3d(Line3DCollection(Bulles[pas*i], colors='b', linewidths=0.25, linestyles=':'))
        ax.set_title('t = {}'.format(t[pas*i]),loc='left')
        return ax
    
    ani = animation.FuncAnimation(fig, animate, frames=Number_frames)
