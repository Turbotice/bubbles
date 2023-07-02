#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:33:26 2019

@author: Alienor

Package to analyse the data from the files droplets.dat and bubbles.dat
    
Include several functions:
    #To import the data:
    -import_particles(dir)
    -import_bubbles(dir)
    -import_droplets(dir)
    
    #Basic extractions functions:
    -number(data)
    -volumes(data,level=7, rescale_by_Vinit=False)
    -positions(data,level=11)
    -radius(data,level=7,rescale_by_Rinit=False)
    
    #Break-up analysis
    -breakup_indices(radius)
    -breakup(radius,time, ind_breakup)
    
    #find the events
    -distance(L1, L2, index_m, index_f1, index_f2)
    -distance2(L1, L2, index_f)
    -determine_event(L1, L2)
    -find_events(volume_bubbles)
    -isbreakup(event)
    -update_daughters(daughters, event)
    -find_events_given_dt(events, time, dt)
    -add_value(t, ind, value)
    -remove_value(t, ind, fillwith=0)
    
    #extract the info for each break-up event (break-up only)
    -volume_mothers(events, volume_bubbles)
    -volume_daughters(events, volume_bubbles)
    -number_daughters(events)
    -weber_mothers(events, volume_bubbles, gamma, eps)
    
    #life time
    -life_time(events, time, vol_bubbles)
    
    #time interval between events
    -time_intervals(events, time, vol_bubbles)
    
    #models for the size distribution
    Martinez-Bazan
    -fD_MB(D, Lambda)
    -fV_MB(V, Lambda)
    #Tsouris & Tavlarides adapted
    -fD_TT(D, D_min)
    -fV_TT(V, V_min)
    
"""

import numpy as np
import scipy.integrate as integrate
from . import variables as va



def import_particles(dir):
    '''
    To import the files bubbles.dat or droplets.dat.
    Takes in argument the name of the file.
    '''
    data = np.genfromtxt(dir, skip_footer=1) 
    #Remove the last time not to have problems with the unfinished simulations
    i = -1
    while data[i, 0] == data[i-1, 0]:
        i -= 1
    return data[:i, :]


def import_bubbles(dir):
    '''
    Take une argument the direction where are the data in the form 'folder/'
    Return the data from the file bubbles.dat
    
    Return an np array of the form:
    step time indiceofthebubble volume x y z
    
    '''
    return import_particles(dir + 'bubbles.dat')

    
def import_droplets(dir):
    '''
    Take une argument the direction where are the data in the form 'folder/'
    Return the data from the file droplets.dat
    
    Return an np array of the form:
    step time indiceofthedroplet volume x y z
    
    '''
    return import_particles(dir + 'droplets.dat')   



def number(data):
    """
    Computes the number of 'particles' as a function of time.
    
    Takes in argument the file droplets.dat or bubbles.dat
    
    Returns two one-dimensionnal arrays : time, number
    
    """
    nbr = np.zeros(int(data[-1, 0] - data[0, 0]) + 1)
    time = np.zeros(int(data[-1, 0] - data[0, 0]) + 1)
    for i in range(len(data)):
        time[int(data[i, 0]- data[0, 0])] = data[i, 1]
        nbr[int(data[i, 0]-data[0, 0])] = max(data[i,2]+1, nbr[int(data[i, 0]-data[0, 0])])
    return time, nbr


def positions(data,level=11):
    
    '''
    Computes the position of each particule as a function of time.
    Take in argument the data extracted by the function import_droplets or import_bubbles and the level used for the simulation
    If the level is not given it is fixed to 11.
    
    Returns 2 arrays: time, position
    
    time: one dimensionnal array of size the number of steps
    np.shape(position) = (number of time steps,number of particles,3)
    position[i,j,k]: the coordinate k of the particule j at step i
    
    All particles with a size inferior to the volume of a cell (120/2**level)**3
    are automatically supressed.
    '''
    
    position = np.zeros([int(data[-1, 0]-data[0, 0])+1, int(max(data[:, 2]))+1, 3])
    time = np.zeros(int(data[-1, 0]-data[0, 0])+1)
    
    i = 0
    t = i
    Vcell = (120/2**level)**3

    while i<len(data)-1:
        vol = []
        if data[i,3]>Vcell:
            vol.append(data[i, 3:])
            
        time[t] = data[i,1]
        while i<len(data)-1 and data[i, 0]==data[i+1, 0]:
            i += 1
            #check that the bubble has a volume superior than the volume of a cell
            if data[i,3]>Vcell:
                vol.append(data[i,3:])
                
        vol = np.sort(vol,0) #order by increasing volume
        
        for j in range(len(vol)):
            position[int(data[i, 0]-data[0, 0]), -j-1, 0] = vol[-j-1, 1] #x
            position[int(data[i, 0]-data[0, 0]), -j-1, 1] = vol[-j-1, 2] #y
            position[int(data[i, 0]-data[0, 0]), -j-1, 2] = vol[-j-1, 3] #z
        i += 1
        t += 1
    #suppress the false particles        
    i = 0
    while i <len(position[0]) and np.max(position[:, i,:])<120/2**(level-3):
        i += 1
    position = position[:, i:]
    return time[:-1], position[:-1]



def volumes(data,level=7, rescale_by_Vinit=False):

    """
    Computes the volume of each particle as a function of time.
    
    Take in argument the data extracted with the function import_droplets or 
    import_bubbles and the level used.
    If the level is not given it is fixed to 12.
    
    Returns two arrays: time, volume
    
    time: one dimensionnal array
    np.shape(volume) = (number of time steps, number of particles)
    volume[i,j]: the volume of the particule j at time i

    When a bubble has not been created yet its volume is set to 0.     
    The volumes are sorted (increasing order). 
    This means that for a simulation with a final number of bubbles of 3, at a
    time before the 1st breakup : volume[i] = [0, 0, V0]
    Then, after the first breakup it will be volume[j] = [0, V1, V0'] V0'>V1
    etc...
    
    All particles with a size inferior to (120/2**level)**3 are suppressed. 
    This means that if the simulation gives 6 bubbles but 2 are smaller than
    the threshold, the code will do the same analysis considering that only 4 
    bubbles are created. In particular len(volume[i]) = 4 for all i.
    """

    volume = [] 
    time = []
    
    i = 0
    
    Vcell = 4/3*np.pi*(120/2**level)**3 #size limit

    while i<len(data):
        vol = [] #list of the bubbles' volume at a given time
        time.append(data[i,1])
        #add the first bubble 
        if data[i,3]>Vcell: #check that the bubble is large enough
            vol.append(data[i,3])
        #add the other bubbles that exist at the same time if their volume is large enough
        while i<len(data)-1 and data[i,0]==data[i+1,0]:
            i += 1
            #check that the bubble has a volume superior than the volume of a cell
            if data[i,3]>Vcell:
                vol.append(data[i,3])
        vol = np.sort(vol) #order the volume in increasing order
        volume.append(vol)
        
        i += 1

    N_b = int(np.max(data[:,2])) #maximum number of bubbles in the simulation (not the final number because of coalescence)
    
    
    if N_b>1:
        V = np.zeros([len(volume), N_b])
        for i in range(len(volume)):
            for j in range(len(volume[i])):
                V[i,N_b-j-1] = volume[i][-j-1]
        #Suppress the false particle and non-dimensionnalise by the volume of the initial bubble
        i = 0
        while i <len(V[0]) and np.max(V[:,i])==0:
            i += 1
        V = V[:, i:]
    else:
        V = np.array(volume[:])
    if rescale_by_Vinit:
        V0 = np.max(V[:,-1])
        #print(V0)
        V = (V/V0)
    
    time = np.array(time)
    
    if np.shape(V[0])[0]==1:
        vol = np.array([V[i][0] for i in range(len(V))])
        V = vol.reshape(-1, 1)   
    return time, V


def radius(data,level=7,rescale_by_Rinit=False):
    ''' 
    Transforms the volumes in radius by doing the approximation that each particule is spherical
    Takes in argument the data given by import_bubbles or import_droplets and the level.
    If the level is not given it is fixed to 8.
    The radius is given considering the particle as a sphere.
    
    Returns 2 arrays: time, radius
    
    -time: 1D array 
    -np.shape(radius) = (Number of time step, number of particles)
    radius[i,j]: the radius of the particle j at step i
    
    The radius are sorted (increasing order).
    
    All particles with a size inferior to 2 times the size of a cell are automatically supressed.
    '''
    
    time, volume = volumes(data,level=level,rescale_by_Vinit=False)
    
    rad = (3*volume/(4*np.pi))**(1/3)
    if rescale_by_Rinit:
        if len(np.shape(rad))==2:
            rad /= np.max(rad[:, -1])
        else:
            print(radius)
            rad/= np.max(rad)

    return time, rad


def breakup_indices(radius):
    '''
    Takes in argument the radius or the volumes of the particles as given by 
    the functions radius or volumes.
    
    Return the list of all the indices of the breakups or coalescence events.
    
    Since sometimes there are "fake" breakup (the code doesn't know if there are
    2 or 1 bubbles so it oscillates very quickly between the two possibilities)
    especially before real breakup (the are several oscillations) we impose that
    the event is real if the daughter lives more than 10 units of time.
    '''
    indices = []
    i = 0
    while i<len(radius):
        
        j,k = i,i
        while j<len(radius)-1 and len(radius[j][radius[j]>0])==len(radius[j+1][radius[j+1]>0]):
            j += 1
        
        k = j+1

        while k <len(radius)-1 and len(radius[k][radius[k]>0])==len(radius[k+1][radius[k+1]>0]):
            k += 1
        
        if k < len(radius)-1:
            Nj1 = len(radius[j][radius[j]>0])
            Nj2 = len(radius[j+1][radius[j+1]>0])
            Nk1 =  len(radius[k][radius[k]>0])
            Nk2 = len(radius[k+1][radius[k+1]>0])
            
            if not (Nj1==Nk2 and Nj2==Nk1 and k-j <= 10):
                indices.append(j)
                i = k
                
            else:
                i = k+1
                
        elif j<len(radius)-1:
            indices.append(j)
            i = k
        
        else:
            i = k
        
    return indices

#def breakup_indices2(radius):
#    indices = []
#    #extract all the indices where the number of bubble changes
#    for i in range(len(radius)-1):
#        l1 = radius[i][radius[i]>0]
#        l2 = radius[i+1][radius[i+1]>0]
#        if len(l1) != len(l2):
#            indices.append(i)
#    #remove the false events (sometimes, there are oscillations of the number
#    #of bubbles at the moment of the breakup due to uncertainty)
#    n1 = len(indices)
#    n2 = 0
#    while n1 != n2:
#        ind = []
#        n1 = len(indices)
#        for i in range(len(indices)-2):
#            l1 = radius[indices[i]][radius[indices[i]]>0]
#            l2 = radius[indices[i+2]][radius[indices[i+1]]>0]
#            if indices[i+1]-indices[i]<=10 and len(l1)==len(l2):
#                pass
#            else:
#                ind.append(indices[i])
#        indices = ind
#        n2 = len(ind)
#        print(n1, n2)
#    return indices
    


def breakup(radius, time, ind_breakup):
    
    '''
    Take in entrance the radius or the volumes of the bubbles and the corresponding times given by the function radius() or volumes() and the number of the break-up of interest.
    
    Return:
        -the time at which the break-up occurs 
        -the different radius of the bubbles before and after.

    '''

    indices = breakup_indices(radius)
    
    if ind_breakup>=len(indices):
        return ('No break-up number {}'.format(ind_breakup))
    return time[indices[ind_breakup]], radius[indices[ind_breakup]],radius[indices[ind_breakup]+10]
    

def distance(L1, L2, index_m, index_f1, index_f2):
    '''
    Given the index of the mother index_m, the index of the 2 daughters
    index_f1 and index_f2, compute the "distance" between the volumes of the 
    bubbles before the event L1, and the one after L2.
    ie: computes the variation of volumes for the bubbles that stay over the event.
    '''
    s = 0
    for i in range(1, len(L1)):
        s2 = -1
        if i > index_m:
            s2 =np.abs(L1[i] - L2[i])
        elif i >= index_f2 and i < index_m:
            s2 = np.abs(L1[i] - L2[i+1])
        elif i > index_f1 and i < index_f2:
            s2 = np.abs(L1[i] - L2[i])
        elif i <= index_f1:
            s2 = np.abs(L1[i] - L2[i-1])
        if s2 != -1:
            s += s2
    return s

def distance2(L1, L2, index_f):
    '''
    Computes the distance between L1 and L2 if the bubble of index index_f was not
    there. Allows to know if it's a ghost bubble or not.
    '''
    s = 0
    for i in range(1,len(L1)):
        s2 = 0
        if i>index_f:
            s2 = np.abs(L1[i]-L2[i])
        elif i<=index_f:
            s2 = np.abs(L1[i]-L2[i-1])
        s += s2
    return s


def determine_event(L1, L2):
    
    '''
    Given 2 lists of volumes L1 and L2, determine the event of breakup that 
    minimize the change of volume between the bubbles that stay before and
    after.
    Test also if it's not a fake event of breakup, ie if a ghost bubble has 
    not appeared.
    '''
    
    l = len(L1)
    min_d = 1000000000000
    min_d_i = -1
    min_d_j = -1
    min_d_k = -1
    
    for i in range(l):
        for j in range(l):
            for k in range(j+1, l):
                d = distance(L1, L2, i, j, k)
                if d < min_d:
                    min_d = d
                    min_d_i = i
                    min_d_j = j
                    min_d_k = k
    d_j = distance2(L1, L2, min_d_j)
    d_k = distance2(L1, L2, min_d_k)
    
    if d_j == min(d_j, d_k, min_d):
        return None, [min_d_j], d_j
    elif d_k == min(d_j, d_k, min_d):
        return None, [min_d_k], d_k
    return [min_d_i], [min_d_j, min_d_k], min_d
    
def find_events(volume_bubbles):
    '''
    For a simulation determination of the breakup/coalescence events.
    Take in argument the array of the volumes given by the function volumes().
    
    Return an array giving the index of breakup (just before and just after), 
    the index of the mother(s), the index of the daughter(s).
    
    So we get data[i, 0] = [index_before, index_after] (useful for further 
    analysis when we will not have anymore index_after = index_before +1)
    data[i, 1] = list(volume_mother(s))
    data[i, 2] = list(volume_daughter(s))
        
    '''
    
    ind_breakup = np.array(breakup_indices(volume_bubbles))
    data = np.zeros([len(ind_breakup), 3], dtype=object)
    
    for i in range(len(ind_breakup)):
        ind = ind_breakup[i]
        N1 = len(volume_bubbles[ind][volume_bubbles[ind]>0])
        N2 = len(volume_bubbles[ind+1][volume_bubbles[ind+1]>0])
        
        data[i,0] = [ind, ind+1]
        if N1<N2:#breakup
            index_event = determine_event(volume_bubbles[ind], volume_bubbles[ind+1])
            data[i, 1] = index_event[0]
            data[i, 2] = index_event[1]
        else:#coalescence
            index_event = determine_event(volume_bubbles[ind+1], volume_bubbles[ind])
            data[i, 1] = index_event[1]
            data[i, 2] = index_event[0]
    
    return data

def isbreakup(event):
    '''
    Test if an event is a breakup and not a coalescence of a disappearance/apparition
    of a ghost bubble
    '''
    return  (not event[1] is None) and (not len(event[1])==2) and len(event[2])>=2


def volume_mothers(events, volume_bubbles):
    '''
    Given all the events returned by the function find events and the volume of 
    bubbles, return the volume of the mother for each break-up (don't take 
    into account the disappearance, apparition and coalescence)
    '''
    V = []
    for i in range(len(events)):
        #check that it's a breakup
        if isbreakup(events[i]):
            V.append(volume_bubbles[events[i, 0][0], events[i,1][0]])
    return V
    

def volume_daughters(events, volume_bubbles):
    '''
    Given all the events returned by the function find_events and the volumes of
    the bubbles, return the volume of daughters for each break-up (don't take 
    into account the disappearance, apparition and coalescence)
    '''
    V = []
    for i in range(len(events)):
        #check that is a breakup
        if isbreakup(events[i]):
            daughters = []
            for j in range(len(events[i,2])):#iterate on each daughter
                daughters.append(volume_bubbles[events[i, 0][1], events[i,2][j]])
            V.append(np.array(daughters))
    return V

def number_daughters(events):
    '''
    Given all the events given by determine_events, count the number of daughters 
    created by breakup events.
    Return the list of the number of daughters by event.
    '''
    N = []
    for i in range(len(events)):
        #check that it's a breakup (and not an apparition/coalescence/disparition)
        if isbreakup(events[i]):
            N.append(len(events[i,2]))
    return N
    
def weber_mothers(events, volume_bubbles, gamma, eps):
    '''
    Given all the events returned by the function find events and the volume of 
    bubbles, return the Weber of the mother for each break-up (don't take 
    into account the disparition, apparition and coalescence)
    '''
    weber = []
    for i in range(len(events)):
        #check that it's a breakup
        if isbreakup(events[i]):
            V = np.array(volume_bubbles[events[i, 0][0], events[i,1][0]])
            d = (6*V/np.pi)**(1/3)
            weber.append(va.WE(d, gamma, eps))
    return weber
    

def update_daughters(daughters, event):
    '''
    Updatse the list of daughters indexes given the previous daughters indexes
    and what happened at the next event.
    '''
    d = daughters.copy()
    part = False #to know if any of the daughters takes part in the new event
    
    if event[1] is None: #apparition (ghost bubble)
        for i in range(len(daughters)):
            #check that the bubble had a volume inferior the one of the bubble
            #that has just been created. In that case the index decreases.
            if daughters[i]<=event[2][0]:
                d[i] -= 1
#         #we don't include the created bubble
    
    elif event[2] is None: #disappearance
        
        for i in range(len(daughters)):
            if daughters[i]< event[1][0]:
                d[i] += 1
            elif daughters[i]==event[1][0]:
                index = i
#                d.pop(i)
                part = True
        if part:
            #remove the bubble that has disappeared
            d.pop(index)
    
    elif isbreakup(event):
        #check that it's not one of the daughters that breaks
        #just update the positions of the daughters
        if not event[1][0] in daughters:
            for i in range(len(daughters)):
                #for bubbles smaller than the smallest one, decrease the index
                if daughters[i]<=event[2][0]:
                    d[i] -= 1
                elif daughters[i]>= event[2][1] and daughters[i]<event[1][0]:
                    d[i] += 1
        else:
            part = True
            for i in range(len(daughters)):
                if daughters[i]<event[1][0] and daughters[i]>=event[2][1]:
                    d[i] += 1
                #remove the mother
                elif daughters[i] == event[1][0]:
                    ind = i
                elif daughters[i] <= event[2][0]:
                    d[i] -= 1
            #add the new daughters
            d += event[2]
            d.pop(ind)
            d.sort()
    
    else: #coalescence
        #check that non of the daughters have coalesced:
        if not (event[1][0] in daughters or event[1][1] in daughters):
            for i in range(len(daughters)):
                if daughters[i]>event[1][1] and daughters[i]<=event[2][0]:
                    d[i] -= 1
                elif daughters[i]<event[1][0]:
                    d[i] += 1
        else:
            part = True
            ind_min, ind_max = -2, -2
            for i in range(len(daughters)):
                if daughters[i]<= event[2][0] and daughters[i]>event[1][1]:
                    d[i] -= 1
                elif daughters[i]<event[1][0]:
                    d[i] += 1
                elif daughters[i]==event[1][1]:
                    ind_max = i
                elif daughters[i]==event[1][0]:
                    ind_min = i
            if ind_max != -2:#remove first the largest index so that does not 
                #change the index of the smallest one
                d.pop(ind_max)
            elif ind_min != -2:
                d.pop(ind_min)
            d.append(event[2][0])
            d.sort()
    return d, part
        
        
def find_events_given_dt(events, time, dt):
    '''
    Given all the binary events (given by find_events) and a typical time of 
    breakup, combine the events so that all the events linked to the initial 
    bubble that breaks and that occurs before dt later are now considered the 
    same event. 
    
    Return an array giving the index of breakup (just before and just after), 
    the index of the mother(s), the index of the daughter(s).
    
    Apparitions of ghost bubble are never considered as part of an event.
    '''
    new_events = []
    #done is created to know which event has been analysed already
    # as independant event of part of a more important one.
    done = np.zeros([len(events)], dtype=bool)
    
    def complete_event(events, time, dt, new_events, done):
        if False in done: #still some events not analysed
            #take the first not analysed event
            i0 = np.min(np.where(done==False))
            done[i0] = True
            #we don't consider coalescence, disparition and apparitions as independant event
            # except if they occur after a breakup and that one of the daughters is part of it
            if not isbreakup(events[i0]):
                new_events.append(list(events[i0]))
                #add the event to the list and continue with the other ones:
                return complete_event(events, time, dt, new_events, done)
    
            else:
                j = i0+1
                old_event = events[i0]
                daughters = old_event[2]
                #look at all the linked events that occur between the first breakup
                #  and t+dt
                while j<len(events) and time[events[j,0][1]]<time[old_event[0][0]]+dt:
                    #update the list of the daughters and the indicator 'done'
                    daughters, done[j] = update_daughters(daughters, events[j])
                    j += 1
                ind_init, ind_end = old_event[0][0], events[j-1,0][1]
                new_events.append([[ind_init, ind_end], old_event[1], daughters])
                # continue with the other breakup
                return complete_event(events, time, dt, new_events, done)
        #all the events have been analysed
        return new_events

    new_events = complete_event(events, time, dt, new_events, done)
    
    return np.array([new_events[i] for i in range(len(new_events))])
    

def add_value(t, ind, value):
    '''
    Given an ordered array where all non important values are at the beginning of
    the array, inject value at the position ind, and shift the position of the others.
    '''
    for i in range(1,ind+1):
        t[i-1] = t[i]
    t[ind] = value

def remove_value(t, ind, fillwith=0):
    '''
    Given an ordered array where all non important values are at the beginning of
    the array, remove a value at the position ind, and shift the position of the others.
    '''
    for i in range(ind,0, -1):
        t[i] = t[i-1]
    t[0] = fillwith

def life_time(events, time, vol_bubbles):
    '''
    Computes the life time of every bubble.
    
    Returns 3 lists, one containing the volume of every bubble, the other containing 
    all the life times. The last one is a list of booleans. The value is True 
    if the bubble broke, and False if it coalesced.
    
    Events contains all the 'binary events' so it's the results of find_events.
    time and vol_bubbles are the outputs of volumes.
    '''
    
    #life time and value of the bubbles before they break or they coalesce
    life_time, V, broke = [], [], []
    #create an array containing all the times at which bubbles are created
    t_creation = np.zeros(len(vol_bubbles[0]))
    t_creation[-1] = time[0]
    
    for i in range(len(events)):
        #if not an apparition
        if isbreakup(events[i]):
            ind_mother = events[i,1][0]
            #add the life time of the mother and its volume
            life_time.append(time[events[i,0][0]]-t_creation[ind_mother])
            V.append(vol_bubbles[events[i,0][0],ind_mother])
            broke.append(True)
            #update the creation times
            remove_value(t_creation, ind_mother)
            add_value(t_creation, events[i,2][1], time[events[i,0][1]])
            add_value(t_creation, events[i,2][0], time[events[i,0][1]])
        
        elif events[i, 2] is None:#disappearance
            remove_value(t_creation, events[i,1][0])
        
        elif events[i,1] is None:#apparition
            add_value(t_creation, events[i,2][0], time[events[i,0][1]])
        else: #coalescence
            ind_mother = events[i,1]
            #add the life time and volume of the 2 bubbles
            life_time.append(time[events[i,0][0]]-t_creation[ind_mother[0]])
            life_time.append(time[events[i,0][0]]-t_creation[ind_mother[1]])
            V.append(vol_bubbles[events[i,0][0], ind_mother[0]])
            V.append(vol_bubbles[events[i,0][0], ind_mother[1]])
            broke.append(False)
            broke.append(False)
            #update the creation times
            remove_value(t_creation, ind_mother[0])
            remove_value(t_creation, ind_mother[1])
            add_value(t_creation, events[i,2][0], time[events[i,0][1]])
            
    return life_time, V, broke


def time_intervals(events, time, vol_bubbles):
    '''
    Computes the life time of every bubble and the series of intervals of time
    that have led to this breakup.
    
    Returns 4 lists:
        -the volume of every bubble
        -the life times. 
        -parent event index
        -list of booleans. The value is True if the bubble broke, and False if it coalesced.
    
    Events contains all the 'binary events' so it's the results of find_events.
    time and vol_bubbles are the outputs of volumes.
    '''
    
    #life time and value of the bubbles before they break or they coalesce
    life_time, Vm, Vd, ind_parent, broke = [], [], [], [], []
    #create an array containing all the times at which bubbles are created
    t_creation = np.zeros(len(vol_bubbles[0]))
    t_creation[-1] = time[0]
    pi = -2*np.ones(len(t_creation))
    pi[-1] = -1

    for i in range(len(events)):
        #if not an apparition
        if isbreakup(events[i]):
            ind_mother = events[i,1][0]
            #add the life time of the mother and its volume
            life_time.append(time[events[i,0][0]]-t_creation[ind_mother])
            Vm.append([vol_bubbles[events[i,0][0],ind_mother]])
            Vd.append([vol_bubbles[events[i, 0][0], events[i, 2][j]] for j in range(len(events[i,2]))])
            ind_parent.append(pi[ind_mother])
            broke.append(True)
            #update the creation times
            remove_value(t_creation, ind_mother)
            add_value(t_creation, events[i,2][1], time[events[i,0][1]])
            add_value(t_creation, events[i,2][0], time[events[i,0][1]])
            #update the parent event index
            remove_value(pi, ind_mother, fillwith=-2)
            add_value(pi, events[i, 2][1], i)
            add_value(pi, events[i, 2][0], i)
        
        elif events[i, 2] is None:#disappearance
            life_time.append(time[events[i,0][0]]-t_creation[events[i, 1][0]])
            ind_parent.append(pi[events[i, 1][0]])
            Vm.append([vol_bubbles[events[i,0][0], events[i,1][0]]])
            Vd.append([])
            broke.append(False)
            remove_value(t_creation, events[i,1][0])
            remove_value(pi, events[i, 1][0], fillwith=-2)
        
        elif events[i,1] is None:#apparition
            life_time.append(0)
            ind_parent.append(-2)
            Vm.append([])
            Vd.append([vol_bubbles[events[i, 0][0], events[i, 2][0]]])
            broke.append(False)
            add_value(t_creation, events[i,2][0], time[events[i,0][1]])
            add_value(pi, events[i,2][0], i)
        else: #coalescence
            ind_mother = events[i,1]
            #add the life time and volume of the 2 bubbles
            life_time.append([time[events[i,0][0]]-t_creation[ind_mother[j]] for j in range(len(ind_mother))])
#            life_time.append(time[events[i,0][0]]-t_creation[ind_mother[1]])
            Vm.append([vol_bubbles[events[i,0][0], ind_mother[j]] for j in range(len(ind_mother))])
            Vd.append([vol_bubbles[events[i, 0][0], events[i, 2][0]]])
            ind_parent.append([pi[ind_mother[0]], pi[ind_mother[1]]])
            broke.append(False)
            broke.append(False)
            #update the creation times
            remove_value(t_creation, ind_mother[0])
            remove_value(t_creation, ind_mother[1])
            remove_value(pi, ind_mother[0], fillwith=-2)
            remove_value(pi, ind_mother[1], fillwith=-2)
            add_value(t_creation, events[i,2][0], time[events[i,0][1]])
            add_value(pi, events[i, 2][0], i)
            
    return life_time, Vm, Vd, ind_parent, broke

    

def T2(radius, gamma, rho =1):
    '''
    Return the period of the second mode of oscillation of the bubble.
    Takes in argument:
        -radius: Can be either a scalar or an array
        -gamma: the surface tension of the interface
    Optionnal argument:
        -rho: the density of the continuous medium
    Return:
        -T2 can be either a scalar or an array, depending on the input radius
    '''
    return 1/2*np.sqrt(rho/(3*gamma))*radius**(3/2)

def tb(radius, gamma, eps, rho=1,C_eps=2):
    '''
    Calculate the turbulent time scale based on the radius of the bubble:
    Take in argument:
        -radius: Can be either a scalar or an array
        -gamma: surface tension of the interface
        -eps: the dissiation rate inside the turbulent flow
    Optional arguments:
        -C_eps: 
        -rho: density of the continuous medium
    
    '''
    return (C_eps/eps)**(1/3)*radius**(2/3)

def We(radius,gamma, eps, C_eps=2, rho=1):
    '''Compute the Weber number of a bubble given it's radius (radius), the
    surface tension (gamma), the disipation rate (eps).
    In option, can give the coefficient between kinetic energy and dissipation 
    rate (by default 2) and the density of the liquid (by default 1).
    '''
    return rho*eps**(2/3)*2**(5/3)/(C_eps**(2/3)*gamma)*radius**(5/3)
    



def fD_MB(D, We0):
    '''
    Probability density function for binary break-ups for the diameters of
    the daughters as given in the paper from Martinez Bazan 2010 'Considerations
    on bubble fragmentation models'
    equation 3.8
    Lambda is linked to the We number of the initial bubble with: 
    Lambda**(-5/3) = We(initial bubble)
    '''
    
    def f(D):
        return D**2*(D*(2/3)-Lambda**(5/3))*((1-D**3)**(2/9)-Lambda**(5/3))
    Dmin = (1/12*We0)**(-3/2)
    Lambda = (Dmin)**(2/5)
#    Dmin = Lambda**(5/2)
    Dmax = (1-Lambda**(15/2))**(1/3)
    I = integrate.quad(f, Dmin, Dmax)
    return f(D)/I[0]

def fV_MB(V, We0):
    '''
    Probability density function for binary break-ups for the volumes of
    the daughters as given in the paper from Martinez Bazan 2010 'Considerations
    on bubble fragmentation models'
    equation 3.8
    Lambda is linked to the We number of the initial bubble with: 
    Lambda**(-5/3) = We(initial bubble)
    '''
    def f(V):
        return (V**(2/9)-Lambda**(5/3))*((1-V)**(2/9)-Lambda**(5/3))
    alpha = 3*2**(-2/9)
    #dmin = (8.2/24*we)**(-3/2)
    dmin = (We0/alpha)**(-3/2)
    Vmin = dmin**3#(8.2/(2*12)*We0)**(-9/2)
    print(Vmin)
    Lambda = Vmin**(2/15)
    print(Lambda)
#    Vmin = Lambda**(15/2)
    Vmax = 1 - Vmin
    I = integrate.quad(f, Vmin, Vmax)
    return f(V)/I[0]

def fD_TT(D, D_min):
    '''
    Probablibily density function for binary breakup developed by Tsouris and 
    Tavlarides (1994) and modified a little bit by Martinez Bazan to conserve
    the volume. The expression is given in equation 3.20 from the paper:
        Consideration on bubble fragmentation model by Martinez bazan.
    Need to give a minimum size D_min which is arbitrary.
    
    The limits of the integral are changed to D_min and D_max since by definition
    D cannot take those values.
    '''
    D_max = (1-D_min**(3))**(2/3)
    def f(D):
        return (D_min**2 + (1 - D_min**3)**(2/3) -1 + 2**(1/3) -D**2 - (1 - D**3)**(2/3))*D**2
    I = integrate.quad(f, D_min, D_max)
    return f(D)/I[0]



def fV_TT(V, V_min):
    '''
    Probablibily density function for binary breakup developed by Tsouris and 
    Tavlarides (1994) and modified a little bit by Martinez Bazan to conserve
    the volume. The expression is given in equation 3.20 from the paper:
        Consideration on bubble fragmentation model by Martinez bazan.
    Need to give a minimum size V_min which is arbitrary.
    
    The limits of the integral are changed to V_min and V_max since the volume
    cannot take those values by definition.
    '''
    V_max = 1-V_min
    def f(V):
        return V_min**(2/3) + (1 - V_min)**(2/3) -1 + 2**(1/3) -V**(2/3) - (1 - V)**(2/3)
    I = integrate.quad(f, V_min, V_max)
    return f(V)/I[0] 

