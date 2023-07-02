#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:14:26 2020

@author: riviere

All the functions to analyze the file bubble.npz and find the events
"""

import numpy as np
import copy

keys = ['x', 'y', 'z']

#prediction of the position of the 2nd bubble
def newx(x0, V0, x1, V1, L):
    '''
    Position of the 2nd daughter given the position and volume of the mother and the 1st daughter:
    x0V0 = x1V1 + x2V2
    x2 = x0V0/(V0 - V1) - x1V1/(V0-V1)
    '''
    return np.mod(-signed_dist(x1, x0, L)*V1/(V0 - V1) + x0, L)

def signed_dist(x1, x0, L):
    '''
    Express x1 in a system of coordinate centered in x0.
    '''
    return np.mod(x1 - x0 + L/2, L) - L/2


def distance(part1, index1, part2, index2, L, keys = ['x', 'y', 'z']):
    dist = 0
    for key in keys:
        dist += signed_dist(part1[key][index1], part2[key][index2], L)**2
    return np.sqrt(dist)

#### Find the compatible bubbles
def find_compatible(part0, traj):
    '''
    Find bubbles that never exist at the same time as part0.
    Return the list of their index.
    '''
    fmin0 = part0['frame'].min()
    fmax0 = part0['frame'].max()

    indnotcompatible = []
    for i in traj['particle'].unique():
        #look for moment where virtually they are both present
        fmin = max(fmin0, traj[traj['particle']==i]['frame'].min())
        fmax = min(fmax0, traj[traj['particle']==i]['frame'].max())
        if fmin<=fmax:
            cond = (traj['particle']==i) & (traj['frame']>= fmin) & (traj['frame']<= fmax)
            cond0 = (part0['frame']>=fmin) & (part0['frame']<=fmax)

            time_1 = traj[cond]['t']
            time_0 = part0[cond0]['t']
            #test if during this time interval they are both present ie: 
            # if there is any time step in common between the two arrays

            if len(time_1[time_1.isin(time_0)])>=1:
                indnotcompatible.append(i)

    return indnotcompatible

def find_candidates_b(partdead, traj):
    fdeath = partdead['frame'].max()
    fbirth = partdead['frame'].min()
    #bubbles existing just after the death
    indcandidate = traj['frame']==fdeath + 1
    #look for bubbles compatible with i0: ie, bubbles never existing at the same time
    traj_sametime = traj[(traj['frame']<=fdeath) & (traj['frame']>= fbirth)]
    indnotcompatible  = find_compatible(partdead, traj_sametime)
    #print(traj[indcandidate]['particle'].unique())
    #print(traj[~traj['particle'].isin(indcompatible)]['particle'].unique())
    return indcandidate & (~traj['particle'].isin(indnotcompatible)) 

def find_candidates_c(partdaughter, traj, fdeath, ldying):
    indcandidate_c = traj['frame']==fdeath
    fmin = partdaughter['frame'].min()
    fmax = partdaughter['frame'].max()
    traj_sametime = traj[(traj['frame']>=fmin) & (traj['frame']<=fmax)]
    indnotcompatible_c = find_compatible(partdaughter, traj_sametime)
    return indcandidate_c & (~traj['particle'].isin(indnotcompatible_c))  & (traj['particle'].isin(ldying))

####
def same_bubble(part0, part1, rel_dV, toldx, Vlim, L):
    index0 = part0[part0['frame']==part0['frame'].max()].index[0]
    index1 = part1[part1['frame']==part1['frame'].min()].index[0]
    v0 = part0['volume'][index0]
    v1 = part1['volume'][index1]
    cond_vol = abs(v0 - v1) < max(rel_dV*max(v0, v1), Vlim/4**3)
    cond_pos = distance(part0, index0, part1, index1, L=L)<3*toldx
    return cond_vol & cond_pos

def test_same_part(part0, part1, tolV, toldx, L):
    index0 = part0.index[0]
    index1 = part1.index[0]
    cond_vol = abs(part0.volume[index0] - part1.volume[index1])<=tolV
    cond_pos = distance(part0, index0, part1, index1, L=L)<5*toldx
    return cond_pos & cond_vol

def test_bubble_in_candidates(partdead0, trajcandidate, tolVol, toldx, L):
        index = trajcandidate.sort_values(by=['volume'], \
                                     ascending = False).index
        for i in index:
            part1 = trajcandidate[trajcandidate['volume']==trajcandidate['volume'][i]]
            same = test_same_part(partdead0, part1, tolVol, toldx, L=L)
            if same:
                return 't', [part1.particle[i]]
        return '', []
            
def compare_onecandidate(partdead0, index0, V0, part1, index1, tolVol, toldx, Vlim, L):
    '''
    The bubbles should not be the same.
    '''
    tolVol = max(tolVol, Vlim/4**3)#if the variations are smaller than the size of one cell take one cell variation
    DVi = part1['volume'][index1] - V0
    cond_vol = abs(DVi) <= tolVol
    dist = distance(part1, index1, partdead0, index0, L)
#    cond_pos = dist <= 3*toldx
    #print(dist)
    if cond_vol and dist<=10*toldx:#small particle jiggles
        return 't', [part1['particle'][index1]]
    elif dist>=15*toldx and cond_vol:#same volume but far from each other
        return 'd', []
    #print(cond_pos, cond_vol)
    elif DVi<0 and abs(DVi)<Vlim:# Breakage with a very small bubble produced
        return 'bs', [part1['particle'][index1]]
    elif DVi > 0 and DVi <= Vlim:# Coalescence with a very small bubble
        return 'cs', [part1['particle'][index1]]
    elif DVi > 0 and DVi>Vlim:# Coalescence that needs to be determined
        return 'c', [part1['particle'][index1]]

    print('bizarre', DVi, Vlim)#logically: abs(DVi)>Vlim: cassure et la 2e n'est pas petite
    return 'pb', []

#####
def find_event_b(partdead0, trajcandidate, tolVol, toldx, Vlim, L):
    index0 = partdead0.index[0]
    Nbr_candidate = len(trajcandidate['particle'].unique())
    
    if Nbr_candidate==0:
        return 'd', [] #disappearance
    #test if the bubble is in one of the candidates
    kind, daughters = test_bubble_in_candidates(partdead0, trajcandidate, tolVol, toldx, L=L)
    if kind=='t':
        return kind, daughters

    V0 = partdead0['volume'][index0]    
    if Nbr_candidate==1:
        index = trajcandidate.index[0]
        return compare_onecandidate(partdead0, index0, V0, trajcandidate, index, tolVol, toldx, Vlim, L=L)


    elif Nbr_candidate==2:
        index = trajcandidate.sort_values(by=['volume'], \
                                         ascending = False).index
        indexM = index[0]
        VM = trajcandidate['volume'][index[0]]
        DVi = VM - V0
        cond_vol = abs(DVi) <= tolVol
        cond_pos = distance(trajcandidate, index[0], partdead0, index0, L) <= 3*toldx
        if DVi < 0:# break-up
            if abs(DVi)< Vlim:# Breakage with a very small bubble produced
                return 'bs', [trajcandidate['particle'][indexM]]
            else:
                index = trajcandidate.index
                V1, V2 = trajcandidate['volume'][index[0]], trajcandidate['volume'][index[1]]
                DV = V1 + V2 - V0
                cond_vol = abs(DV) <= tolVol
                dist = 0
                for key in keys:
                    x1 = newx(partdead0[key][index0], V0, trajcandidate[key][index[0]], V1, L=L)
                    dist += signed_dist(trajcandidate[key][index[1]], x1, L)**2
                cond_pos = np.sqrt(dist) <= 3*toldx
                if cond_vol and cond_pos:#break-up!
                    return 'b', [trajcandidate['particle'][index[0]], \
                                 trajcandidate['particle'][index[1]]]
                elif DV<0 and abs(DV)<Vlim:# Breakup with a third small bubble produced
                    return 'bst', [trajcandidate['particle'][index[0]], \
                                   trajcandidate['particle'][index[1]]]
                elif DV>0 and abs(DV)<Vlim:#breakage plus coalescence with a small bubble
                    return 'bsct', [trajcandidate['particle'][index[0]], \
                                   trajcandidate['particle'][index[1]]]
                return 'pb', [] #case DV>Vlim 


        elif DVi > 0:# coalescence
            if abs(DVi)<Vlim:# coalescence with a small bubble
                return 'cs', [trajcandidate['particle'][indexM]]
            return 'c', [trajcandidate['particle'][indexM]]

    else:
        VM = trajcandidate['volume'].max()
        indexM = trajcandidate[trajcandidate['volume']==VM].index[0]
        DVi = VM - V0
        cond_vol = abs(DVi) <= tolVol
        cond_pos = distance(trajcandidate, indexM, partdead0, index0, L) <= 3*toldx
        if cond_vol and cond_pos:# same bubble
            return 't', [trajcandidate['particle'][indexM]]
        elif DVi < 0:# break-up
            if abs(DVi)< Vlim:#Breakage with a very small bubble produced
                return 'bs', [trajcandidate['particle'][indexM]]
            else:
                #test all the possibilities for the breakup
                index = trajcandidate.sort_values(by=['volume'], \
                                                    ascending = False).index
                for i in range(len(index)-1):
                    for j in range(i, len(index)):
                        indb1 = index[i]#first candidate
                        indb2 = index[j]#second candidate
                        DV = trajcandidate['volume'][indb1] \
                            +trajcandidate['volume'][indb2]\
                            - V0
                        cond_vol = abs(DV) <= tolVol
                        #test the positions
                        dist = 0
                        for key in keys:
                            x1 = newx(partdead0[key][index0], V0, \
                                      trajcandidate[key][indb1], trajcandidate['volume'][indb1], L=L)
                            dist += signed_dist(trajcandidate[key][indb2], x1, L)**2
                        cond_pos = np.sqrt(dist) <= 3*toldx
                        if cond_vol and cond_pos:#break-up
                            return 'b', [trajcandidate['particle'][indb1],\
                                         trajcandidate['particle'][indb2]]
                        elif DV<0 and abs(DV)<Vlim:
                            return 'bst', [trajcandidate['particle'][indb1],\
                                         trajcandidate['particle'][indb2]]
            return 'pb', []


        elif DVi > 0:# coalescence
            if abs(DVi)<Vlim:# coalescence with a small bubble
                return 'cs', [trajcandidate['particle'][indexM]]
            else:# coalescence that needs to be determined
                return 'c', [trajcandidate['particle'][indexM]]
            
def find_coalescence(partdead0, partdaughter0, trajcandidate_c, tolVol, toldx, Vlim, L):
    index_c0 = partdaughter0.index[0]
    index0 = partdead0.index[0]
    Nbr_candidate_c = len(trajcandidate_c['particle'].unique())
    if Nbr_candidate_c==1:
        print('hum... bizarre')
        return 'pb', []
    elif Nbr_candidate_c==2:
        index = trajcandidate_c.index
        DV = trajcandidate_c['volume'][index[0]] + trajcandidate_c['volume'][index[1]]\
            - partdaughter0['volume'][index_c0]
        cond_vol = abs(DV) <= tolVol
        #test the positions
        dist = 0
        for key in keys:
            x1 = newx(partdaughter0[key][index_c0], partdaughter0['volume'][index_c0], \
                      trajcandidate_c[key][index[0]], trajcandidate_c['volume'][index[0]], L=L)
            dist += signed_dist(trajcandidate_c[key][index[1]], x1, L)**2
        cond_pos = np.sqrt(dist) <= 3*toldx
        if cond_vol and cond_pos:# found the two bubbles that have coalesced
            return 'c', [trajcandidate_c['particle'][index[0]], \
                         trajcandidate_c['particle'][index[1]]]
        else: # no mother, was a disappearance
            return 'd', []
    else: #Nbr_candidate >2
        index = list(trajcandidate_c.index)
        index.remove(index0)
        for i in index:
            DV = trajcandidate_c['volume'][i] + trajcandidate_c['volume'][index0]\
                - partdaughter0['volume'][index_c0]
            cond_vol = abs(DV) <= tolVol
            #test the positions
            dist = 0
            for key in keys:
                x1 = newx(partdaughter0[key][index_c0], partdaughter0['volume'][index_c0], \
                          trajcandidate_c[key][i], trajcandidate_c['volume'][i], L=L)
                dist += signed_dist(trajcandidate_c[key][index0], x1, L)**2
            cond_pos = np.sqrt(dist) <= 3*toldx
            if cond_vol and cond_pos:# found the two bubbles that have coalesced
                return 'c', [trajcandidate_c['particle'][index_c0], \
                             trajcandidate_c['particle'][i]]
            else: # no mother, was a disappearance
                return 'd', []
    
########
def find_event(partdead, dying, traj, rel_DV, toldx, Vlim, L):
    candidates = find_candidates_b(partdead, traj)

    fdeath = partdead['frame'].max()
    partdead0 = partdead[partdead['frame']==fdeath]
    #print(partdead0)
    #print(traj[candidates])
    index0 = partdead0.index[0]
    tolV0 = partdead['volume'][index0]*rel_DV
    
    index_daughters, index_mother = [], []
    #a = find_event_b(partdead0, traj, candidates, \
    #                                   tolVol=tolV0, toldx=dx, Vlim=Vlim, L=L)
    #print(a)
    trajcandidates_b = traj[candidates]
    kind, index_daughters = find_event_b(partdead0, trajcandidates_b, \
                                       tolVol=tolV0, toldx=toldx, Vlim=Vlim, L=L)
    #if potential coalescence, search for the mothers
    if kind=='c':
        part1 = traj[traj['particle']==index_daughters[0]]
        part10 = part1[part1['frame']==fdeath + 1]
        index_c0 = part10.index[0]
        tolV0_c = part10['volume'][index_c0]*rel_DV

        candidates_c = find_candidates_c(part1, traj, fdeath, dying)
        #print(part10)
        #print(traj[candidates_c])
        trajcandidates_c = traj[candidates_c]
        kind, index_mother = find_coalescence(partdead0, part10, trajcandidates_c, tolVol=tolV0_c, \
                                              toldx=toldx, Vlim = Vlim, L=L )
        if kind == 'pb':
            #look for the daughters of the daughter should find one which is the dead one
            candidates_b2 = find_candidates_b(part1, traj)
            #print(traj[candidates_b2])
            for i in traj[candidates_b2]['particle'].unique():
                part2 = traj[traj['particle']==i]
                if same_bubble(partdead, part2, 2*rel_DV, 3*toldx, Vlim, L=L):
                    kind = 'tsp'
                    index_mother = []
                    index_daughters = [i]
            
        if kind == 'd':# event was a disappearance: no child
            index_daughters = []
    return kind, index_mother, index_daughters


#####
def find_dying(traj):
    '''
    Take in entry the trajectories of the bubbles. Return the list of indexes of 
    all bubbles that are disappearing (dying).
    '''
    Tfinal = traj['t'].max()
    #dying bubbles: all the bubbles that die before the end of the simulation
    ind_dying = traj.groupby(['particle'])['t'].max() < Tfinal
    dying_part = []
    for i in traj['particle'].unique():
        if ind_dying[i]:
            dying_part.append(i)
    return dying_part

def clean_events(traj0, events):
    traj = traj0.copy()
    clean_events = copy.deepcopy(events)
    #suppress the spurious events and update the traj file to suppress the spurious bubbles
    for key, value in events.items():
        if key != 0 and ((value['death'] in ['sp', 'd', '']) and len(value['mother'])==0):
            del clean_events[key]
#            traj.drop(traj.particle==key, inplace=True)
            traj = traj[traj.particle!=key].copy()
        elif (value['death']=='d') and ((value['mother']==[]) and  (value['daughter']==[])):
            #print('spurious')
            del clean_events[key]

        else:
            if len(value['mother'])==1:
                clean_events[key]['mother'] = value['mother'][0]
            if  len(value['daughter'])==1:
                clean_events[key]['daughter'] = value['daughter'][0]
                
    list_events = np.unique(np.array([event[1]['death'] for event in clean_events.items()]))
    while 't' in list_events:
        c0 = copy.deepcopy(clean_events)
        for key in clean_events.keys():
            if key in c0.keys():
                if c0[key]['death']=='t':
                    key_daughter = c0[key]['daughter'][0]
                    if key_daughter in c0.keys():
                        c0[key]['death'] = clean_events[key_daughter]['death']
                        c0[key]['daughter'] = clean_events[key_daughter]['daughter']
                        del c0[key_daughter]
                        mask = traj['particle'].isin([key_daughter])
                        traj.loc[mask, 'particle'] = key
                        #traj['particle'] = traj['particle'].replace([key_daughter], key)
                    else:
                        c0[key]['death'] = ''
                        c0[key]['daughter'] = []
        clean_events = copy.deepcopy(c0)
        list_events = np.unique(np.array([event[1]['death'] for event in c0.items()]))

    #remove the small bubbles that are not interacting with the others
    c = copy.copy(clean_events)
    for key, value in c.items():
        if (value['death']=='d') and (value['mother']==[]):
            del clean_events[key]
            traj = traj[traj.particle!=key].copy()
        else:
            part = traj[traj.particle==key]
            tmin, tmax = part.t.min(), part.t.max()
            clean_events[key]['t0'] = tmin
            clean_events[key]['t1'] = tmax
    return clean_events, traj


################

def find_events(traj, rel_DV, toldx, Vlim, L):
    dying = find_dying(traj)
    #print(dying)
    #initialize the dictionary of events
    events = {}
    for i in traj['particle'].unique():
        events[i] = {}
        events[i]['death'], events[i]['mother'], events[i]['daughter'] = '', [], []
    
    for i0 in dying:
        #print(i0)
        partdead = traj[traj['particle']==i0]
        kind, index_mother, index_daughters = find_event(partdead, dying, traj,\
                                                         rel_DV, toldx, Vlim, L)
        if not i0 in index_mother:
            index_mother.append(i0)
        #daughters created before the mother and the daughter already has a mother
        cond_pb1 =  (np.array(index_daughters) < i0).any() and \
            (np.array([len(events[i]['mother']) for i in index_daughters])>0).any()
        #daughter created before the mother which is the initial bubble
        cond_pb1p =  (np.array(index_daughters) < i0).any() and (np.array(index_daughters) ==0).any()
        #mothers created after the daughter
        cond_pb2 = (kind == 'c') and (np.array(index_mother)>index_daughters[0]).any()
        if cond_pb1 or cond_pb1p or cond_pb2:
            kind = 'sp'
            index_daughters, index_mother = [], []
        if kind=='pb':
            index_daughters = []
        print(kind, index_mother, index_daughters)

        events[i0]['death'] = kind

        for i in index_daughters:
            if not index_mother in events[i]['mother']:
                events[i]['mother'].append(index_mother)
        for j in index_mother:
            if not index_daughters in events[j]['daughter']:
                events[j]['daughter'].append(index_daughters)
        #filter to remove spurious events, tracking pb etc...
    events, traj = clean_events(traj, events)
        
    return events, traj

def find_generation(dico, generation, key_mother):
    gen = max(generation, dico[key_mother]['generation'])
    dico[key_mother]['generation'] = gen
    for i in range(len(dico[key_mother]['daughter'])):
        key_daughter = dico[key_mother]['daughter'][i]
        find_generation(dico, gen+1, key_daughter)