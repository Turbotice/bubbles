#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:38:27 2019

@author: Alienor

File containing all the physical variables used for the simulations.
The first index always refers to the medium. The value 0 corresponds to a
forcing of 0.1 (so the Reynolds 37), 1 to the medium with a forcing 0.3 (Re = 76).
When accurates, the second index refers to the Weber number in the simulation in
the order : We=2, 5, 10, 15, 1

So in general you have parameter[medium][We].

"""

import numpy as np
from . import fi as fi

############### Numerical parameters #################
#OLD DATA
#Number of simulations per We and Re:
#the last line refers to the merged data (with level 11 and 12) for the Re 37
#N_simu = np.array([[19, 15, 20, 18],[20, 20, 0, 0], [38, 0, 0, 19]])
N_simu = np.array([[39, 19, 20, 21],[20, 20, 0, 0], [59,39,20,21]])
#Nbr_unbroken_bubbles = np.array([[5,0,0,0],[1,0,0,0], [8, 0, 0, 0]])

#the last list is the sum of the two previous ones
Nbr_unbroken_bubbles = np.array([[8,0,0,0],[1,0,0,0], [9,0,0,0]])
#Size of the box
L_box = 120

############# Physical constants ###########

# eps = C_esp * u_rms^3/L FALSE
#new def: u_(d0)^2 = C (eps d0)^(2/3)  
C = 2
C_eps = 1/C**(3/2)
WEC = 3

######### Caracteristic properties of the medium ############
# the values are computed from the precursors (the simulations that produce the
# turbulent mediums). We run the simlations for a long time after the transition
# to turbulence. Then the properties are computed on the stationary part.

#Reynolds number at the Taylor microscale
Re = np.array([37.9, 75.9])
STD_Re = np.array([5.3, 11.6]) #standard deviation of the Reynolds number
#dissipation rate 
EPS = np.array([9.2, 212.9])
EPS2 = {'Re 37':9.2, 'Re 76':212.9}
STD_EPS = np.array([0.9, 52.0])
STD_EPS2 = {'Re 37':0.9, 'Re 76':52.0}
#density of kinematic energy
KIN = np.array([57.9, 610.4])
KIN2 = {'Re 37':57.9, 'Re 76':610.4}
STD_KIN = np.array([9.3, 146])
STD_KIN2 = {'Re 37':9.3, 'Re 76':146}
#Viscosity
MU = 0.01*(L_box/(2*np.pi))**2*1/2
#Density of the liquid
RHO = 1
#Reynolds number at large scale
RE = 3/20*Re**2

####### Caracteristic properties of the bubble #########

#surface tension coefficient
GAMMA = np.array([[141, 50, 28, 18.7, 280],[1100,456,228,152, 1]]) #the value of GAMMA[1, -1] is false since not used
GAMMA2 = {'We 1.5':595, 'We 3':280, 'We 6':141, 'We 15':50, 'We 30': 28, 'We 45': 18.7}
# Diameter of the initial bubble. It is set to 16 normally but simulations show
# that only 15.7 is reached
D0 = 15.7

######### Usefull Scales ############

##lengthscales

#Hinze lengthscale: size of the smallest stable bubble ie for We = Wec ~ 1,5
L_HINZE = np.array([WEC**(3/5)*C_eps**(2/5)*EPS[i]**(-2/5)*(GAMMA[i]/RHO)**(3/5) for i in range(len(EPS))])
L_HINZE2 = {key:WEC**(3/5)*C_eps**(2/5)*EPS2['Re 37']**(-2/5)*(item/RHO)**(3/5) for key, item in GAMMA2.items()}
#Taylor microscale: linked to the autocorrelation function of the velocity
L_TAYLOR = 1/RHO*np.sqrt(10*MU*KIN/EPS)
L_TAYLOR2 = {key:1/RHO*np.sqrt(10*MU*KIN2[key]/EPS2[key]) for key in KIN2.keys()}
#Kolmogorov scale: size at which the energy dissipation occurs, size of the smallest eddies
L_KOLMOGOROV = ((MU/RHO)**3/EPS)**(1/4)
L_KOLMOGOROV2 = {key:((MU/RHO)**3/EPS2[key])**(1/4) for key in EPS2.keys()}
#Lenght used in Meneveau, caracteristic scale of the turbulent medium:
L_MENEVEAU = 1/EPS*(2*KIN/(3*RHO))**(3/2)
L_MENEVEAU2 = {key:1/EPS2[key]*(2*KIN2[key]/(3*RHO))**(3/2) for key in EPS2.keys()}

##Timescales

#Hinze timescale: link between T2 and tb
T_HINZE = np.zeros(np.shape(GAMMA))
for i in range(len(GAMMA)):
    T_HINZE[i] = (12)**(2/5)*(GAMMA[i]/RHO)**(2/5)*(C_eps/EPS[i])**(3/5)
#Kolmogorov timescale
T_KOLMOGOROV = np.sqrt(MU/RHO/EPS)
#Time used by Meneveau, says that it's the eddy turnover time:
T_MENEVEAU = (2*KIN)/(3*RHO*EPS)

##caracteristic velocities

#Kolmogorov velocity
U_KOLMOGOROV = (EPS*MU/RHO)**(1/4)

### Scalings ###
#Size of the largest eddies
l0 = L_KOLMOGOROV*RE**(3/4) ##from S.B.Pope TurbulentFlow (p186): (l0/eta)~Re^(-3/4)


##### Useful functions #####

def WE(diameter, gamma=None, eps=None, withvolume=False):
    ''''
    Weber number at the Taylor lengthscale associated to a bubble of 
    diameter=diameter, surface tension=gamma in a medium where the dissipation 
    rate is eps.
    If gamma and eps are not given, the We is computed for all the values of
    gamma and eps used in the simulations.
    
    Returns: We
        -if gamma and eps are given and are scalars, We is a scalar
        -else, We is an array
    '''
    if withvolume:
        diameter = 2*(3/(4*np.pi)*diameter)**(1/3)
    if gamma is None:
        We = np.zeros(np.shape(GAMMA))
        for i in range(len(GAMMA)):
            if eps is None:
                We[i] = RHO*EPS[i]**(2/3)*diameter**(5/3)/(GAMMA[i]*C_eps**(2/3))
            else:
                We[i] = RHO*eps**(2/3)*diameter**(5/3)/(GAMMA[i]*C_eps**(2/3))
    elif gamma is not None and eps is not None:
        We = RHO*eps**(2/3)*diameter**(5/3)/(gamma*C_eps**(2/3))
    elif gamma is not None and eps is None:
        We = RHO*EPS**(2/3)*diameter**(5/3)/(gamma*C_eps**(2/3))
    return We

### Weber number of the simulations:
We = np.array([WE(16, GAMMA[i], EPS[i]) for i in range(len(EPS))])

def Tb(d, eps=None):
    '''
    Caracteristic time scale of eddies of size d.
    If epsilon, the dissipation rate, is not given, the caracteristic time is
    computed for every value of epsilon used in the simulations.
    '''
    if eps is None:
        return EPS**(-1/3)*d**(2/3)
    else:
        return eps**(-1/3)*d**(2/3)

def Tn(l, d, gamma=None):
    '''
    Period of the mode l of oscillation of a sphere of diameters d.
    If a value for gamma is not given the period is computed for every value of
    gamma used in the simulations.
    '''
    if gamma is None:
        return 1/fi.fn(l,GAMMA,d, rho=RHO)
    else:
        return 1/fi.fn(l,gamma,d,rho=RHO)
