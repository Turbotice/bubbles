#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:18:58 2023

@author: alienor

Improved version of the previous file. 

"""
import numpy as np
#import scipy as scipy
# from scipy.interpolate import interp1d
from scipy.spatial import SphericalVoronoi
from scipy.optimize import newton
from scipy.spatial import cKDTree

# from .. import toolsA as tools
#from stephane.sdeform.spherical import Area_voronoi, real_ylm #coef_harmonics, 
from spherical import Area_voronoi, real_ylm #coef_harmonics, 


def recenter(X, Xcenter, L):
    '''
    Modify X in place to follow the center of mass according to the previous 
    Xcenter in a box of size L. 
    Xcenter is also modified in place. 
    
    X = np.transpose([x, y, z])
    X.shape = (nbr of points, dimension)
    Xn next position to recenter.
    Xp previous position. 
    Xcenter, dimension 3. up until now Xcenter
    Xcenter.shape = (dimension,)
    L: int or float, box size.
    '''

    np.mod(X - Xcenter, L, out=X)
    X -= L/2
    Xcenter +=  np.mean(X, axis=0)
    
# x = np.linspace(-1,1, 10)
# L = 20
# X0 = np.transpose([x,x,x]) + L/2
# Xc = np.zeros(3)
# R = np.cumsum((np.random.random([10**5,3])+1)/3, axis=1)
# for i in range(R.shape[1]):
#     X = X0 + R[i]
#     recenter(X, Xc, L)
# print(X)
# print(Xc)

# theta = np.linspace(0, np.pi, 100)
# phi = np.linspace(0, 2*np.pi, 100)
# r = 7
# x = r*np.sin(theta)*np.cos(phi) + 0.05
# y = r*np.sin(theta)*np.sin(phi) - 0.3
# z = r*np.cos(theta) + 0.1
# X0 = np.transpose([x, y, z])

def coeff00(p, A, theta, phi):
    ''' 
    Given the coordinates of the interface p = [X, Y, Z].transpose(), the 
    Voronoi area A and the angles theta and phi, compute the coefficient of the
    l=0, m=0 in the spherical harmonic decomposition.
    '''
    Y = np.linalg.norm(p, axis=1)
    return np.sum(Y*real_ylm(0, 0, phi, theta)*A)


# def coeff11_shifted_x0(x0, p, A, theta, phi):
#     ''' 
#     Given the coordinates of the interface, compute the l=1, m=1 coefficient of
#     the spherical harmonic decomposition when the points are recentered around 
#     x_center = [x0, 0, 0].
#     Arguments:
#     -coordinates of the interface are p = np.array([X, Y, Z]).transpose(),
#     -the Voronoi area A
#     - angles theta and phi
#     '''
#     # p0 = np.zeros_like(p)
#     # p0[:, 0] = x0
#     # ptilde = p-p0
#     p0 = p - np.ones_like(p)*np.array([x0, 0, 0])

#     c00 = coeff00(p0, A, theta, phi) #remove the contribution from the 0th harmonic
#     Y = np.linalg.norm(p0, axis=1) - c00*real_ylm(0, 0, phi, theta)
#     return np.sum(Y*real_ylm(1, 1, phi, theta)*A)

# def coeff1m1_shifted_y0(y0, p, A, theta, phi):
#     ''' 
#     Given the coordinates of the interface, compute the l=1, m=-1 coefficient of
#     the spherical harmonic decomposition when the points are recentered around 
#     x_center = [0, y0, 0].
#     Arguments:
#     -coordinates of the interface are p = np.array([X, Y, Z]).transpose(),
#     -the Voronoi area A
#     - angles theta and phi
#     '''
#     # p0 = np.zeros_like(p)
#     # p0[:, 1] = y0
#     # ptilde = p-p0
#     p0 = p - np.ones_like(p)*np.array([0, y0, 0])

#     c00 = coeff00(p0, A, theta, phi) #remove the contribution from the 0th harmonic
#     Y = np.linalg.norm(p0, axis=1) - c00*real_ylm(0, 0, phi, theta)
#     return np.sum(Y*real_ylm(-1, 1, phi, theta)*A)

# def coeff10_shifted_z0(z0, p, A, theta, phi):
#     ''' 
#     Given the coordinates of the interface, compute the l=1, m=0 coefficient of
#     the spherical harmonic decomposition when the points are recentered around 
#     x_center = [0, 0, z0].
#     Arguments:
#     -coordinates of the interface are p = np.array([X, Y, Z]).transpose(),
#     -the Voronoi area A
#     - angles theta and phi
#     '''
#     # p0 = np.zeros_like(p)
#     # p0[:, 2] = z0
#     # p0 = p-p0
#     p0 = p - np.ones_like(p)*np.array([0, 0, z0])
#     c00 = coeff00(p0, A, theta, phi) #remove the contribution from the 0th harmonic
#     Y = np.linalg.norm(p0, axis=1) - c00*real_ylm(0, 0, phi, theta)
#     return np.sum(Y*real_ylm(0, 1, phi, theta)*A)

def coeff1i_shifted(translation, m, p, A, theta, phi):
    '''
    Compute the spherical coefficient of the harmonic (1, m), in a translated 
    frame. 
    If m = -1, translation along y. 
    If m = 0, translation along z. 
    If m = 1, translation along x.

    Parameters
    ----------
    translation : TYPE
        DESCRIPTION.
    m : int
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    shift = np.zeros(p.shape[1])
    shift[(m + 2)%3] = translation
    p0 = p - np.ones_like(p)*shift
    c00 = coeff00(p0, A, theta, phi) #remove the contribution from the 0th harmonic
    Y = np.linalg.norm(p0, axis=1) - c00*real_ylm(0, 0, phi, theta)
    
    return np.sum(Y*real_ylm(m, 1, phi, theta)*A)



# def find_centernomode1(p, A, theta, phi):
#     try:
#         dy = newton(coeff1m1_shifted_y0, 0, args=(p,A, theta, phi))
#     except:
#         # print('no cvg y')
#         x = np.linspace(-5, 5, 300)
#         val = [coeff1m1_shifted_y0(xi, p, A, theta, phi) for xi in x]
#         dy = x[np.argmin(np.abs(val))]
#     try:
#         dz = newton(coeff10_shifted_z0, 0, args=(p,A, theta, phi))
#     except:
#         # print('no cvg z')
#         x = np.linspace(-5, 5, 300)
#         val = [coeff10_shifted_z0(xi, p, A, theta, phi) for xi in x]
#         dz = x[np.argmin(np.abs(val))]        

#     try:
#         dx = newton(coeff11_shifted_x0, 0, args=(p, A, theta, phi))
#     except:
#         # print('no cvg x')
#         x = np.linspace(-5, 5, 300)
#         val = [coeff11_shifted_x0(xi, p, A, theta, phi) for xi in x]
#         dz = x[np.argmin(np.abs(val))] 
    
#     return np.array([dx, dy, dz])


def find_centerwomode1(p, A, theta, phi):
    '''
    More compact version of the above function.

    Parameters
    ----------
    p : np.array
        Vector of coordinates. p = np.tranpose([x, y, z]).
    A : np.array
        Voronoi area.
    theta : np.array
        theta angle.
    phi : np.array
        phi angle.

    Returns
    -------
    displacement : np.array
        DESCRIPTION.

    '''
    displacement = np.zeros(3)
    
    for m in range(-1, 2):
        try:
            displacement[(m + 2)%3] = newton(coeff1i_shifted, 0, \
                                             args=(m, p, A, theta, phi), tol=1e-8, maxiter=40)
        except: 
            x = np.linspace(-5, 5, 300)
            val = [coeff1i_shifted(xi, m, p, A, theta, phi) for xi in x]
            displacement[(m + 2)%3] = x[np.argmin(np.abs(val))]
            
    return displacement

def position_center(X, eps=2e-5, N_itermax=80):
    
    # p_c = np.zeros(3)

    norm = np.linalg.norm(X, axis=1, keepdims=True)
    #remove spurious interface
    indok = norm<30.
    X = X[indok[:,0]]
    # X, norm = X[indok[:,0]], norm[indok[:,0]]
    p_c = np.mean(X, axis=0)
    X -= p_c
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    
    Xnorm = X/norm
    radii = np.linalg.norm(Xnorm, axis=1)

    max_discrepancy = np.abs(radii -1).max()
    
    #points too close to each other
    kd_tree = cKDTree(Xnorm)
    pairs = kd_tree.query_pairs(r=2.*max_discrepancy)
    indu = np.full(len(Xnorm[:,0]), True)
    for (i, j) in pairs:
        indu[i] = False
    
    X, norm, Xnorm = X[indu], norm[indu], Xnorm[indu]
    
    sv = SphericalVoronoi(Xnorm, 1., [0., 0., 0.], threshold=1.5*max_discrepancy)
    sv.sort_vertices_of_regions()
    A = Area_voronoi(sv)
    A /= 4*np.pi
    
    theta = np.arccos(Xnorm[:,2])
    phi = np.arctan2(X[:,1], X[:,0])

    displacement = find_centerwomode1(X, A, theta, phi)/2
    
    index = 1

    while np.linalg.norm(displacement)>eps and index<N_itermax:
        # print(np.linalg.norm(displacement))
        p_c += displacement

        X -= displacement
        
        # r, theta, phi = tools.spherical(*X.T)
        norm = np.linalg.norm(X, axis=1, keepdims=True)

        # try:
        #     sv = SphericalVoronoi(X/norm, 1, [0, 0, 0], threshold=1e-12)
        # except:
        Xnorm = X/norm
        radii = np.linalg.norm(Xnorm, axis=1)
        max_discrepancy = np.abs(radii -1).max()
        sv = SphericalVoronoi(Xnorm, 1, [0, 0, 0], threshold=1.5*max_discrepancy)
            
        sv.sort_vertices_of_regions()
        A = Area_voronoi(sv)
        A /= 4*np.pi

        theta = np.arccos(Xnorm[:,2])
        phi = np.arctan2(X[:,1], X[:,0])

        displacement = find_centerwomode1(X, A, theta, phi)/2
        index += 1
    # for m in range(-1, 2):
    #     print(coeff1i_shifted(0, m, X, A, theta, phi))
        
    return p_c, A, index, norm, theta, phi


