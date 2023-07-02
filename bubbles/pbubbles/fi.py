#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:30:47 2019

@author: Alienor

Package to analyse the data given by the files fix, fiy, fiz.
Include the following functions :
    
    #load, combine data
    -coordinates_interface(dir)
    -combine_data(names)
    -interface(dir)
    
    #Extend data on a regularly space on theta and phi
    -extend_data(r, theta, phi)
    
    #Recenter the data
    -recenter(X, L) #by following the bubble
    -load_interface(dir)
    -load_interface_centered(dir, L=120)
    
    #Center the data where the 1st harmonic is null
    -compute_position_center(dir, L=120, Nitermax=80, eps=2e-5)
    -position_center(x, y, z, eps=2e-5, N_itermax=40)
    -find_centernomode1(p, A, theta, phi)
    -coeff00(p, A, theta, phi)
    -coeff11_shifted_x0(x0, p, A, theta, phi)
    -coeff1m1_shifted_y0(y0, p, A, theta, phi)
    -coeff10_shifted_z0(z0, p, A, theta, phi)
    
    #when the data are centered at the point where there is no 1st harmonic 
    compute the fields of interest
    -zeta_rms_corrected3(dir)
    -compute_spherical_decomposition3(dir, nmax=6, L=120)
    
    #others
    -fourier_transform(coeff, time, Npoints=1e4, omega=False)
    -fn(l,gamma,d,rho=1)
"""


import numpy as np
import glob
from . import dropbub as db
from . import toolsA as tools
import scipy as scipy
from stephane.sdeform.spherical import Area_voronoi, coef_harmonics, real_ylm
from scipy.interpolate import interp1d
from collections import defaultdict



def coordinates_interface(dir):
    
    """
    Takes in argument the direction of the fi* files giving the coordinate
    of the interface, in the form 'direction'.
    Return 3 arrays, (xdata, ydata, zdata) which are just the "sum" of the fi*. 
    For each file for each step the list of the coordinates is extracted and
    concatenated to the other values from the other files.
    Because of the adaptive meshgrid the number of points describing the 
    inteface at each step varies.
    
    Structure of the output:
    xdata[i,0]: time at step i
    xdata[i,1]: list of the x coordinates of the interface at step i, the 
    length of this list varies for each step.
    """
    
    #import the data
    names_x = sorted(glob.glob(dir+'/fix*.dat'))
    names_y = sorted(glob.glob(dir+'/fiy*.dat'))
    names_z = sorted(glob.glob(dir+'/fiz*.dat'))
    
    xdata = combine_data(names_x)
    print('success for xdata')
    ydata = combine_data(names_y)
    print('success for ydata')
    zdata = combine_data(names_z)
    print('success for zdata')
    
    return (xdata, ydata, zdata)

def combine_data(names):
    
    """
    Given the list of names of the fi* files, extracts the coordinates of the
    interface for each step and combines them.
    The list of files should contains only files with the same coordinate 
    x, y or z.
    
    Returns an array containing the coordinates of the interface for each step.
    
    Structure of the output:
    data[i,0]: time at step i
    data[i,1]: list of the x coordinates of the interface at step i, the length
    of this list varies for each step.
    """

    data = np.array([line.split() for line in open(names[0])])
    
    i = 0
    while i<len(names) and type(data[0])!=list:
        i += 1
        data = np.array([line.split() for line in open(names[i])])
 
    for name in names[i+1:]:
        temporary = np.array([line.split() for line in open(name)])
        if type(temporary[0])==list: ##allow to remove all empty files
            for i in range(len(temporary)):
                temporary[i].pop(0)     #remove the time
            
            n = min(len(data), len(temporary))
            data = data[:n]
            temporary = temporary[:n]
    
            data += temporary #add the values
            
    for elm in data:   #transform the strings into floats
        for i in range(len(elm)):
            elm[i] = float(elm[i]) 
            
    return (data)

def interface(dir):
    """
    Given the direction of the folder containing the fi_x, fi_y, fi_z files
    containing the list of the x, y, z coordinates of the interface for each
    time step returns the spherical coordinates of the interface in the frame 
    centered at the center of the bubble.
    
    Returns:
        -t: array containing the time for each step
        -r: array containing the list of the r coordinates for each step
        -r_mean: array containing the mean value of the radius for each step
        computed from the volume of the bubble given by the file 'bubbles.dat'
        -theta: array containing the list of the theta angles for each point of
        the interface at each step
        -phi: array containing the list of the phi angles for each point of the
        interface at each step
    
    """
    #load the coordinates of the interface
    xdata = np.array([line.split() for line in open(dir+'fi_x.dat')])
    ydata = np.array([line.split() for line in open(dir+'fi_y.dat')])
    zdata = np.array([line.split() for line in open(dir+'fi_z.dat')])
    
    for i in range(len(xdata)):
        for j in range(len(xdata[i])):
            xdata[i][j] = float(xdata[i][j])
            ydata[i][j] = float(ydata[i][j])
            zdata[i][j] = float(zdata[i][j])
    
    bubbles = db.import_bubbles(dir)
    
    #compute the mean radius as a function of time
    #the condition on the volume of the bubble allows to exclude the unphysical 
    #bubbles
    mask = np.where(bubbles[:,3]>1900)
    
    r_mean = (3/(4*np.pi)*bubbles[:,3][mask])**(1/3)
    
    #compute the coordinate of the center of the bubble
    
    x_center = bubbles[:,4][mask]  
    y_center = bubbles[:,5][mask]
    z_center = bubbles[:,6][mask]
    
#    x_center = [np.mean(xdata[i]) for i in range(len(xdata))]
#    y_center = [np.mean(ydata[i]) for i in range(len(ydata))]
#    z_center = [np.mean(zdata[i]) for i in range(len(zdata))]
    
    
    t_max = min(len(xdata),len(x_center))
    
    #compute the x, y, z coordinates in the frame centered at the center of the 
    #bubble
    x_pos = [np.array(xdata[i][1:])-x_center[i] for i in range(t_max)]
    y_pos = [np.array(ydata[i][1:])-y_center[i] for i in range(t_max)]
    z_pos = [np.array(zdata[i][1:])-z_center[i] for i in range(t_max)]
    
    #compute the spherical coordinates
    r, theta, phi,t = [],[],[],[]
    
    for i in range(len(x_pos)):
        out = tools.spherical(x_pos[i],y_pos[i],z_pos[i])
        r.append(out[0])
        theta.append(out[1])
        phi.append(out[2])
        t.append(xdata[i][0])
    
    return t, r, r_mean, theta, phi

def extend_data(r,theta,phi):
    
    '''
    Given a set of non organized values of r, theta and phi defined with the 
    physical conventions, extends the domain in theta and phi to avoid the 
    problem with the borders if then you need to interpolate the dataset. 
    The extension respects the boundary conditions of a sphere:
        r(theta, phi) = r(theta, phi + 2pi)
        r(theta=pi, phi) = constant
        r(theta=0, phi) = constant
    
    Inputs: r,theta,phi
    -theta is the colatitude, it has to be betweeen 0 and pi
    -phi is the longitude, it has to be between 0 and 2pi
    -r is the value of the radius for each theta and phi
    r, theta and phi are 1D arrays.
    
    Returns: new_r, new_theta, new_phi
        -new_theta: the extended colatitude now between -dtheta and pi+dtheta
        -new_phi: the extended longitude now between -pi and 3pi
        -new_r: the value of the radius for each value of new_theta and new_phi
    new_r, new_theta, new_phi are 1D arrays
    '''

    ext_r = [elm for elm in r]
    ext_theta = [elm for elm in theta]
    ext_phi = [elm for elm in phi]
    ind_theta_min = np.where(theta==np.min(theta))[0][0]
    ind_theta_max = np.where(theta==np.max(theta))[0][0]
    dtheta = np.pi/len(theta)

    
    for i in range(len(r)):
        if phi[i]>=np.pi:
            ext_r.append(r[i])
            ext_phi.append(phi[i]-2*np.pi)
            ext_theta.append(theta[i])

            #add two lines at the bottom of the domain
            ext_r.append(r[ind_theta_min])
            ext_r.append(r[ind_theta_min])
            ext_theta.append(0)
            ext_theta.append(-dtheta)
            ext_phi.append(phi[i])
            ext_phi.append(phi[i])

            ext_r.append(r[ind_theta_min])
            ext_r.append(r[ind_theta_min])
            ext_theta.append(0)
            ext_theta.append(-dtheta)
            ext_phi.append(phi[i]-2*np.pi)
            ext_phi.append(phi[i]-2*np.pi)
            
        elif phi[i]<np.pi:
            ext_r.append(r[i])
            ext_phi.append(phi[i]+2*np.pi)
            ext_theta.append(theta[i])            
            
            #add two lines at the top of the domain
            ext_r.append(r[ind_theta_max])
            ext_r.append(r[ind_theta_max])
            ext_theta.append(np.pi)
            ext_theta.append(np.pi+dtheta)
            ext_phi.append(phi[i])
            ext_phi.append(phi[i])

            ext_r.append(r[ind_theta_max])
            ext_r.append(r[ind_theta_max])
            ext_theta.append(np.pi)
            ext_theta.append(np.pi+dtheta)
            ext_phi.append(phi[i]+2*np.pi)
            ext_phi.append(phi[i]+2*np.pi)
    
    return ext_r, ext_theta, ext_phi
    
def recenter(X, L):
    '''
    Given X, the coordinates of the interface at all times
    in a squared box of size L with periodic boundary conditions, recenter the
    coordinates.
    X.shape =  (Number of time steps, 2)
    X[i][0] = ti #time at step i
    X[i][1] = [xij] #list of x coordinates of the interface at step i
    Be careful, len(X[i][1]) may vary.
    '''
    time = np.zeros(len(X))
    x_translated = np.zeros_like(X)
    x_translated[0] = np.asarray(X[0][1:]) - L/2
    time[0] = X[0][0]
    #total displacement of the bubble
    displacement = np.zeros(len(X))
    for i in range(1, len(X)):
        time[i] = X[i][0]
        x_translated[i] = np.mod(np.asarray(X[i][1:]) - displacement[i-1], L) - L/2
        displacement[i] = displacement[i-1] + np.mean(x_translated[i])

    return x_translated, time, displacement 

def load_interface(dir):
    try:
        xdata = np.load(dir+'fi_x.npz')['arr_0']
        ydata = np.load(dir+'fi_y.npz')['arr_0']
        zdata = np.load(dir+'fi_z.npz')['arr_0']
        
    except:
        xdata = np.array([line.split() for line in open(dir+'fi_x.dat')])
        ydata = np.array([line.split() for line in open(dir+'fi_y.dat')])
        zdata = np.array([line.split() for line in open(dir+'fi_z.dat')])
        
        for i in range(len(xdata)):
            for j in range(len(xdata[i])):
                xdata[i][j] = float(xdata[i][j])
                ydata[i][j] = float(ydata[i][j])
                zdata[i][j] = float(zdata[i][j])
        np.savez(dir + 'fi_x', xdata)
        np.savez(dir + 'fi_y', ydata)
        np.savez(dir + 'fi_z', zdata)
    return xdata, ydata, zdata

def load_interface_centered(dir, L=120):
    xdata, ydata, zdata = load_interface(dir)
    
    print('data loaded')
    bubbles = db.import_bubbles(dir)

    #recenter the data
    xdata, time, dx = recenter(xdata, L)
    ydata, time, dy = recenter(ydata, L)
    zdata, time, dz = recenter(zdata, L)
    
        #compute the mean radius as a function of time
    #the condition on the volume of the bubble allows to exclude the unphysical 
    #bubbles
    mask = np.where(bubbles[:,3]>1900)
    
    #compute the coordinate of the center of the bubble
#    r_mean = (3/(4*np.pi)*bubbles[:,3][mask])**(1/3)
    
    x_center = bubbles[:,4][mask]  
    y_center = bubbles[:,5][mask]
    z_center = bubbles[:,6][mask]
    
    t_max = min(len(xdata),len(x_center))
    
    #remove the displacement of the bubble to the center of mass
    x_center[0] -= L/2
    y_center[0] -= L/2
    z_center[0] -= L/2
    x_center[1:t_max] = np.mod(x_center[1:t_max] - dx[:t_max-1], L) - L/2
    y_center[1:t_max] = np.mod(y_center[1:t_max] - dy[:t_max-1], L) - L/2
    z_center[1:t_max] = np.mod(z_center[1:t_max] - dz[:t_max-1], L) - L/2
    

    
    
    x_pos = xdata[:t_max][1:]
    y_pos = ydata[:t_max][1:]
    z_pos = zdata[:t_max][1:]

    return x_pos, y_pos, z_pos
    

def zeta_rms_corrected3(dir):
    '''
    Compute the standard deviation of the radius of the bubble. Takes into
    account the non uniform repartition of points at the interface by weighting
    each point by its Voronoi area. Data are translated so that they are
    centered at the point where the 1st spherical harmonic is null.
    
    '''
    #load the coordinates of the interface that have been recentered
    dico = tools.load_dictionary(dir + 'data_centered')

    zeta_rms = np.zeros(len(dico['t']))

    for i in range(len(dico['t'])):
        A = dico['A'][i]
        p = dico['p'][i]
        x, y, z = p[:,0], p[:,1], p[:,2]

        #compute r at each point
        r = np.sqrt(x**2 + y**2 + z**2)
        #compute the mean radius
        r_mean = np.sum(A*r)
        #compute the standard deviation of the radius
        zeta_rms[i] = np.sqrt(np.sum(A*(r - r_mean)**2))

    time = dico['t']

    return time, zeta_rms


  
def coeff00(p, A, theta, phi):
    ''' 
    Given the coordinates of the interface p = [X, Y, Z].transpose(), the 
    Voronoi area A and the angles theta and phi, compute the coefficient of the
    l=0, m=0 in the spherical harmonic decomposition.
    '''
    Y = np.linalg.norm(p, axis=1)
    return np.sum(Y*real_ylm(0, 0, phi, theta)*A)

def coeff11_shifted_x0(x0, p, A, theta, phi):
    ''' 
    Given the coordinates of the interface, compute the l=1, m=1 coefficient of
    the spherical harmonic decomposition when the points are recentered around 
    x_center = [x0, 0, 0].
    Arguments:
    -coordinates of the interface are p = np.array([X, Y, Z]).transpose(),
    -the Voronoi area A
    - angles theta and phi
    '''
    p0 = np.zeros_like(p)
    p0[:, 0] = x0
    ptilde = p-p0
    c00 = coeff00(ptilde, A, theta, phi) #remove the contribution from the 0th harmonic
    Y = np.linalg.norm(ptilde, axis=1) - c00*real_ylm(0, 0, phi, theta)
    return np.sum(Y*real_ylm(1, 1, phi, theta)*A)

def coeff1m1_shifted_y0(y0, p, A, theta, phi):
    ''' 
    Given the coordinates of the interface, compute the l=1, m=-1 coefficient of
    the spherical harmonic decomposition when the points are recentered around 
    x_center = [0, y0, 0].
    Arguments:
    -coordinates of the interface are p = np.array([X, Y, Z]).transpose(),
    -the Voronoi area A
    - angles theta and phi
    '''
    p0 = np.zeros_like(p)
    p0[:, 1] = y0
    ptilde = p-p0
    c00 = coeff00(ptilde, A, theta, phi) #remove the contribution from the 0th harmonic
    Y = np.linalg.norm(ptilde, axis=1) - c00*real_ylm(0, 0, phi, theta)
    return np.sum(Y*real_ylm(-1, 1, phi, theta)*A)

def coeff10_shifted_z0(z0, p, A, theta, phi):
    ''' 
    Given the coordinates of the interface, compute the l=1, m=0 coefficient of
    the spherical harmonic decomposition when the points are recentered around 
    x_center = [0, 0, z0].
    Arguments:
    -coordinates of the interface are p = np.array([X, Y, Z]).transpose(),
    -the Voronoi area A
    - angles theta and phi
    '''
    p0 = np.zeros_like(p)
    p0[:, 2] = z0
    ptilde = p-p0
    c00 = coeff00(ptilde, A, theta, phi) #remove the contribution from the 0th harmonic
    Y = np.linalg.norm(ptilde, axis=1) - c00*real_ylm(0, 0, phi, theta)
    return np.sum(Y*real_ylm(0, 1, phi, theta)*A)

def find_centernomode1(p, A, theta, phi):
    x = np.linspace(-20, 20, 300)

    c1m1, c10, c11 = [], [], []
    for i in range(len(x)):
        c1m1.append(coeff1m1_shifted_y0(x[i], p, A, theta, phi))
        c10.append(coeff10_shifted_z0(x[i], p, A, theta, phi))
        c11.append(coeff11_shifted_x0(x[i], p, A, theta, phi))
    
    f1m1 = interp1d(x, c1m1, fill_value='extrapolate')
    f10 = interp1d(x, c10, fill_value='extrapolate')
    f11 = interp1d(x, c11, fill_value='extrapolate')
    try:
        y0 = np.max(np.where(f1m1(x)>0))
        dy = scipy.optimize.newton(f1m1, x[y0])
    except:
        val = f1m1(x)
        dy = x[np.where(val==min(np.abs(val)))]
    try:
        z0 = np.max(np.where(f10(x)>0))
        dz = scipy.optimize.newton(f10, x[z0])
    except:
        val = f10(x)
        dz = x[np.where(val==min(np.abs(val)))]
    try:
        x0 = np.max(np.where(f11(x)>0))
        dx = scipy.optimize.newton(f11, x[x0])
    except:
        val = f11(x)
        dx = x[np.where(val==min(np.abs(val)))]
    
    return dx, dy, dz

def position_center(x, y, z, eps=2e-5, N_itermax=40):
    
    xi, yi, zi = x, y, z
    x_c, y_c, z_c = [0], [0], [0]
    
    #compute the Voronoi diagram for each point
    r, theta, phi = tools.spherical(xi, yi, zi)
    p = np.transpose([xi, yi, zi])
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    pnorm = p/norm
    
    sv = scipy.spatial.SphericalVoronoi(pnorm, 1, [0, 0, 0])
    sv.sort_vertices_of_regions()
    A = Area_voronoi(sv)
    A /= 4*np.pi
    dx, dy, dz = find_centernomode1(p, A, theta, phi)
    i = 1


    while np.sqrt(dx**2 + dy**2 + dz**2)/2>eps and i<N_itermax:

        x_c.append(x_c[-1] + dx/2)
        y_c.append(y_c[-1] + dy/2)
        z_c.append(z_c[-1] + dz/2)
        
        xi -= dx/2
        yi -= dy/2
        zi -= dz/2
        
        r, theta, phi = tools.spherical(xi, yi, zi)
        #compute the Voronoi diagram for each point
        p = np.transpose([xi, yi, zi])
        norm = np.linalg.norm(p, axis=1, keepdims=True)
        pnorm = p/norm
        
        sv = scipy.spatial.SphericalVoronoi(pnorm, 1, [0, 0, 0])
        sv.sort_vertices_of_regions()
        A = Area_voronoi(sv)
        A /= 4*np.pi
        dx, dy, dz = find_centernomode1(p, A, theta, phi)
        i += 1
        
    x_c.append(x_c[-1] + dx/2)
    y_c.append(y_c[-1] + dy/2)
    z_c.append(z_c[-1] + dz/2)
        
    return x_c, y_c, z_c, p, A, i, theta, phi

def compute_position_center(dir, L=120, Nitermax=80, eps=2e-5):
    '''
    Compute a dicitonary containing the interface coordinates recentered at the
    point where the 1st spherical harmonic is null.
    Keys of the dictionary:
    -d['t']: list of the times that have been analyzed
    -d['p']: list of arrays containing for each time the coordinates of the
    interface. 
    So at step i d['p'][i] = np.array([X, Y, Z]).transpose() where X, Y,
    Z are the list of the interface coordinates at step i. 
    So d['p'][i][j] = [xj, yj, zj]
    -d['A']: List of the Voronoi areas at each time. 
    So at step i d['A'][i] is the 1d array containing all the areas for each
    point.
    -d['theta'], d['phi']: List of the angles theta and phi for each point at
    each step. The angles are defined using the physical conventions (longitude
    and colatitude)
    -d['cvg?']: List of boolean. The value is True if at the end of the
    computation the number of iterations is smaller that the maximum number of
    iteration Nitermax. Else it is False.

    Compute a dictionary containing for all times the series of cvg of the
    centers p; A; xi, yi, zi
    '''
    name = dir + 'data_centered'
    try:#check if the dictionary was already computed.
        d = tools.load_dictionary(name)
    except:#create a new one.
        d = defaultdict(dict)
        d['p'] = []
        d['A'] = []
        d['theta'], d['phi'] = [], []
        d['t'] = []
        d['cvg?'] = []

    #load the coordinates of the interface
    xdata, ydata, zdata = load_interface(dir)

    t_max = min(len(xdata), len(ydata), len(zdata))
    # naively recenter the data by following the bubble
    # necessary because of the periodic bounary conditions
    xdata, time, *_ = recenter(xdata[:t_max], L)
    ydata, *_ = recenter(ydata[:t_max], L)
    zdata, *_ = recenter(zdata[:t_max], L)
    print('data loaded')
    
    #stop the calculation when the bubble breaks
    bubbles = db.import_bubbles(dir)
    tv, vol = db.volumes(bubbles)
    if len(vol.shape)>1 and vol.shape[1]>1:
        ind = np.min(np.where(vol[:,-2]>0))
        tvmax = tv[ind]
        indmax = np.max(np.where(time<=tvmax))
    else:
        indmax = len(time)
    #start the computation from the 5th step because before not enough points on
    # the interface
    # compute only every 5 points because its enough to follow the evolution of 
    # the deformation. Could probably compute even less points.
    for i in range(5, indmax, 5):
        if not time[i] in d['t']:#check that it was not computed before
          # should add a condition on d['cvg?'] so recompute the points where
          # the convergence was not reached
            x, y, z = xdata[i], ydata[i], zdata[i]
            #remove the spurious points
            r2 = x**2 + y**2 + z**2
            ind0 = np.where(r2<400)
            x, y, z = x[ind0], y[ind0], z[ind0]
            #compute the real position of the center
            x_ci, y_ci, z_ci, p, A, N_iter, theta, phi = position_center(x, y, z, N_itermax=Nitermax, eps=eps)
            #add the new point to the dictionary
            d['p'].append(p)
            d['A'].append(A)
            d['theta'].append(theta)
            d['phi'].append(phi)
            d['t'].append(time[i])
            if N_iter == Nitermax:#check the convergence
                try:#because before I did not have the argument d['cvg?']
                    d['cvg?'].append(False)
                except:
                    pass
                print('bad cvg at step ', i)
            elif N_iter<Nitermax:
                try:#because some dictionaries were created before I add this argument so d['cvg?'] does not exist
                    d['cvg?'].append(True)
                except:
                    pass
            if i%50==0:
                tools.save_dictionary(d, name)
            if i%500==0:
                print(i, 'steps ok')
    tools.save_dictionary(d, name)
    return d

def compute_spherical_decomposition3(dir, nmax=6, L=120):
    '''
    Given the coordinates of the interface of a bubble, computes the spherical
    decomposition. Takes into account the non uniform repartition of points at
    the interface by weighting each point by its Voronoi area. Define the center
    of the bubble as the position where the 1st harmonic is null.
    
    '''
    #load the coordinates of the interface that have been recentered
    dico = tools.load_dictionary(dir + 'data_centered')

    scoeff = np.zeros(len(dico['t']), dtype=object)
    ind, pb, PB = [], [], False
    for i in range(len(dico['t'])):
        try:
            scoeff[i] = coef_harmonics(dico['p'][i], dico['A'][i],
                  dico['theta'][i], dico['phi'][i], nmax=nmax)
            ind.append(i)
        except:
            pb.append(i)
            PB = True
    if PB:
        print('Problem for {} times'.format(len(pb)))
    time = np.array(dico['t'])
    scoeff = scoeff[ind]
    for i in range(len(scoeff)):
        scoeff[i]['t'] = time[i]
    return scoeff, pb


        
def fourier_transform(coeff, time, Npoints=1e4, omega=False):
    '''Compute the fourier transform of an irregularly spaced data sample.
    First do an interpolation on a regular grid, then compute the Fourier tranform.
    '''

    time = time - time[0]
    t, step = np.linspace(0, time[-1], Npoints, retstep=True)
    interp = scipy.interpolate.interp1d(time, coeff, fill_value='extrapolate')
    newcoeff = interp(t)
    fft = np.fft.fft(newcoeff)
    fft1d = np.fft.fftshift(fft)

    if omega:
        omega = 2*np.pi*np.fft.fftfreq(len(fft1d),step)
    else:
        omega = np.fft.fftfreq(len(fft1d),step)

    omega = np.fft.fftshift(omega)
    
    return fft1d, omega
    
    
    
def fn(l,gamma,d,rho=1):
    '''
    Frequency of the mode l of oscillation of a bubble of diameter d, surface
    tension gamma. The density of the liquid is rho. By default the value is 
    set to 1.
    '''
    return 1/(2*np.pi)*np.sqrt(8*(l-1)*(l+1)*(l+2)*gamma/(rho*d**3))
    
    
    
