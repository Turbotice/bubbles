# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.spatial import cKDTree
import numpy as np

def voronoi(interf_points,points):

    voronoi_kdtree = cKDTree(interf_points) #fonction qui trace un diagramme de Voronoi. 
    #Ce diagrame permet de crée des régions où sont regorupés tout les points les points proches 
    #d'un point de l'interface. Ça permet de calculer la distance des points à l'interface
    test_point_dist, test_point_regions = voronoi_kdtree.query(points)
    
    return test_point_dist, test_point_regions


def gaussian(x, sig): #fonction de Gauss utlisé pour la normalisation et le calcul des poids de chaque points
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-1/2*(x/sig)**2)

def interpol_gauss(x,dv,interf_points,points,d,sig): #Interpolation gaussienne des données
    
    dist = np.linspace(0, d, 100)
    
    test_point_dist, test_point_regions = voronoi(interf_points,points)
    weights = gaussian(test_point_dist -dist[:, np.newaxis], sig)
    
    norm = np.sum(weights*dv,axis=1)
    x_interp = np.sum(x*weights*dv,axis=1)/norm
    
    return x_interp
    
    

