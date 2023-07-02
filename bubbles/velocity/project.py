# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.spatial import cKDTree
import numpy as np


def compute_n(points,interf_points,region_index):
    
    normale=np.zeros([len(points),3])
    
    normale[:,:] = points[:,:] - interf_points[:,:][region_index]
    norm = np.sqrt(np.sum(normale[:,:]**2 , axis=1))
    normale /= norm[:,np.newaxis]
    
    return normale
    
    
def calculate_orthogonal_vectors(v): #calculer les deux vecteurs orthonormaux à un vecteur v
    
    # Générer un vecteur aléatoire de même dimension que v
    random_vector = np.random.randn(len(v))
    
    # Calculer le produit vectoriel entre v et le vecteur aléatoire
    cross_product = np.cross(v, random_vector)
    
    # Calculer le produit vectoriel entre v et le produit vectoriel précédent
    orthogonal_vector = np.cross(v, cross_product)
    
    # Normaliser les vecteurs
    cross_product_normalized = cross_product / np.linalg.norm(cross_product)
    orthogonal_vector_normalized = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    
    return cross_product_normalized, orthogonal_vector_normalized
    
    
