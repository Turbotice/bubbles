# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



# fonctions du projet de dissipation-bulles : voronoi, calcul des normales, distances, projections etc


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
    
def expo_decr(x,a,b,cte):
    return np.exp(a*x+b)+cte
    
params = np.zeros(9)

def surf(x, y, z):#equation of the surface
    global params
#     a2, a1, b2, b1, c2, c1, ab, ac, bc, d = params
    a1,b1 ,c1, ab , bc, ac , a2 , b2 , c2 = params
    return a2*x**2 + a1*x + b2*y**2 + b1*y + c2*z**2 + c1*z + ab*x*y + ac*x*z + bc*y*z

def gradient(x, y, z):#gradient of the surface 
    global params
#     a2, a1, b2, b1, c2, c1, ab, ac, bc, d = params
    a1, b1 ,c1, ab , bc, ac , a2 , b2 , c2 = params
    gx = 2*a2*x + a1 + ab*y+ ac*z
    gy = 2*b2*y + b1 + ab*x + bc*z
    gz = 2*c2*z + c1 + ac*x + bc*y
    return np.asarray([gx, gy, gz])

def system(XYZLamb, M):
    '''
    to solve:
    x - xM - lambda gx = 0
    y - yM - lambda gy = 0
    z - zM - lambda gz = 0
    surf(x, y, z) = 0
    
    ie: OM and gradient are colinear and O is on the surface.
    '''
    g = gradient(*XYZLamb[:3])
    p1 = M - XYZLamb[:3] - XYZLamb[3]*g
    p1 = np.append(p1, surf(*XYZLamb[:3]) )
    return p1
   
   
def interpolate_surf(interface_points,voronoi_kdtree):
#arguements : interface_points = coordonées des points de l'interface
#voronoi_kdtree = arbre de données à partir duquel est calculé le voronoi (cf voronoi)

#return dictionnaire avec les coefficients de l'équation de la surface locale pour chaque points de l'interface
#     calcule des coefficients d'interpolation de la surf pour tous les points de l'interface
    surface = {}
    for i , element in enumerate(interface_points):
        (distances,voisins) = voronoi_kdtree.query(element,k=20)# récupère les 20 points de l'interface les plus proche d'element
        vois = interface_points[voisins,:] # coordonnnées des voisins
        A = np.c_[vois[:,:], vois[:,0]*vois[:,1],vois[:,2]*vois[:,1],vois[:,0]*vois[:,2], vois[:,:]**2] 
        params,_,_,_ = scipy.linalg.lstsq(A, np.ones(vois.shape))
        params = params[:,0] #coefficient de l'équation la surface locale (dans le même ordre que dans la fonction surf)
        surface[i] = params
    return surface
    
    
    
def compute_dist_norm(points, interface_points, regions, surface):
#arguments:
#points = coord des points d'étude
#interface_points = coordonnées des points de l'interface
#regions = liste des indices des points de l'interface les plus proches pour le tableau "points" (cf voronoi)
#surface = dictionnaire des coefficients de l'équation de la surface locale pour chaque point de l'interface (résultat de la fonction interpolate_surf)

#return: distance = tableau des distances à l'interface pour chaque points, normale = tableau des vecteurs normales pour chaque points

# calcule de la distance à l'interface pour tous les points mis en arguments dans 'points'
    distance =  np.zeros(len(points))
    normale = np.zeros([len(points),3])
    for i , element in enumerate(points):
        indice_interface = regions[i] # indice du point de l'interface le plus proche d'element
        interf = interface_points[indice_interface] #coordonnées de ce point
        params = surface[indice_interface]
        out = fsolve(system, np.append(interf+1,1), args=(element), full_output=True)
        sol = out[0]
        normal = sol[:3]/np.linalg.norm(sol[:3])
        n_voronoi = element - interf
        distance[i] = np.sum(normal*n_voronoi)
        normale[i] = normal
    return distance, normale
    
    
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


def proj_vit(vitesse,normale):
    normv = np.abs(np.sum(vitesse[:]*normale[:],axis=1))
    
    v_n = np.zeros([len(normv),3])
    v_n[:,0] = normv*normale[:,0]
    v_n[:,1] = normv*normale[:,1]
    v_n[:,2] = normv*normale[:,2]
    
    tangv = np.linalg.norm(vitesse[:] - v_n[:],axis=1)
    
    return normv, tangv

def proj_grad_out(vitesse,normale,tangente1,tangente2):
    
    gradv_n=np.zeros([len(vitesse),3])
    gradv_n[:,0]=np.sum(vitesse[:,6:9]*normale[:,0:],axis=1)
    gradv_n[:,1]=np.sum(vitesse[:,9:12]*normale[:,0:],axis=1)
    gradv_n[:,2]=np.sum(vitesse[:,12:15]*normale[:,0:],axis=1)
    normgdn=np.sum(gradv_n[:,0:3]**2,axis=1)

    gradv_t1=np.zeros([len(vitesse),3])
    gradv_t1[:,0]=np.sum(vitesse[:,6:9]*tangente1[:,0:],axis=1)
    gradv_t1[:,1]=np.sum(vitesse[:,9:12]*tangente1[:,0:],axis=1)
    gradv_t1[:,2]=np.sum(vitesse[:,12:15]*tangente1[:,0:],axis=1)
    normgdt1=np.sum(gradv_t1[:,0:3]**2,axis=1)


    gradv_t2=np.zeros([len(vitesse),3])
    gradv_t2[:,0]=np.sum(vitesse[:,6:9]*tangente2[:,0:],axis=1)
    gradv_t2[:,1]=np.sum(vitesse[:,9:12]*tangente2[:,0:],axis=1)
    gradv_t2[:,2]=np.sum(vitesse[:,12:15]*tangente2[:,0:],axis=1)
    normgdt2=np.sum(gradv_t2[:,0:3]**2,axis=1)
    
    return normgdn , normgdt1 , normgdt2
    
    



    
    

