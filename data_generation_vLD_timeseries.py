import numpy as np
import pylab as pl
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import math

#==============================================================================
# Génération des données initiales
#==============================================================================

#permet de générer un jeu de DOA et de distances à partir d'un nombre de points, avec une incertitude de +/- 15° et +/- 15Nm
def data_threat(n_points):
    data_single_threat_DOA = np.linspace(0, 360 - 360 / n_points, n_points)
    data_single_threat_dist = np.linspace(0, 360 - 360 / n_points, n_points)
    data_single_threat_DOA = data_single_threat_DOA + (np.random.rand(len(data_single_threat_DOA))-0.5*np.ones((len(data_single_threat_DOA))))*15
    data_single_threat_dist = data_single_threat_dist + (np.random.rand(len(data_single_threat_dist))-0.5*np.ones((len(data_single_threat_dist))))*15
    rand.shuffle(data_single_threat_dist)
    data_single_threat = np.ones((len(data_single_threat_DOA), 2))
    data_single_threat[:,0] = data_single_threat_DOA
    data_single_threat[:,1] = data_single_threat_dist
    
    return data_single_threat


#permet d'actualiser la position des data en fonction du vecteur (X,Y) à leur appliquer
def data_threat_actualization(data, X_ref, Y_ref):
    
    for i in range(0,len(data)):
        if abs(data[i][0]) < 180:
            X_point = np.cos(data[i][0]*2*np.pi/360)*abs(data[i][1])
            Y_point = np.sin(data[i][0]*2*np.pi/360)*abs(data[i][1])
        else : 
            X_point = np.cos((data[i][0]-360)*2*np.pi/360)*abs(data[i][1])
            Y_point = np.sin((data[i][0]-360)*2*np.pi/360)*abs(data[i][1])
    
        X_point = X_point + X_ref
        Y_point = Y_point + Y_ref
        
        if X_point == 0 :
            X_point = 0.001
        data[i][0] = math.atan(Y_point/X_point)*180/np.pi
        if X_point <0:
            data[i][0] = data[i][0] + 180
        elif Y_point <0 : 
            data[i][0] = 360+ data[i][0]
        data[i][1] = np.sqrt(X_point*X_point + Y_point*Y_point)
        
         

    return data
#==============================================================================
# Fonctions de transfert des capteurs
#==============================================================================
# Cette fonction permet de générer les fonctions de transfert en DOA des capteurs
# n_points : le pas utilisé pour grider l'espace
# rand : les erreurs à utiliser : rand_range1 = biais, rand_range2 = bruit

def Define_Transfer_func_DOA (n_points, rand_range1, rand_range2):
    # en degrees de 0 à 360
    DOA_input=np.linspace(0,360-360/n_points,n_points)

    if rand_range1 !=0:
        for i in range(0,n_points):
            DOA_input[i]=DOA_input[i]+rand_range1*(0.5-np.sqrt(abs(np.sin(2*(i*2*3.14/n_points)))))

    if rand_range2 != 0:
        for i in range(0,n_points):
            DOA_input[i]=DOA_input[i]+rand.uniform(-rand_range2,rand_range2)

    return DOA_input

# Générer les points en sortie du capteur (ajout de biais uniformes sur les points)
def Transfer_func_DOA (data, y_reference_points,rand_range1, rand_range2):
    # en degrees de 0 à 360
    if len(y_reference_points)==0:
        A_input = np.linspace(0, 360 - 360 / n_points, n_points)
    else:
        x_reference_points = data[:,0]
        A_input = np.interp(np.linspace(0, 360 - 360 / n_points, n_points), x_reference_points, y_reference_points)

    if rand_range1!=0:
        for i in range(0, n_points):
            A_input[i] =  A_input[i] + rand.uniform(-rand_range1,rand_range1)

    if rand_range2 != 0:
        for i in range(1, n_points):
            A_input[i] =  A_input[i] + rand.uniform(-rand_range2, rand_range2)

    A_input=A_input%360

    return A_input

# Cette fonction permet de générer les fonctions de transfert en distance des capteurs
# n_points : le pas utilisé pour grider l'espace
# rand : les erreurs à utiliser : rand_range1 = biais, rand_range2 = bruit

def Define_Transfer_func_dist (n_points, rand_range1, rand_range2):
    #en Nm de 0 à max_distance
    dist_input=np.linspace(0,max_distance-max_distance/n_points,n_points)

    if rand_range1 != 0:
        for i in range(0,n_points):
            dist_input[i]=dist_input[i]+rand.uniform(1,rand_range1)

    if rand_range2 != 0:
        for i in range(1,n_points):
            dist_input[i]=dist_input[i]+rand.gauss(-rand_range2,rand_range2)
    return dist_input

# Générer les points en sortie du capteur
def Transfer_func_dist(data, y_reference_points, rand_range1, rand_range2):
    # en degrees de 0 à 360
    if len(y_reference_points) == 0:
        A_input = np.linspace(0, 360 - 360 / n_points, n_points)
    else:
        x_reference_points = data[:,1]
        A_input = np.interp(np.linspace(0, 360 - 360 / n_points, n_points), x_reference_points, y_reference_points)

    if rand_range1 != 0:
        for i in range(0, n_points):
            A_input[i] = A_input[i] + rand.uniform(-rand_range1, rand_range1)

    if rand_range2 != 0:
        for i in range(1, n_points):
            A_input[i] = A_input[i] + rand.uniform(-rand_range2, rand_range2)

    return A_input

#==============================================================================
# Génération des données des capteurs pour une ou deux sources
#==============================================================================
#Calculer les DOA des deux capteurs pour une seule source
def DOA_single_source(data1, data2):
    DOA_pair=np.ones((len(data1)*2,3))
    DOA_sensor1_1=Transfer_func_DOA(data1, DOA_ref_pts_sensor1,DOA_sensor1_rand_1, DOA_sensor1_rand_2)
    DOA_sensor1_2=Transfer_func_DOA(data2, DOA_ref_pts_sensor1,DOA_sensor1_rand_1, DOA_sensor1_rand_2)
    DOA_sensor2_1=Transfer_func_DOA(data1, DOA_ref_pts_sensor2,DOA_sensor2_rand_1, DOA_sensor2_rand_2)
    DOA_sensor2_2=Transfer_func_DOA(data2, DOA_ref_pts_sensor2,DOA_sensor2_rand_1, DOA_sensor2_rand_2)
    DOA_pair[:len(data1),0] = DOA_sensor1_2
    DOA_pair[len(data1):,0] = DOA_sensor1_1
    DOA_pair[:len(data1),1] = DOA_sensor2_2
    DOA_pair[len(data1):,1] = DOA_sensor2_1
    return DOA_pair

#Calculer les DOA des deux capteurs pour une deux sources
def DOA_two_sources(data1, data2):
    DOA_pair = np.zeros((len(data1)*2,3))
    DOA_sensor1_1=Transfer_func_DOA(data1, DOA_ref_pts_sensor1,DOA_sensor1_rand_1, DOA_sensor1_rand_2)
    DOA_sensor1_2=Transfer_func_DOA(data2, DOA_ref_pts_sensor1,DOA_sensor1_rand_1, DOA_sensor1_rand_2)
    DOA_sensor2_1=Transfer_func_DOA(data1, DOA_ref_pts_sensor2,DOA_sensor2_rand_1, DOA_sensor2_rand_2)
    DOA_sensor2_2=Transfer_func_DOA(data2, DOA_ref_pts_sensor2,DOA_sensor2_rand_1, DOA_sensor2_rand_2)
   
    DOA_pair[:len(data1),0] = DOA_sensor1_1
    DOA_pair[len(data1):,0] = DOA_sensor1_2
    DOA_pair[len(data1):,1] = DOA_sensor2_1
    DOA_pair[:len(data1),1] = DOA_sensor2_2
    return DOA_pair

# pour calculer les valeurs d angle
def modulo_neg(a):
    if a>=0:
        r=a%360
    else:
        r=(a+360)%360-360
    return r

# pour calculer les distances pour une seule source
def Dist_single_source(data1, data2):
    dist_pair = np.ones((len(data1)*2, 3))
    dist_sensor1_1 = Transfer_func_dist(data1, dist_ref_pts_sensor1, dist_sensor1_rand_1, dist_sensor1_rand_2)
    dist_sensor2_1 = Transfer_func_dist(data1, dist_ref_pts_sensor2, dist_sensor2_rand_1, dist_sensor2_rand_2)
    dist_sensor1_2 = Transfer_func_dist(data2, dist_ref_pts_sensor1, dist_sensor1_rand_1, dist_sensor1_rand_2)
    dist_sensor2_2 = Transfer_func_dist(data2, dist_ref_pts_sensor2, dist_sensor2_rand_1, dist_sensor2_rand_2)
    dist_pair[:len(data1), 0] = dist_sensor1_2
    dist_pair[:len(data1), 1] = dist_sensor2_2
    dist_pair[len(data1):, 0] = dist_sensor1_1
    dist_pair[len(data1):, 1] = dist_sensor2_1
    return dist_pair

# pour calculer les distances pour deux sources en fonction du paramètre bounded
def Dist_two_sources(data1, data2):
    dist_pair = np.zeros((len(data1)*2, 3))
    dist_sensor1_1 = Transfer_func_dist(data1, dist_ref_pts_sensor1, dist_sensor1_rand_1, dist_sensor1_rand_2)
    dist_sensor1_2 = Transfer_func_dist(data2, dist_ref_pts_sensor1, dist_sensor1_rand_1, dist_sensor1_rand_2)
    
    dist_sensor2_1 = Transfer_func_dist(data1, dist_ref_pts_sensor2, dist_sensor2_rand_1, dist_sensor2_rand_2)
    dist_sensor2_2 = Transfer_func_dist(data2, dist_ref_pts_sensor2, dist_sensor2_rand_1, dist_sensor2_rand_2)

    dist_pair[:len(data1), 0] = dist_sensor1_1
    dist_pair[len(data1):, 0] = dist_sensor1_2
    dist_pair[:len(data1), 1] = dist_sensor2_2
    dist_pair[len(data1):, 1] = dist_sensor2_1
    return dist_pair

#==============================================================================
# Génération des quadruplets de données
#==============================================================================
#Générer les quadruplets de données à partir des DOA et des distances
def Generate_quadruplets(data1, data2,single,as_delta):

    quadruplets_out=np.zeros((len(data1)*2,3,2))




    if single==1:

        quadruplets_out[:, :, 0] = DOA_single_source(data1, data2)
        quadruplets_out[:, :, 1] = Dist_single_source(data1, data2)
        quadruplets_out[:, 2, 0] = np.ones(len(data1)*2)

    else:
        quadruplets_out[:, :, 0] = DOA_two_sources(data1,data2)
        quadruplets_out[:, :, 1] = Dist_two_sources(data1,data2)


    if as_delta==1 :
        quadruplets_out[:, 1, :]=quadruplets_out[:, 1, :]-quadruplets_out[:,0, :]

        for i in range(n_points):

            if quadruplets_out[i, 1, 0] > 180:
                quadruplets_out[i, 1, 0] = quadruplets_out[i, 1, 0] - 360
            if quadruplets_out[i, 1, 0] <-180:
                quadruplets_out[i, 1, 0] = quadruplets_out[i, 1, 0] + 360
                

    return quadruplets_out

#pour générer les data
def Generate_data_quad(data1, data2):

    #generation of two sets of quadruplets flagued
    #shape is 3D (n_point , 3 for [2 sensors + flag_single/double] , 2 for DOA & dist)
    print("quadruplets generation")
    quad_single_as_delta = Generate_quadruplets(data1 , data2, 1, 1)
    quad_double_bounded_as_delta = Generate_quadruplets(data1, data2, 0, 1)
    print("generation over")

    #creating the x and y vectors for single and double sources
    x_single = np.zeros((n_points*2, len(quad_single_as_delta[0,:-1,0])*len(quad_single_as_delta[0,0,:])))
    y_single = np.zeros((n_points*2,1))
    x_double = np.zeros((n_points*2, len(quad_double_bounded_as_delta[0,:-1,0])*len(quad_double_bounded_as_delta[0,0,:])))
    y_double = np.zeros((n_points*2,1))
    print("x single : " + str(x_single.shape))
    print("x double : " + str(x_double.shape))
    x_single[:,:len(quad_single_as_delta[0,:-1,0])] = quad_single_as_delta[:,:-1,0] 
    x_single[:,len(quad_single_as_delta[0,0,:]):] = quad_single_as_delta[:,:-1,1] 
    y_single = quad_single_as_delta[:,2,0]
    x_double[:,:len(quad_double_bounded_as_delta[0,:-1,0])] = quad_double_bounded_as_delta[:,:-1,0] 
    x_double[:,len(quad_double_bounded_as_delta[0,0,:]):] = quad_double_bounded_as_delta[:,:-1,1] 
    y_double = quad_double_bounded_as_delta[:,2,0] 
    
    if(len(x_single[0,:])!=len(x_double[0,:])):
        print('data length error')
    #creating the final x and y vectors
    x = np.zeros((len(x_single[:,0])+len(x_double[:,0]),len(x_single[0,:])))
    x[:n_points*2] = x_single
    x[n_points*2:] = x_double
    y = np.zeros(len(y_single[:])+len(y_double[:]))
    y[:n_points*2] = y_single
    y[n_points*2:] = y_double
    
    
 
    
    return x,y


#__________________________________________________________

#==============================================================================
# Boucle principale
#==============================================================================
n_points = 2
n_steps = 360
timesteps = 5
timesteps = timesteps+1

DOA_sensor1_rand_1_fixed=40
DOA_sensor1_rand_2_fixed=3
DOA_sensor2_rand_1_fixed=15
DOA_sensor2_rand_2_fixed=1

DOA_sensor1_rand_1=0
DOA_sensor1_rand_2=5
DOA_sensor2_rand_1=0
DOA_sensor2_rand_2=5

dist_sensor1_rand_1_fixed=8
dist_sensor1_rand_2_fixed=0
dist_sensor2_rand_1_fixed=3
dist_sensor2_rand_2_fixed=0

dist_sensor1_rand_1=4
dist_sensor1_rand_2=0
dist_sensor2_rand_1=2
dist_sensor2_rand_2=0

max_distance=300
SF1=1.3
SF2=1.025
max_DOA_gap=360/n_steps + SF1*0.78*(DOA_sensor1_rand_1 + DOA_sensor1_rand_2 + DOA_sensor2_rand_1 + DOA_sensor2_rand_2 + DOA_sensor1_rand_1_fixed + DOA_sensor1_rand_2_fixed + DOA_sensor2_rand_1_fixed+ DOA_sensor2_rand_2_fixed)
max_dist_gap=max_distance/n_steps + SF2*(dist_sensor1_rand_1_fixed*2 + dist_sensor1_rand_2_fixed*2 + dist_sensor2_rand_1_fixed*2 + dist_sensor2_rand_2_fixed*2+dist_sensor1_rand_1*2 + dist_sensor1_rand_2*2 + dist_sensor2_rand_1*2 + dist_sensor2_rand_2*2)

DOA_ref_pts_sensor1 = Define_Transfer_func_DOA (n_points, DOA_sensor1_rand_1_fixed, DOA_sensor1_rand_2_fixed)
DOA_ref_pts_sensor2 = Define_Transfer_func_DOA (n_points, DOA_sensor2_rand_1_fixed, DOA_sensor2_rand_2_fixed)
dist_ref_pts_sensor1 = Define_Transfer_func_dist (n_points, dist_sensor1_rand_1_fixed, dist_sensor1_rand_2_fixed)
dist_ref_pts_sensor2 = Define_Transfer_func_dist (n_points, dist_sensor2_rand_1_fixed, dist_sensor2_rand_2_fixed)

data1 = data_threat(n_points)
data2 = data_threat(n_points)
DOA_gap = 60

#data generation - 1st step
print('t0')
x, y = Generate_data_quad(data1, data2)


#definition of airplane trajectory
X_ref = 0
Y_ref = 0

#creation of final outputs
finaldata = np.zeros((x.shape[0],x.shape[1] ,timesteps))
finaloutput = np.zeros((y.shape[0], timesteps))
finaldata[:,:,0] = x
finaloutput[:,0] = y

#calculation for each step of the aircraft trajectory
for i in range(1, timesteps) : 
    data_threat_actualization(data1, X_ref, Y_ref)
    data_threat_actualization(data2, X_ref, Y_ref)
    print('t'+str(i))
    x, y = Generate_data_quad(data1, data2)
    finaldata[:,:,i] = x
    finaloutput[:,i] = y


#non ambiguous data suppression
index = np.where(abs(finaldata[:,1,:])>max_DOA_gap)
index2 = np.unique(index[0])
finaldata = np.delete(finaldata,index2,axis=0)
finaloutput = np.delete(finaloutput, index2, axis=0)


#shuffling the final outputs
new_index = np.arange(len(finaloutput))
rand.shuffle(new_index)
data_shuffled_x = np.zeros((len(finaloutput), 4,timesteps))
data_shuffled_y = np.zeros((len(finaloutput), 1,timesteps))
print("-starting shuffling")
for i in range(0, len(finaloutput)):
    data_shuffled_x[i, :] = finaldata[new_index[i], :,:]
    data_shuffled_y[i] = finaloutput[new_index[i],:]
finaldata = data_shuffled_x
finaloutput = data_shuffled_y
print("shuffling over")

finaldata = finaldata[:,:,1:]
finaloutput = finaloutput[:,:,1:]


#==============================================================================
# Ecriture des données
#==============================================================================
for i in range(0,timesteps-1):
    print('-writing data '+'train_data_timeseries_number'+str(i)+'.csv')
    train_data = np.zeros((len(finaldata), len(finaldata[0,:,0])+1))
    train_data[:, :len(finaldata[0,:,0])]=finaldata[: ,:,i]
    train_data[:, len(finaldata[0,:,0]):]=finaloutput[:,:,i]
    pd.DataFrame(train_data).to_csv('train_data_timeseries'+str(i)+'.csv',index=False,header=False)
    print('-data written : '+'train_data_timeseries'+str(i)+'.csv')


print('-DONE.')