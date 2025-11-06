import numpy as np
from collections import deque
import copy
import math
import random
import time


NbClients = 5
clientLocs = []

while len(clientLocs) < NbClients:
    x = random.randint(0, NbClients*10)
    y = random.randint(0, NbClients*10)
    
    # On vérifie si le point (x, y) existe déjà
    if [x, y] not in clientLocs:
        clientLocs.append([x, y])

clientLocs = np.array(clientLocs)

print("Client Locations:\n", clientLocs)

# calcule la matrix des distances entre les clients
def distance_matrix(locs):
    n = locs.shape[0] # Calcul du nombre de clients
    dist_matrix = np.zeros((n, n)) # Création d'une matrix n x n remplie de zéros
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(locs[i] - locs[j]) # Calcul de la distance euclidienne entre les clients i et j
    dist_matrix = np.round(dist_matrix).astype(int) # Arrondi des distances et conversion en entiers

    return dist_matrix

matrix = distance_matrix(clientLocs)
print("Distance Matrix:\n", distance_matrix(clientLocs))

filename = f"instance/matrix_distances_{NbClients+1}x{NbClients+1}.csv"  
np.savetxt(filename, matrix, delimiter=",", fmt='%d')
