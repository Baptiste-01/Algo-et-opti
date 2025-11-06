import numpy as np
from collections import deque
import copy
import math
import random
import time


depot = 0

def voisinMinPoid(matrix, listeClient, cur):
    poidMinTrajet = 0
    nextVoisin = -1
    
    for i in listeClient:
        if matrix[cur][i] > 0 and poidMinTrajet == 0:
            nextVoisin = i
            poidMinTrajet = matrix[cur][i]
        elif matrix[cur][i] > 0 and matrix[cur][i] < poidMinTrajet:
            poidMinTrajet = matrix[cur][i]
            nextVoisin = i

    return nextVoisin

def voisinsClientGraphematrix(matrix, sommet):
    voisins = [i for i in range(len(matrix)) if matrix[sommet][i] > 0]
    return voisins 

def poidCycle(matrix, cycle):
    poids_total = 0
    for i in range(len(cycle) - 1):
        poids_total += matrix[cycle[i]][cycle[i + 1]]
    poids_total += matrix[cycle[-1]][cycle[0]] 
    return poids_total



def recherche_tabou_cycle(matrix, start, firstNeighbor, iter_max=100):

    # On copie la matrix pour ne pas modifier l’originale
    matrix_copy = copy.deepcopy(matrix)

    # Le cycle que nous construisons (liste d’indices de sommets)
    cycle = [start]

    # Liste tabou : elle garde les derniers sommets visités pour éviter les retours
    tailleTabou = len(matrix) + 5
    tabou = deque(maxlen= tailleTabou)
    tabou.append(start)

    # Le sommet courant (celui où on se trouve actuellement)
    cur = start

    # Boucle principale de la recherche tabou
    for _ in range(iter_max):

        if len(cycle) == 1:
            voisin = firstNeighbor
        else:
            voisins = voisinsClientGraphematrix(matrix_copy, cur) # On récupère la liste des voisins encore connectés du sommet courant
           
            candidats = [i for i in voisins if i not in tabou] # On enlève les voisins qui sont "tabou" 

            # S’il n’y a aucun voisin disponible, on ne peut plus avancer
            if not candidats:
                break

            voisin = voisinMinPoid(matrix_copy, candidats, cur)

        # On retire l’arête entre le sommet courant et le voisin choisi
        matrix_copy[cur][voisin] = 0
        matrix_copy[voisin][cur] = 0
        
        cycle.append(voisin) # On ajoute ce voisin au cycle    
        tabou.append(voisin) # On ajoute le sommet courant dans la liste tabou

        cur = voisin

    # On retourne le chemin (cycle) trouvé
    return cycle



def tabou_multi_start(matrix, nb_lancements=10, iter_max=100):
    """
    Lance plusieurs recherches tabou depuis des sommets de départ aléatoires,
    puis retourne le meilleur cycle (le plus long) trouvé.

    - nb_lancements : nombre d’essais (points de départ différents)
    - iter_max : nombre d’itérations par recherche
    """

    meilleur_cycle = []  # Le meilleur cycle global (le plus court)
    tempsMeilleurCycle = 0
    goodI = 0

    # On répète l’expérience plusieurs fois (multi-start)
    for i in range(nb_lancements):

        start = depot

        firstNeighbor = 0

        while matrix[start][firstNeighbor] == 0:
            firstNeighbor = random.randint(1, len(matrix)-1)

        # On effectue une recherche tabou locale à partir de ce sommet
        cycle = recherche_tabou_cycle(matrix, start, firstNeighbor, iter_max)

        # On affiche le résultat intermédiaire
        print(f"Lancement {i+1}: départ={firstNeighbor}, longueur du cycle={len(cycle)}, temps du trajet={poidCycle(matrix, cycle)}")

        
        if tempsMeilleurCycle == 0:
            tempsMeilleurCycle = poidCycle(matrix, cycle)
            meilleur_cycle = cycle
            goodI = i+1
        elif poidCycle(matrix, cycle) < tempsMeilleurCycle:
            meilleur_cycle = cycle
            tempsMeilleurCycle = poidCycle(matrix, cycle)
            goodI = i+1

    # Après tous les lancements, on renvoie le meilleur
    return meilleur_cycle, tempsMeilleurCycle, goodI


# Mesure du temps d’exécution
start_time = time.time()

print("### Recherche tabou multi-start sur la Zone A ###\n")
print("Nombre de clients :", len(matrix))

# Lancement du multi-start (10 essais, taille tabou = 5, 100 itérations max)
meilleur_cycle, tempsMeilleurCycle, goodI = tabou_multi_start(matrix, 20, 100)

# Fin du chrono
end_time = time.time()
execution_time_ms = (end_time - start_time) * 1000

# Affichage du meilleur résultat trouvé
print("\n=== Meilleur cycle trouvé ===")
print("Lancement n°", goodI, "Longueur du cycle :", len(meilleur_cycle)+ 1, "  Temps du cycle :", tempsMeilleurCycle)
for s in meilleur_cycle:
    print(s + 1, "-> ", end='')
print(meilleur_cycle[0]+1)  # on revient au départ pour fermer le cycle

print("\nTemps d'exécution :", round(execution_time_ms, 2), "ms")
