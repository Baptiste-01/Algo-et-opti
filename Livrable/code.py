import numpy as np
from collections import deque
import copy
import random
import time
import pandas as pd

# Lecture de la matrice
matrix = pd.read_csv("instance/6x6.csv", header=None).to_numpy()

depot = 0
nbTrucks = 2

# truckCycles[0] = temps de chaque camion
# truckCycles[1] = cycle de chaque camion
truckCycles = [
    [0] * nbTrucks,
    [[] for _ in range(nbTrucks)]
]

def voisinMinPoid(matrix, listeClient, cur):
    poidMinTrajet = 0
    nextVoisin = -1
    for i in listeClient:
        if matrix[cur][i] > 0 and (poidMinTrajet == 0 or matrix[cur][i] < poidMinTrajet):
            nextVoisin = i
            poidMinTrajet = matrix[cur][i]
    return nextVoisin, poidMinTrajet

def voisinsClientGraphematrix(matrix, sommet):
    return [i for i in range(len(matrix)) if matrix[sommet][i] > 0]

def poidCycle():
    return sum(truckCycles[0])

def recherche_tabou_cycle(matrix, start):
    matrix_copy = copy.deepcopy(matrix+1)
    tabou = deque(maxlen=len(matrix))
    tabou.append(start)
    
    for i in range(nbTrucks):
        tabou.append(truckCycles[1][i][-1])
        

    while len(tabou) < len(matrix):
        # Choisir le camion avec le temps minimal
        truckAtMove = truckCycles[0].index(min(truckCycles[0]))
        cur = truckCycles[1][truckAtMove][-1]

        voisins = voisinsClientGraphematrix(matrix_copy, cur)
        candidats = [v for v in voisins if v not in tabou]

        if not candidats:
            break

        voisin, temps = voisinMinPoid(matrix_copy, candidats, cur)

        # Retirer l'arête
        matrix_copy[cur][voisin] = 0
        matrix_copy[voisin][cur] = 0

        # Mettre à jour le cycle et le temps
        truckCycles[1][truckAtMove].append(voisin)
        truckCycles[0][truckAtMove] += temps
        tabou.append(voisin)

def tabou_multi_start(matrix, nb_lancements=20):
    tempsMeilleurCycle = float('inf')
    goodI = -1
    bestTime = None  # ici on stockera seulement le meilleur

    for i in range(nb_lancements):
        global truckCycles
        truckCycles = [
            [0] * nbTrucks,
            [[] for _ in range(nbTrucks)]
        ]

        # Choix aléatoire du premier client pour chaque camion
        for j in range(nbTrucks):
            truckCycles[1][j] = [depot]
            while True:
                firstNeighbor = random.randint(1, len(matrix)-1)
                if not any(firstNeighbor in cycle for cycle in truckCycles[1]):
                    truckCycles[1][j].append(firstNeighbor)
                    truckCycles[0][j] = matrix[depot][firstNeighbor]
                    break

        recherche_tabou_cycle(matrix, depot)

        total = poidCycle()
        print(f"Lancement {i+1} terminé : Temps du cycle = {total}")
        for k in range(nbTrucks):
            print(f"Premier client du camion {k+1} : {truckCycles[1][k][0]+1}")
            print(f"Cycle du camion {k+1} : ", " -> ".join(str(x+1) for x in truckCycles[1][k]))
            print(f"Temps total du camion {k+1} : {truckCycles[0][k]}")
            print()

        # Si c'est le meilleur, on sauvegarde les cycles
        if total < tempsMeilleurCycle:
            tempsMeilleurCycle = total
            goodI = i

            # On stocke le meilleur cycle
            bestTime = [
                truckCycles[0].copy(),                    # copie des temps des camions
                [cycle.copy() for cycle in truckCycles[1]]  # copie profonde des cycles
            ]

            print(f"→ Nouveau meilleur cycle sauvegardé ! Lancement {i+1}.\n")

    return tempsMeilleurCycle, goodI, bestTime


start_time = time.time()
tempsMeilleurCycle, goodI, bestTime = tabou_multi_start(matrix)
execution_time_ms = (time.time() - start_time) * 1000

print("\n=== Meilleur cycle trouvé ===")
print("Lancement n°", goodI+1, "  Temps du cycle :", tempsMeilleurCycle)

for i in range(nbTrucks):
    print(f"Cycle du camion {i+1} : ", " -> ".join(str(x+1) for x in bestTime[1][i]))
    print(f"Temps total du camion {i+1} : {bestTime[0][i]}\n")

print("Temps d'exécution :", round(execution_time_ms, 2), "ms")


print("\nTemps d'exécution :", round(execution_time_ms, 2), "ms")
