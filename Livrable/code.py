import csv
import numpy as np
from collections import deque
import copy
import random
import time
import math
import os
import pandas as pd
import matplotlib.pyplot as plt

# ================== CONFIG / SWITCH ==================
print("=== Sélection de l'instance ===")
print("1 - 6X6.csv")
print("2 - 11X11.csv")
print("3 - 51X51.csv")
print("4 - 101X101.csv")
print("5 - 201X201.csv")
print("6 - 501X501.csv")
print("7 - 1001X1001.csv")
print("8 - 1501X1501.csv")
print("9 - 2001X2001.csv")
choix = input("Choisissez la matrice à utiliser (1-9) : ")

if choix == "1":
    csv_path = "instance/6X6.csv"
    nameFile = "6X6.csv"
elif choix == "2":
    csv_path = "instance/11X11.csv"
    nameFile = "11X11.csv"
elif choix == "3":
    csv_path = "instance/51X51.csv"
    nameFile = "51X51.csv"
elif choix == "4":
    csv_path = "instance/101X101.csv"
    nameFile = "101X101.csv"
elif choix == "5":
    csv_path = "instance/201X201.csv"
    nameFile = "201X201.csv"
elif choix == "6":
    csv_path = "instance/501X501.csv"
    nameFile = "501X501.csv"
elif choix == "7":
    csv_path = "instance/1001X1001.csv"
    nameFile = "1001X1001.csv"
elif choix == "8":
    csv_path = "instance/1501X1501.csv"
    nameFile = "1501X1501.csv"
elif choix == "9":
    csv_path = "instance/2001X2001.csv"
    nameFile = "2001X2001.csv"
else:
    csv_path = "instance/6X6.csv"
    nameFile = "6X6.csv"

try:
    nbTrucks = int(input("Nombre de camions à utiliser : "))
except:
    nbTrucks = 10


depot = 0
MAX_CYCLE_TIME = 720  
SEUIL1 = 240
SEUIL2 = 480

# === Utilitaires de lecture (définit avant usage) ===
def lire_matrice_csv(filename):
    matrice = []
    with open(filename, newline='') as f:
        lecteur = csv.reader(f)
        for ligne in lecteur:
            # ignorer champs vides
            valeurs = [int(float(x)) for x in ligne if x.strip() != ""]
            if valeurs:
                matrice.append(valeurs)
    return matrice


if not os.path.exists("matrice"):
    os.makedirs("matrice")

# === Chargement de la matrice de base (celle choisie) ===
try:

    try:
        base_matrix = np.loadtxt(csv_path, delimiter=",", dtype=int).tolist()
    except Exception:
        base_matrix = lire_matrice_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

matrix = base_matrix

# ================== Fonctions algorithme (inchangées en interface) ==================
def voisinMinPoid(matrix_local, listeClient, cur):
    poidMinTrajet = 0
    nextVoisin = -1
    for i in listeClient:
        if matrix_local[cur][i] > 0 and (poidMinTrajet == 0 or matrix_local[cur][i] < poidMinTrajet):
            nextVoisin = i
            poidMinTrajet = matrix_local[cur][i]
    return nextVoisin, poidMinTrajet

def voisinsClientGraphematrix(matrix_local, sommet):
    return [i for i in range(len(matrix_local)) if matrix_local[sommet][i] > 0]

def poidCycle():
    return sum(truckCycles[0])

# ================== Fonction demandée : recherche_tabou_cycle (avec matrices horaires) ==================
# On copie la matrix pour ne pas modifier l’originale
def recherche_tabou_cycle(matrix, start):
    matrix_copy = copy.deepcopy(matrix)
    tabou = deque(maxlen=len(matrix))
    tabou.append(start)

    matrix8h_copy = copy.deepcopy(matrix8h)
    matrix12h_copy = copy.deepcopy(matrix12h)
    matrix16h_copy = copy.deepcopy(matrix16h)
    
    for i in range(nbTrucks):
        tabou.append(truckCycles[1][i][-1])

    while len(tabou) < len(matrix):

        truckAtMove = truckCycles[0].index(min(truckCycles[0]))
        cur = truckCycles[1][truckAtMove][-1]

        if  truckCycles[0][truckAtMove] < SEUIL1:
            matrixAtUse = matrix8h_copy
        elif truckCycles[0][truckAtMove] < SEUIL2:
            matrixAtUse = matrix12h_copy
        elif truckCycles[0][truckAtMove] >= MAX_CYCLE_TIME:
            print(f"Camion {truckAtMove} n'a plus de temps pour terminer sa tourné. Essayez avec plus de camions.")
            break
        else:
            matrixAtUse = matrix16h_copy
        

        voisins = voisinsClientGraphematrix(matrixAtUse, cur)
        candidats = [v for v in voisins if v not in tabou]

        if not candidats:
            break

        voisin, temps = voisinMinPoid(matrix_copy, candidats, cur)

        matrix8h_copy[cur][voisin] = 0
        matrix8h_copy[voisin][cur] = 0
        matrix12h_copy[cur][voisin] = 0
        matrix12h_copy[voisin][cur] = 0
        matrix16h_copy[cur][voisin] = 0
        matrix16h_copy[voisin][cur] = 0

        truckCycles[1][truckAtMove].append(voisin)
        truckCycles[0][truckAtMove] += temps
        tabou.append(voisin)

    # Retour au dépôt
    for i in range(nbTrucks):
        last_visited = truckCycles[1][i][-1]
        truckCycles[1][i].append(depot)
        truckCycles[0][i] += matrix[last_visited][depot]


# ================== Recherche tabou multi-start (interface conservée) ==================
def tabou_multi_start(matrix_local, nb_lancements=20):
    tempsMeilleurCycle = float('inf')
    goodI = -1
    bestTime = None

    try:
        creer_fichiers_avec_bouchons()
    except Exception as e:
        print(f"Création fichiers bouchons échouée ou déjà faite : {e}")

    for i in range(nb_lancements):
        global truckCycles
        truckCycles = [
            [0] * nbTrucks,
            [[] for _ in range(nbTrucks)]
        ]

        for j in range(nbTrucks):
            truckCycles[1][j] = [depot]
            attempts = 0
            while True:
                attempts += 1
                if attempts > 1000:
                    raise RuntimeError("Impossible d'initialiser firstNeighbor (trop d'essais).")
                firstNeighbor = random.randint(1, len(matrix_local)-1)
                if not any(firstNeighbor in cycle for cycle in truckCycles[1]):
                    truckCycles[1][j].append(firstNeighbor)
                    truckCycles[0][j] = matrix_local[depot][firstNeighbor]
                    break

        recherche_tabou_cycle(matrix_local, depot)

        total = poidCycle()
        print(f"Lancement {i+1} terminé : Temps du cycle = {total}")

        for k in range(nbTrucks):
            print(f"Premier client du camion {k+1} : {truckCycles[1][k][0]+1}")
            print(f"Cycle du camion {k+1} : ", " -> ".join(str(x+1) for x in truckCycles[1][k]))
            print(f"Temps total du camion {k+1} : {truckCycles[0][k]}")
            print()

        if total < tempsMeilleurCycle:
            tempsMeilleurCycle = total
            goodI = i
            bestTime = [
                truckCycles[0].copy(),
                [cycle.copy() for cycle in truckCycles[1]]
            ]
            print(f"→ Nouveau meilleur cycle sauvegardé ! Lancement {i+1}.\n")

    return tempsMeilleurCycle, goodI, bestTime

# ================== Partie bouchons (création des 3 matrices horaires) ==================
def generer_facteur_bouchon(heure):
    seed_value = hash(f"bouchon_{heure}") % (2**32)
    random.seed(seed_value)
    intensite = 0.5 + 0.5 * math.sin((heure - 8) / 24 * 2 * math.pi)
    facteur = 2.0 * intensite
    if facteur <= 0:
        facteur = 1
    return facteur

def facteurs_variation(matrice, pourcentage):
    n = len(matrice)
    toutes_les_routes = [(i, j) for i in range(n) for j in range(i + 1, n) if matrice[i][j] != 0]
    nb_a_modifier = int(len(toutes_les_routes) * pourcentage)
    if nb_a_modifier <= 0:
        return []
    routes_selectionnees = random.sample(toutes_les_routes, nb_a_modifier)
    for i, j in routes_selectionnees:
        p = random.uniform(-0.3, 0.3)
        nouvelle_valeur = matrice[i][j] * (1 + p)
        matrice[i][j] = matrice[j][i] = max(1, int(round(nouvelle_valeur)))
    return routes_selectionnees

def creer_fichiers_avec_bouchons():

    instances = [csv_path]
    heures = [8, 12, 16]
    matrixCreated = []

    for instance in instances:
        chemin_original = instance  
        try:
            matrice_base = lire_matrice_csv(chemin_original)
        except FileNotFoundError:
            print(f"Fichier source introuvable : {chemin_original}")
            continue

        n = len(matrice_base)
        base_name = os.path.basename(instance).replace('.csv','')

        for heure in heures:
            nom_sortie = f"matrice/{base_name}_{heure}h.csv"
            matrice_copie = copy.deepcopy(matrice_base)
            facteur_global = generer_facteur_bouchon(heure)

            proportion_routes_affectees = 0.3
            routes_affectees = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < proportion_routes_affectees:
                        routes_affectees.add((i, j))

            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in routes_affectees:
                        variation_locale = random.uniform(0.8, 1.4)
                        facteur_total = facteur_global * variation_locale
                        nouvelle_valeur = int(round(matrice_base[i][j] * facteur_total))
                        matrice_copie[i][j] = matrice_copie[j][i] = nouvelle_valeur
                    else:
                        matrice_copie[i][j] = matrice_copie[j][i] = int(matrice_base[i][j])

            # sauvegarde (écrase si existant)
            with open(nom_sortie, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(matrice_copie)

            print(f"✓ Fichier créé : {nom_sortie}")
            matrixCreated.append((nom_sortie))
    
    matrix8h = lire_matrice_csv(matrixCreated[0])
    matrix12h = lire_matrice_csv(matrixCreated[1])
    matrix16h = lire_matrice_csv(matrixCreated[2])

    return matrix8h, matrix12h, matrix16h
  
# ================== Fonctions de simulation / affichage (inchangées) ==================
def cout_effectif(matrice_local, i, j, heure):
    base = matrice_local[i][j]
    if base == 0:
        return 0
    facteur_bouchon = generer_facteur_bouchon(heure)
    cout = base * facteur_bouchon
    return max(1, int(round(cout, 0)))

def simulation_journee(matrice_local, nom_fichier):
    print(f"\n=== Simulation sur {nom_fichier} ===")
    heures = list(range(0, 25, 4))
    for h in heures:
        facteur = generer_facteur_bouchon(h)
        cout_05 = cout_effectif(matrice_local, 0, 5, h)
        print(f"Heure {h:2d}h | Facteur bouchon: {facteur:.2f} | Coût 0->5: {cout_05}")

# ================== Vérifications et tests ==================
def verifier_modifications():
    print("🔍 VÉRIFICATION DES MODIFICATIONS")
    print("=" * 50)
    random.seed(42)

    try:
        matrice_test = lire_matrice_csv(csv_path)
    except FileNotFoundError:
        print(f"Fichier introuvable : {csv_path}")
        return

    n = len(matrice_test)
    routes_non_nulles_original = sum(1 for i in range(n) for j in range(i+1,n) if matrice_test[i][j] != 0)
    print(f"Routes non-nulles originales: {routes_non_nulles_original}")

    for heure in [8, 12, 16]:
        print(f"\n--- Heure {heure}h ---")
        matrice_copie = copy.deepcopy(matrice_test)
        random.seed(hash(f"test_{heure}") % (2**32))
        modifications = facteurs_variation(matrice_copie, 0.3)
        routes_modifiees = sum(1 for i in range(n) for j in range(i+1,n) if matrice_copie[i][j] != matrice_test[i][j])
        print(f"Routes modifiées comptées: {routes_modifiees}")
        print(f"Modifications annoncées: {len(modifications)}")
        print(f"COHÉRENT: {routes_modifiees == len(modifications)}")

# ================== MAIN : exécution ==================
if __name__ == "__main__":
    matrix8h, matrix12h, matrix16h = creer_fichiers_avec_bouchons()

    verifier_modifications()

    start_time = time.time()
    tempsMeilleurCycle, goodI, bestTime = tabou_multi_start(matrix)
    execution_time_ms = (time.time() - start_time) * 1000

    print("\n=== Meilleur cycle trouvé ===")
    print("Lancement n°", goodI+1, "  Temps du cycle :", tempsMeilleurCycle)
    if bestTime:
        for i in range(nbTrucks):
            print(f"Cycle du camion {i+1} : ", " -> ".join(str(x+1) for x in bestTime[1][i]))
            print(f"Temps total du camion {i+1} : {bestTime[0][i]}\n")
    else:
        print("Aucun meilleur cycle sauvegardé.")

    print("Temps d'exécution :", round(execution_time_ms, 2), "ms")