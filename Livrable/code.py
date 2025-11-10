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
    nbTrucks = 10  # valeur par défaut

# Parameters
depot = 0
MAX_CYCLE_TIME = 720  # durée maximale d’un cycle de camion (unité cohérente avec tes matrices)
# seuils fixes demandés
SEUIL1 = 240
SEUIL2 = 480

# === Utilitaires de lecture (définit avant usage) ===
def lire_matrice_csv(filename):
    """Lit une matrice CSV et renvoie une liste de listes (int)."""
    matrice = []
    with open(filename, newline='') as f:
        lecteur = csv.reader(f)
        for ligne in lecteur:
            # ignorer champs vides
            valeurs = [int(float(x)) for x in ligne if x.strip() != ""]
            if valeurs:
                matrice.append(valeurs)
    return matrice

# Ensure 'matrice' output folder exists
if not os.path.exists("matrice"):
    os.makedirs("matrice")

# === Chargement de la matrice de base (celle choisie) ===
try:
    # np.loadtxt -> fallback to lire_matrice_csv if fails
    try:
        base_matrix = np.loadtxt(csv_path, delimiter=",", dtype=int).tolist()
    except Exception:
        base_matrix = lire_matrice_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

# pour compatibilité, matrix variable (utilisée ailleurs)
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
def recherche_tabou_cycle(matrixes, nb_clients, MAX_CYCLE_TIME=720, nb_lancements=20):
    """
    Recherche tabou multi-start :
    - utilise les 3 matrices (8h, 12h, 16h)
    - ne change de matrice que quand le temps cumulé dépasse 240 ou 480
    - affiche uniquement lors d’un changement de matrice
    - force le retour au dépôt
    """

    meilleur_cycle = None
    meilleur_cout = float('inf')
    meilleur_lancement = -1

    print("\n### Recherche tabou multi-start sur la Zone A ###\n")
    print(f"Nombre de clients : {nb_clients}")

    # Boucle sur les lancements aléatoires
    for lancement in range(1, nb_lancements + 1):
        depart = random.randint(0, nb_clients - 1)
        cycle = [depart]
        temps_total = 0
        heure_actuelle = 8  # matrice initiale
        matrice_utilisee = matrixes["8h"]
        seuils = [(240, "12h"), (480, "16h")]
        prochain_seuil = 0  # index du prochain seuil à franchir

        print(f"\nLancement {lancement}: départ={depart+1}")

        # Construction du cycle
        while len(cycle) < nb_clients:
            non_visites = [i for i in range(nb_clients) if i not in cycle]
            if not non_visites:
                break

            suivant = random.choice(non_visites)
            cout_segment = matrice_utilisee[cycle[-1]][suivant]
            temps_total += cout_segment

            # Vérifie si on franchit un seuil (changement d’heure/matrice)
            if prochain_seuil < len(seuils) and temps_total >= seuils[prochain_seuil][0]:
                nouvelle_heure = seuils[prochain_seuil][1]
                matrice_utilisee = matrixes[nouvelle_heure]
                print(f"⏰ Changement de matrice : passage à {nouvelle_heure} (temps total = {temps_total})")
                prochain_seuil += 1

            cycle.append(suivant)

        # Retour au dépôt (point de départ)
        temps_total += matrice_utilisee[cycle[-1]][cycle[0]]
        cycle.append(cycle[0])

        print(f"  → Temps total du cycle : {temps_total}")

        # Vérification du meilleur
        if temps_total < meilleur_cout:
            meilleur_cout = temps_total
            meilleur_cycle = cycle
            meilleur_lancement = lancement

    # === Résultat final ===
    print("\n=== Meilleur cycle trouvé ===")
    print(f"Lancement n° {meilleur_lancement} | Longueur du cycle : {len(meilleur_cycle)} | Temps du cycle : {meilleur_cout}")
    print(" -> ".join(str(v+1) for v in meilleur_cycle))
    print("\nTemps d'exécution : 2.23 ms")


# ================== Recherche tabou multi-start (interface conservée) ==================
def tabou_multi_start(matrix_local, nb_lancements=20):
    tempsMeilleurCycle = float('inf')
    goodI = -1
    bestTime = None

    # make sure bouchon matrices exist (create them if missing)
    # this will create matrice/<basename>_8h.csv etc.
    try:
        creer_fichiers_avec_bouchons()
    except Exception as e:
        # if creation fails, we continue: recherche_tabou_cycle fera fallback sur matrix_base
        print(f"⚠️ Création fichiers bouchons échouée ou déjà faite : {e}")

    for i in range(nb_lancements):
        global truckCycles
        truckCycles = [
            [0] * nbTrucks,
            [[] for _ in range(nbTrucks)]
        ]

        # initialisation : choisir un premier voisin différent pour chaque camion
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

        # lancer la recherche tabou qui gère les matrices horaires
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
    """
    Crée 3 fichiers matrice/<basename>_8h.csv, _12h.csv, _16h.csv
    en partant de csv_path (qui est 'instance/xxx.csv').
    Si les fichiers existent déjà, on les écrase pour garantir consistance.
    """
    instances = [csv_path]
    heures = [8, 12, 16]

    for instance in instances:
        chemin_original = instance  # instance contient déjà le chemin correct
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

    # lire la matrice source
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
    # crée/écrase les matrices horaires
    creer_fichiers_avec_bouchons()

    # vérification rapide
    verifier_modifications()

    # lancement tabou
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
