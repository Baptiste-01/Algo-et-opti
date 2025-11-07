import csv
import numpy as np
from collections import deque
import copy
import math
import random
import time

import pandas as pd


depot = 0
matrix = pd.read_csv("instance/11x11.csv", header=None).to_numpy()
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


#bon 
def lire_matrice_csv(filename):
    """
    Lit une matrice complète depuis un fichier CSV.
    Retourne une liste de listes (matrice).
    """
    matrice = []
    with open(filename, newline='') as f:
            lecteur = csv.reader(f)
            for ligne in lecteur:
                # on ignore les champs vides
                valeurs = [int(float(x)) for x in ligne if x.strip() != ""]
                if valeurs:  # si la ligne n’est pas vide
                    matrice.append(valeurs)
    return matrice


#bon 
def generer_facteur_bouchon(heure):
    """
    Génère un facteur global de bouchon selon l'heure de la journée.
    - Peu de bouchons la nuit
    - Maximal vers 8h et 17h
    """

    seed_value = hash(f"bouchon_{heure}") % (2**32)
    random.seed(seed_value)
    # Heure normalisée sur 24h → sinus pour faire un cycle
    intensite = 0.5 + 0.5 * math.sin((heure - 8) / 24 * 2 * math.pi)
    # Variation entre 1.0 et 3.0 environ
    facteur =  2.0 * intensite  
    if facteur <= 0:
        facteur = 1
    return facteur

def facteurs_variation(matrice, pourcentage):
    """
    Applique des variations aléatoires (positives ou négatives)
    sur un certain pourcentage de routes, sans doublons.
    Retourne la liste des routes modifiées.
    """
    n = len(matrice)
    toutes_les_routes = [(i, j) for i in range(n) for j in range(i + 1, n) if matrice[i][j] != 0]
    nb_a_modifier = int(len(toutes_les_routes) * pourcentage)
    routes_selectionnees = random.sample(toutes_les_routes, nb_a_modifier)
    
    for i, j in routes_selectionnees:
        p = random.uniform(-0.3, 0.3)  # ±30% de variation locale
        nouvelle_valeur = matrice[i][j] * (1 + p)
        matrice[i][j] = matrice[j][i] = max(1, int(round(nouvelle_valeur)))
    
    return routes_selectionnees


def cout_effectif(matrice, i, j, heure):
    """
    Retourne le coût dynamique entre 2 villes à une heure donnée.
    Préserve la symétrie : cout(i,j) = cout(j,i)
    """
    base = matrice[i][j]
    if base == 0:
        return 0
    
    # Facteur global du trafic (selon l'heure)
    facteur_bouchon = generer_facteur_bouchon(heure)

    
    cout = base * facteur_bouchon 
    return max(1, int(round(cout, 0)))  # ✅ Éviter les 0


def simulation_journee(matrice, nom_fichier):
    """
    Simule une journée complète de trafic sur une matrice donnée.
    """
    print(f"\n=== Simulation sur {nom_fichier} ===")
    heures = list(range(0, 25, 4))  # toutes les 4 heures
    for h in heures:
        facteur = generer_facteur_bouchon(h)
        cout_05 = cout_effectif(matrice, 0, 5, h)
        print(f"Heure {h:2d}h | Facteur bouchon: {facteur:.2f} | Coût 0->5: {cout_05}")
import random

def creer_fichiers_avec_bouchons():
    matrix_instances = ['11x11.csv']
    heures = [8, 12, 20]

    for instance in matrix_instances:
        print(f"\n{'='*50}")
        print(f"Traitement de {instance}")
        print(f"{'='*50}")

        chemin_original = f"instance/{instance}"
        try:
            matrice_base = lire_matrice_csv(chemin_original)
            n = len(matrice_base)
        except FileNotFoundError:
            print(f"Fichier {chemin_original} non trouvé")
            continue

        for heure in heures:
            nom_sortie = f"matrice/{instance.replace('.csv', '')}_{heure}h.csv"
            print(f"\nCréation de {nom_sortie}...")

            matrice_copie = copy.deepcopy(matrice_base)


            facteur_global = generer_facteur_bouchon(heure)

            proportion_routes_affectees = 0.3  # 30 % des routes changent
            routes_affectees = set()

            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < proportion_routes_affectees:
                        routes_affectees.add((i, j))

            # --- Application des changements ---
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in routes_affectees:
                        # route touchée → facteur global + variation locale
                        variation_locale = random.uniform(0.8, 1.4)
                        facteur_total = facteur_global * variation_locale
                        nouvelle_valeur = int(round(matrice_base[i][j] * facteur_total))
                    
                        matrice_copie[i][j] = matrice_copie[j][i] = nouvelle_valeur
                    else:
                        # route non touchée → inchangée
                        matrice_copie[i][j] = matrice_copie[j][i] = int(matrice_base[i][j])

            # --- Sauvegarde ---
            with open(nom_sortie, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(matrice_copie)

            print(f"✓ Fichier créé : {nom_sortie}")

# c pour tester que tout fonctionne correctement avec la matrice 11x11
def test_bouchons():
    """
    Test uniquement le système de bouchons avec la matrice 11x11
    """

    # 1. Lire la matrice originale
    print("1. Lecture de la matrice 11x11...")
    print(lire_matrice_csv("instance/11x11.csv"))
    matrice_originale = lire_matrice_csv("instance/11x11.csv")
    print(f"   ✅ Matrice originale : {len(matrice_originale)}x{len(matrice_originale)}")
    
    # 2. Tester la simulation sur 24h
    print("\n2. Simulation sur 24h...")
    simulation_journee(matrice_originale, "11x11.csv")
    
    # 3. Créer les 3 fichiers avec bouchons (UNIQUEMENT CET APPEL)
    print("\n3. Création des fichiers avec bouchons...")
    creer_fichiers_avec_bouchons()  # ✅ Juste cet appel
    
    print("\n" + "=" * 50)
    print("🎉 TEST BOUCHONS TERMINÉ !")
    print("3 fichiers créés dans le dossier 'matrice/'")
    print("=" * 50)


def verifier_modifications():
    """Vérifie que le nombre de modifications est cohérent"""
    print("🔍 VÉRIFICATION DES MODIFICATIONS")
    print("=" * 50)
    random.seed(42)
    matrice_test = lire_matrice_csv("instance/11x11.csv")
    n = len(matrice_test)
    
    # Compter les routes non-nulles originales
    routes_non_nulles_original = 0
    for i in range(n):
        for j in range(i + 1, n):
            if matrice_test[i][j] != 0:
                routes_non_nulles_original += 1
    
    print(f"Routes non-nulles originales: {routes_non_nulles_original}")
    
    for heure in [8, 12, 20]:
        print(f"\n--- Heure {heure}h ---")
        matrice_copie = copy.deepcopy(matrice_test)
        
        # Appliquer variations
        random.seed(hash(f"test_{heure}") % (2**32))
        modifications = facteurs_variation(matrice_copie, 0.3)
        
        # Compter les routes modifiées
        routes_modifiees = 0
        for i in range(n):
            for j in range(i + 1, n):
                if matrice_copie[i][j] != matrice_test[i][j]:
                    routes_modifiees += 1
        
        print(f"Routes modifiées comptées: {routes_modifiees}")
        print(f"Modifications annoncées: {len(modifications)}")
        print(f"COHÉRENT: {routes_modifiees == len(modifications)}")


# Ajoutez cet appel avant test_bouchons()
verifier_modifications()
test_bouchons()
