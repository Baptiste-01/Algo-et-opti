import csv
import numpy as np
from collections import deque
import copy
import random
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
# Lecture de la matrice
matrix = np.loadtxt("instance/101x101.csv", delimiter=",", dtype=int)

depot = 0
nbTrucks = 10

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
    matrix_copy = copy.deepcopy(matrix)
    tabou = deque(maxlen=len(matrix))
    tabou.append(start)
    
    for i in range(nbTrucks):
        tabou.append(truckCycles[1][i][-1])

    while len(tabou) < len(matrix):
        truckAtMove = truckCycles[0].index(min(truckCycles[0]))
        cur = truckCycles[1][truckAtMove][-1]

        voisins = voisinsClientGraphematrix(matrix_copy, cur)
        candidats = [v for v in voisins if v not in tabou]

        if not candidats:
            break

        voisin, temps = voisinMinPoid(matrix_copy, candidats, cur)

        matrix_copy[cur][voisin] = 0
        matrix_copy[voisin][cur] = 0

        truckCycles[1][truckAtMove].append(voisin)
        truckCycles[0][truckAtMove] += temps
        tabou.append(voisin)

    # Retour au dépôt
    for i in range(nbTrucks):
        last_visited = truckCycles[1][i][-1]
        truckCycles[1][i].append(depot)
        truckCycles[0][i] += matrix[last_visited][depot]

def tabou_multi_start(matrix, nb_lancements=20):
    tempsMeilleurCycle = float('inf')
    goodI = -1
    bestTime = None

    for i in range(nb_lancements):
        global truckCycles
        truckCycles = [
            [0] * nbTrucks,
            [[] for _ in range(nbTrucks)]
        ]

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

        if total < tempsMeilleurCycle:
            tempsMeilleurCycle = total
            goodI = i
            bestTime = [
                truckCycles[0].copy(),
                [cycle.copy() for cycle in truckCycles[1]]
            ]
            print(f"→ Nouveau meilleur cycle sauvegardé ! Lancement {i+1}.\n")

    return tempsMeilleurCycle, goodI, bestTime


# === Lancement principal ===
start_time = time.time()
tempsMeilleurCycle, goodI, bestTime = tabou_multi_start(matrix)
execution_time_ms = (time.time() - start_time) * 1000

print("\n=== Meilleur cycle trouvé ===")
print("Lancement n°", goodI+1, "  Temps du cycle :", tempsMeilleurCycle)
for i in range(nbTrucks):
    print(f"Cycle du camion {i+1} : ", " -> ".join(str(x+1) for x in bestTime[1][i]))
    print(f"Temps total du camion {i+1} : {bestTime[0][i]}\n")

print("Temps d'exécution :", round(execution_time_ms, 2), "ms")


# === Fonctions de lecture et simulation (inchangées) ===

def lire_matrice_csv(filename):
    matrice = []
    with open(filename, newline='') as f:
        lecteur = csv.reader(f)
        for ligne in lecteur:
            valeurs = [int(float(x)) for x in ligne if x.strip() != ""]
            if valeurs:
                matrice.append(valeurs)
    return matrice

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
    routes_selectionnees = random.sample(toutes_les_routes, nb_a_modifier)
    
    for i, j in routes_selectionnees:
        p = random.uniform(-0.3, 0.3)
        nouvelle_valeur = matrice[i][j] * (1 + p)
        matrice[i][j] = matrice[j][i] = max(1, int(round(nouvelle_valeur)))
    
    return routes_selectionnees

def cout_effectif(matrice, i, j, heure):
    base = matrice[i][j]
    if base == 0:
        return 0
    facteur_bouchon = generer_facteur_bouchon(heure)
    cout = base * facteur_bouchon 
    return max(1, int(round(cout, 0)))

def simulation_journee(matrice, nom_fichier):
    print(f"\n=== Simulation sur {nom_fichier} ===")
    heures = list(range(0, 25, 4))
    for h in heures:
        facteur = generer_facteur_bouchon(h)
        cout_05 = cout_effectif(matrice, 0, 5, h)
        print(f"Heure {h:2d}h | Facteur bouchon: {facteur:.2f} | Coût 0->5: {cout_05}")

# === Partie bouchons ===

def creer_fichiers_avec_bouchons():
    matrix_instances = ['101x101.csv']
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

            with open(nom_sortie, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(matrice_copie)

            print(f"✓ Fichier créé : {nom_sortie}")

# c pour tester que tout fonctionne correctement avec la matrice 101x101
def test_bouchons():
    """
    Test uniquement le système de bouchons avec la matrice 101x101
    """

    # 1. Lire la matrice originale
    #print("1. Lecture de la matrice 101x101...")
    #print(lire_matrice_csv("instance/101x101.csv"))
    matrice_originale = lire_matrice_csv("instance/101x101.csv")
    # print(f"   ✅ Matrice originale : {len(matrice_originale)}x{len(matrice_originale)}")
    
    # 2. Tester la simulation sur 24h
    print("\n2. Simulation sur 24h...")
    simulation_journee(matrice_originale, "101x101.csv")
    
    # 3. Créer les 3 fichiers avec bouchons (UNIQUEMENT CET APPEL)
    print("\n3. Création des fichiers avec bouchons...")
    creer_fichiers_avec_bouchons()
    print("\n" + "=" * 50)
    print("🎉 TEST BOUCHONS TERMINÉ !")
    print("3 fichiers créés dans le dossier 'matrice/'")
    print("=" * 50)


def verifier_modifications():
    print("🔍 VÉRIFICATION DES MODIFICATIONS")
    print("=" * 50)
    random.seed(42)
    matrice_test = lire_matrice_csv("instance/101x101.csv")
    n = len(matrice_test)
    
    routes_non_nulles_original = 0
    for i in range(n):
        for j in range(i + 1, n):
            if matrice_test[i][j] != 0:
                routes_non_nulles_original += 1
    print(f"Routes non-nulles originales: {routes_non_nulles_original}")
    
    for heure in [8, 12, 20]:
        print(f"\n--- Heure {heure}h ---")
        matrice_copie = copy.deepcopy(matrice_test)
        random.seed(hash(f"test_{heure}") % (2**32))
        modifications = facteurs_variation(matrice_copie, 0.3)
        routes_modifiees = 0
        for i in range(n):
            for j in range(i + 1, n):
                if matrice_copie[i][j] != matrice_test[i][j]:
                    routes_modifiees += 1
        print(f"Routes modifiées comptées: {routes_modifiees}")
        print(f"Modifications annoncées: {len(modifications)}")
        print(f"COHÉRENT: {routes_modifiees == len(modifications)}")


# === Exécution finale ===
verifier_modifications()
test_bouchons()

#test analyse statistiques
def run_multiple_experiments(matrix, n_runs=20):
    results = []
    for i in range(n_runs):
        start = time.time()
        best_cost, best_iter, best_solution = tabou_multi_start(matrix, nb_lancements=1)
        duration = time.time() - start
        results.append({
            "run": i + 1,
            "cost": best_cost,
            "iteration": best_iter,
            "time_s": duration
        })
        print(f" Run {i+1}/{n_runs} terminé — Coût = {best_cost} | Temps = {duration:.2f}s")
    return pd.DataFrame(results)


# === test ===
df_results = run_multiple_experiments(matrix, n_runs=20)
df_results.to_csv("resultats_tabou.csv", index=False)
df_results.plot(y='cost', kind='bar', title='Coût des cycles sur 20 runs')
plt.xlabel('Run')

plt.ylabel('Coût du cycle')
plt.show()

def analyse_resultats(df):
    print("\n === Statistiques globales ===")
    print(df.describe()[["cost", "time_s"]])

    cout_moyen = df["cost"].mean()
    ecart_type = df["cost"].std()
    cout_min = df["cost"].min()
    cout_max = df["cost"].max()

    print(f"\nCoût moyen : {cout_moyen:.2f}")
    print(f"Écart-type : {ecart_type:.2f}")
    print(f"Meilleur coût : {cout_min:.2f}")
    print(f"Pire coût : {cout_max:.2f}")

analyse_resultats(df_results)

plt.figure(figsize=(6, 5))
plt.boxplot(df_results["cost"], showmeans=True)
plt.title("Distribution des coûts (Tabu Search)")
plt.ylabel("Coût total")
plt.grid(alpha=0.3)
plt.show()


plt.figure(figsize=(6, 5))
plt.hist(df_results["time_s"], bins=8, color="skyblue", edgecolor="black")
plt.title("Distribution des temps d'exécution")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence")
plt.grid(alpha=0.3)
plt.show()


iterations = np.arange(0, 50)
costs_example = [5000 / (1 + 0.1*i) + np.random.normal(0, 50) for i in iterations]

plt.figure(figsize=(7, 4))
plt.plot(iterations, costs_example, marker='o')
plt.title("Exemple de convergence du coût au fil des itérations")
plt.xlabel("Itérations")
plt.ylabel("Coût total")
plt.grid(alpha=0.3)
plt.show()

opt_cost = 27591  # exemple : solution optimale X-n101-k25
df_results["gap_%"] = 100 * (df_results["cost"] - opt_cost) / opt_cost

print("\n=== GAP par rapport à la solution optimale ===")
print(df_results[["run", "cost", "gap_%"]])

print(f"\n Moyenne du gap : {df_results['gap_%'].mean():.2f}%")

plt.figure(figsize=(6,5))
plt.boxplot(df_results["gap_%"], showmeans=True)
plt.title("Distribution du GAP (%) par rapport à l’optimum")
plt.ylabel("GAP (%)")
plt.grid(alpha=0.3)
plt.show()