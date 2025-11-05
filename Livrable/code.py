from collections import deque
import copy
import random
import time


listeClient = [5, 8, 3, 6, 12]
depot = 0

matrice = [
 [0, 12, 5, 19, 1, 8, 16, 11, 6, 7, 13, 17, 10],
 [12, 0, 4, 3, 15, 18, 20, 8, 19, 11, 14, 5, 9],
 [5, 4, 0, 14, 17, 10, 2, 13, 7, 18, 8, 9, 16],
 [19, 3, 14, 0, 12, 6, 9, 20, 15, 2, 10, 11, 7],
 [1, 15, 17, 12, 0, 13, 11, 3, 8, 4, 16, 14, 5],
 [8, 18, 10, 6, 13, 0, 5, 9, 20, 16, 3, 15, 19],
 [16, 20, 2, 9, 11, 5, 0, 7, 14, 19, 18, 12, 4],
 [11, 8, 13, 20, 3, 9, 7, 0, 10, 17, 2, 5, 15],
 [6, 19, 7, 15, 8, 20, 14, 10, 0, 12, 9, 16, 18],
 [7, 11, 18, 2, 4, 16, 19, 17, 12, 0, 15, 8, 6],
 [13, 14, 8, 10, 16, 3, 18, 2, 9, 15, 0, 19, 20],
 [17, 5, 9, 11, 14, 15, 12, 5, 16, 8, 19, 0, 13],
 [10, 9, 16, 7, 5, 19, 4, 15, 18, 6, 20, 13, 0]
]

matrice_copy = copy.deepcopy(matrice)


def voisinMinPoid(matrice, listeClient, cur):
    """
    Retourne le voisin de poids minimum parmi les voisins du sommet 'cur'
    """
    poidTrajet = []
    nextVoisin = -1
    
    for i in listeClient:
        if matrice[cur][i] > 0 and min(matrice[cur][i]) :
            nextVoisin = i
            poidTrajet.append(matrice[cur][i])
        elif matrice[cur][i] > 0:
            poidTrajet.append(matrice[cur][i]) 

    return nextVoisin

def voisinsClientGrapheMatrice(matrice, sommet):
    """
    Retourne la liste des voisins d’un sommet dans le graphe
    """
    voisins = [i for i in enumerate(listeClient) if matrice[sommet][i] > 0]
    return voisins 

def poidCycle(matrice, cycle):
    """
    Calcule le poids total d’un cycle donné
    """
    poids_total = 0
    for i in range(len(cycle) - 1):
        poids_total += matrice[cycle[i]][cycle[i + 1]]
    poids_total += matrice[cycle[-1]][cycle[0]] 
    return poids_total



def recherche_tabou_cycle(matrice, start, iter_max=100):
    """
    Cherche un long chemin eulérien à partir d’un sommet initial 'start'
    en utilisant une liste tabou pour éviter de revenir sur des sommets récents.

    - matrice : matrice d’adjacence du graphe
    - start : sommet de départ
    - taille_tabou : nombre maximal de sommets récents à "interdire"
    - iter_max : nombre maximal d’itérations (limite d’exploration)
    """

    # On copie la matrice pour ne pas modifier l’originale
    matrice_copy = copy.deepcopy(matrice)

    # Le cycle que nous construisons (liste d’indices de sommets)
    cycle = start

    # Liste tabou : elle garde les derniers sommets visités pour éviter les retours
    tailleTabou = len(listeClient) + 5
    tabou = deque(maxlen= tailleTabou)
    tabou.append(start)

    # Le sommet courant (celui où on se trouve actuellement)
    cur = start

    # Boucle principale de la recherche tabou
    for _ in range(iter_max):

        # On récupère la liste des voisins encore connectés du sommet courant
        voisins = voisinsClientGrapheMatrice(matrice_copy, cur)

        # On enlève les voisins qui sont "tabou" 
        candidats = [i for i in voisins if i not in tabou]

        # S’il n’y a aucun voisin disponible, on ne peut plus avancer
        if not candidats:
            break

        voisin = voisinMinPoid(matrice_copy, listeClient, cur)

        # On retire l’arête entre le sommet courant et le voisin choisi
        matrice_copy[cur][voisin] = 0
        matrice_copy[voisin][cur] = 0
        
        cycle.append(voisin) # On ajoute ce voisin au cycle    
        tabou.append(cur) # On ajoute le sommet courant dans la liste tabou

        cur = voisin

    # On retourne le chemin (cycle) trouvé
    return cycle



def tabou_multi_start(matrice, nb_lancements=10, iter_max=100):
    """
    Lance plusieurs recherches tabou depuis des sommets de départ aléatoires,
    puis retourne le meilleur cycle (le plus long) trouvé.

    - nb_lancements : nombre d’essais (points de départ différents)
    - iter_max : nombre d’itérations par recherche
    """

    meilleur_cycle = []  # Le meilleur cycle global (le plus court)
    tempsMeilleurCycle = 0

    # On répète l’expérience plusieurs fois (multi-start)
    for i in range(nb_lancements):

        start = depot

        # On effectue une recherche tabou locale à partir de ce sommet
        cycle = recherche_tabou_cycle(matrice, start, iter_max)

        # On affiche le résultat intermédiaire
        print(f"Lancement {i+1}: départ={start+1}, longueur du cycle={len(cycle)}")

        
        if tempsMeilleurCycle == 0:
            tempsMeilleurCycle = poidCycle(matrice, cycle)
        elif poidCycle(matrice, cycle) > tempsMeilleurCycle:
            meilleur_cycle = cycle

    # Après tous les lancements, on renvoie le meilleur
    return meilleur_cycle



# Mesure du temps d’exécution
start_time = time.time()

print("### Recherche tabou multi-start sur la Zone A ###\n")

# Lancement du multi-start (10 essais, taille tabou = 5, 100 itérations max)
meilleur_cycle = tabou_multi_start(matrice, nb_lancements=10, iter_max=100)

# Fin du chrono
end_time = time.time()
execution_time_ms = (end_time - start_time) * 1000

# Affichage du meilleur résultat trouvé
print("\n=== Meilleur cycle trouvé ===")
print("Longueur du cycle :", len(meilleur_cycle))
for s in meilleur_cycle:
    print(s + 1, "-> ", end='')
print(meilleur_cycle[0] + 1)  # on revient au départ pour fermer le cycle

print("\nTemps d'exécution :", round(execution_time_ms, 2), "ms")


prob_bouchon = 0.3
facteur = 3
def traffic_jam(matrice, prob_bouchon, facteur):
    matrice_copy = copy.deepcopy(matrice)

    for x in range(0, len(matrice)):
        for y in range(0,len(matrice)):
            if x != y:
                if random.random() < prob_bouchon:
                    matrice_copy[i][j] *= facteur
    return matrice_copy

matrice_avec_bouchons = traffic_jam(matrice, prob_bouchon, facteur)