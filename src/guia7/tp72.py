import numpy as np
import matplotlib.pyplot as plt
import random
'''
Ant Colony System (ACS)

ALGORITMO:

    1. Inicializar tiempo t = 0 y las feromonas sigma_ij(t) = U(0, sigma_0)
    2. Ubicar N hormigas en el nodo origen
    3. Repetir hasta condición de parada:
       3.1 Para cada hormiga k = 1, 2, ..., N:
           3.1.1 Inicializar camino de la hormiga p_k(t) = vacío
           3.1.2 Repetir:
               - Seleccionar el siguiente nodo según la probabilidad p_ij^k(t)
               - Agregar un paso (i, j) al camino p_k(t)
             Hasta que se alcance el destino
           3.1.3 Calcular la longitud del camino encontrado f(p_k(t))
       3.2 Para cada conexión (i, j):
           - Reducir las feromonas por evaporación: sigma_ij(t) = (1 - rho) * sigma_ij(t)
           - Depositar feromonas proporcionalmente a la bondad de la solución:
             delta_sigma_ij^k(t) = Q / f(p_k(t)) para la actualización global
                         Q / d_ij para la actualización local
           - Actualizar feromonas: sigma_ij(t + 1) = sigma_ij(t) + Σ delta_sigma_ij^k(t)
       3.3 Incrementar tiempo t ← t + 1
       3.4 Asegurar que todas las hormigas sigan el mismo camino
    4. Devolver el mejor camino encontrado

VARIABLES:

    t : tiempo en cada instante del algoritmo
    N : número de hormigas
    sigma_ij(t) : cantidad de feromonas en la conexión entre los nodos i y j en el tiempo t
    sigma_0 : valor inicial de las feromonas
    p_k(t) : camino recorrido por la hormiga k en el tiempo t
    p_ij^k(t) : probabilidad de que la hormiga k elija el nodo j desde el nodo i en el tiempo t
    (i, j) : una conexión entre los nodos i y j
    f(p_k(t)) : longitud del camino recorrido por la hormiga k en el tiempo t
    rho : tasa de evaporación de las feromonas
    Q : parámetro que controla la cantidad de feromonas depositadas
    d_ij : distancia entre los nodos i y j
    delta_sigma_ij^k(t) : cantidad de feromonas depositadas por la hormiga k en la conexión (i, j)
    Σ delta_sigma_ij^k(t) : suma de las feromonas depositadas por todas las hormigas en la conexión (i, j)

'''

def inicializacion(n_ants, n_cities):
    # Inicializar tiempo t = 0 y las feromonas sigma_ij(t) = U(0, sigma_0)


#--- Inicialización de parámetros ---#
distance_matrix = np.genfromtxt('./data/gr17.csv', delimiter=',')   # Matriz de distancias
pheromone_matrix = np.ones_like(distance_matrix)                    # Matriz de feromonas (inicializada llena de 1s) 
n_ants = 30                            # Número de hormigas
max_it = 100                           # Número de iteraciones
evap_var = 0.3                         # Tasa de evaporación de feromonas
Qfer = 1                               # Cantidad de feromonas a depositar
alpha = 1                              # Exponente para feromona
beta = 1                               # Exponente para distancia
best_tour = [None, float('inf')]       # Mejor camino conseguido
n_cities = distance_matrix.shape[0]    # Longitud del camino total