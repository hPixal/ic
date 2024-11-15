import numpy as np
import matplotlib.pyplot as plt
import random

'''
Algoritmo: Enjambre de Partículas (gEP)

Variables:
- n_particulas: Número de partículas
- x_k(t): Posición de la partícula k en el tiempo t
- v_k(t): Velocidad de la partícula k en el tiempo t
- y_k: Mejor posición individual de la partícula k
- y_hat: Mejor posición global (entre todas las partículas)
- f(x): Función de evaluación (objetivo a minimizar)
- c1, c2: Coeficientes de aceleración (factores de aprendizaje)
- r1, r2: Valores aleatorios entre 0 y 1
- minimos, maximos: Límites inferior y superior de la búsqueda

1. Inicialización:
    Para cada partícula k = 1, 2, ..., n_particulas:
        1.1. Inicializar la posición x_k(0) aleatoriamente entre [minimos, maximos]
        1.2. Inicializar la velocidad v_k(0) con un valor aleatorio
        1.3. Establecer la mejor posición personal y_k = x_k(0)

2. Repetir hasta que se cumpla la condición de finalización:
    Para cada partícula k = 1, 2, ..., n_particulas:
    
        2.1. Evaluar la función objetivo para la partícula k:
            Si f(x_k(t)) < f(y_k), entonces:
                - y_k = x_k(t)  (Actualizar la mejor posición personal)
            Si f(y_k) < f(y_hat), entonces:
                - y_hat = y_k  (Actualizar la mejor posición global)

        2.2. Actualizar la velocidad de la partícula k:
            v_k(t+1) = v_k(t) + c1 * r1 * (y_k - x_k(t)) + c2 * r2 * (y_hat - x_k(t))
        
        2.3. Actualizar la posición de la partícula k:
            x_k(t+1) = x_k(t) + v_k(t+1)

3. Condición de finalización:
    - Repetir hasta que se cumpla un criterio de parada
        - Máximo número de iteraciones o
        - Tolerancia pequeña en el valor de f(y_hat)

4. Devolver la mejor solución:
    - Devolver y_hat, que es la mejor posición encontrada por las partículas.
'''
def f2(x):
    return ((x[0]**2 + x[1]**2)**0.25) * ((np.sin(50*(x[0]**2 + x[1]**2)**0.1))**2 + 1)

def f1(x):
    return -x * np.sin(np.sqrt(abs(x)))

def inicializacion(n_particulas, N, minimos, maximos, funcion_error):
    x_k   = np.zeros((n_particulas,N))   # x_k(t): Posición de la partícula k en el tiempo t
    v_k   = np.zeros((n_particulas,N))   # v_k(t): Velocidad de la partícula k en el tiempo t
    y_k   = np.zeros((n_particulas,N+1)) # y_k: Mejor posición individual de la partícula k y valor
    y_hat = np.zeros(N+1)                # y_hat: Mejor opción global (entre todas las partículas) y valor
    
    for k in range(n_particulas): # Para cada partícula k
        for j in range(N):
            x_k[k][j] = np.random.uniform(minimos[j], maximos[j])
            v_k[k][j] = np.random.uniform(-20, 20)
            y_k[k][j] = x_k[k][j]
            
        y_k[k][N] = funcion_error(x_k[k])  # Guardar el valor de la función objetivo en la última columna de y_k
    y_hat[:] = y_k[np.argmin(y_k[:, -1])]  # Encontrar la mejor partícula global

    return x_k, v_k, y_k, y_hat

def algoritmo_enjambre_particulas(n_particulas, minimos, maximos, funcion_error, r0, r1, c0, c1, max_iter):
    N = len(minimos)
    x_k, v_k, y_k, y_hat = inicializacion(n_particulas, N, minimos, maximos, funcion_error)
    
    # Historico de y_hat
    historico_y_hat = np.zeros((max_iter, N+1))
    
    for i in range(max_iter):
        for k in range(n_particulas):
            # Actualizar posición
            for j in range(N):
                x_k[k][j] = x_k[k][j] + v_k[k][j]
            
            # Actualizar velocidad
            v_k[k] = v_k[k] + c0 * r0 * (y_k[k][:N] - x_k[k]) + c1 * r1 * (y_hat[:N] - x_k[k])
            
            # Evaluar la nueva posición
            y_k[k][N] = funcion_error(x_k[k])
            
            # Actualizar mejor posición global si es necesario
            if y_k[k][N] < y_hat[N]:
                y_hat[:] = y_k[k]  # Actualizar el mejor global
        historico_y_hat[i] = y_hat
    
    return y_hat, historico_y_hat

def ejercicio2():
    n_particulas = 100
    max_iter = 5000
    minimos = [-100, -100]
    maximos = [100, 100]
    funcion_error = f2
    r0 = np.random.rand()
    r1 = np.random.rand()
    c0 = 2
    c1 = 2
    print("Ejercicio 2:")
    y_hat, historico_y_hat = algoritmo_enjambre_particulas(n_particulas, minimos, maximos, funcion_error, r0, r1, c0, c1, max_iter)
    print(f"y_hat: {y_hat}")
    
    error_values = historico_y_hat[:, -1]

    plt.figure(figsize=(10, 6))
    plt.plot(range(max_iter), error_values, label="y_hat")
    plt.title("Reducción de y_hat a lo largo de la iteración", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("y_hat", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def ejercicio1():
    n_particulas = 50
    max_iter = 5000
    minimos = [-100]
    maximos = [100]
    funcion_error = f1
    r0 = np.random.rand()
    r1 = np.random.rand()
    c0 = 2
    c1 = 2
    print("Ejercicio 1:")
    y_hat, historico_y_hat = algoritmo_enjambre_particulas(n_particulas, minimos, maximos, funcion_error, r0, r1, c0, c1, max_iter)
    print(f"y_hat: {y_hat}")
    
    error_values = historico_y_hat[:, -1]

    plt.figure(figsize=(10, 6))
    plt.plot(range(max_iter), error_values, label="y_hat")
    plt.title("Reducción de y_hat a lo largo de la iteración", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("y_hat", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
    
ejercicio1()
ejercicio2()