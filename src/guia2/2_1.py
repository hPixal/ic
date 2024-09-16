import random
import numpy as np
import matplotlib.pyplot as plt

def load_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

def sigmoid(v):
    return ((2/(1+np.exp(-v)))-1)

def prime_sigmoid(y):
    return ((1+y)*(1-y))

# INICIALIZACIONES
# defininimos los datos de la siguiente manera:
# - Un arreglo que va a relacionar las capas de la red 
#   con la cantidad de entradas que tiene 
# - Una variable que contiene la cantidad de capas
# - Una variable que contiene la la cantidad de salidas

vec_num_entradas = np.array([ 2,2,1 ])
num_capas = len(vec_num_entradas)
num_salidas = vec_num_entradas[num_capas-1]

# cargo los datos del csv

data = load_csv('XOR_trn.csv')

# obtengo el vector de entradas y salidas deseadas vacio

x = np.empty(len(data),dtype=object) 
y_d = np.empty(len(data),dtype=object)

x = data[:,0:vec_num_entradas[0]]; 
y_d = data[:,vec_num_entradas[0]];

# agego el umbral/sesgo

x = np.hstack((-np.ones(len(x)).reshape(len(x),1),x)) # magia
y = np.empty(len(x)-1,dtype=object)

# ahora tengo los datos cargados de la forma
# x = [ -1, x1, x2 
#       -1, x1, x2 ]; los -1 son los biases

# inicio los pesos

w = []

for i in range(len(vec_num_entradas)-1):
    # Genera una matriz de alto igual al numero de entradas de la siguiente capa y
    # ancho igual al numero de entradas de la capa actual +1 (bias) y lo mete en un vector
    w.append(np.random.rand(vec_num_entradas[i+1],vec_num_entradas[i]+1))

# El elemento w_ij = w[i][j] de la matriz me da los pesos correspondientes a la neurona j de la capa i. 
# El elemento w_ij[k] me da el peso asociado a la entrada k de la neurona j de la capa i.

# los datos de arranque seran:
# nro_epoca 
# max_epoca
# lr 
# umbral_error

epoca = 1
max_epoca = 100
lr = 0.1
umbral_error = 0.01
delta = np.empty(num_capas-1)

while  epoca <= max_epoca:
    
    # APRENDIZAJE
    # propagacion hacia adelante
    for i in range(len(x)):
        entradas = x[i] 
        for j in range(num_capas-1):
            V = np.dot(w[j],entradas)
            y[j] = sigmoid(V)
            entradas = np.hstack((-1,y[j]))
            
        # propagacion hacia atras
        error = y_d[i] - y[-1]
        delta[-1] = error*(1/2)*prime_sigmoid(y[-1])     # Con el error, obtengo el delta de la capa de salida
        for i in range(num_capas-2,0,-1):
            w_i = w[i][:,1:].T                          # No tomo el peso w0 (umbral) porque no tiene delta para propagar
            d = np.dot(w_i,delta[i])
            delta[i-1] = d*(1/2)*prime_sigmoid(y[i-1])  # Con los pesos de la capa i obtenemos el delta de la capa i-1
        
        # actualizacion de los pesos
        entradas = x[i]
        for j in range(num_capas):
            w[j] = w[j] + lr*delta[j]*entradas

    # EVALUACION
    cant_error = 0
    cont_mse = 0
    for i in range(len(x)):
        entradas = x[i] 
        for j in range(num_capas):
            y[j] = sigmoid(np.dot(w[j],entradas))
            entradas = np.hstack((-1,y[j]))
            
        error = y_d[i] - y[-1]
        cant_error = cant_error + error**2
        cont_mse = cont_mse + 1
    mse = cant_error/cont_mse
    if mse < umbral_error:
        break
    epoca = epoca + 1









# CARGA DE DATOS