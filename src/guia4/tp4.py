import numpy as np
import matplotlib.pyplot as plt

def Datos(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

def Pesos(dim,entradas):
    w = np.empty(dim,object) # creamos la matriz de vectores
    for fila in w:
        for i in range(len(fila)):
            # Inicializamos los pesos de la neurona [fila][columna i]
            # con un vector del tamanio de las entradas
            fila[i] = np.random.rand(entradas)-0.5
    return w

def distancia(x,w):
    dist = np.empty(w.shape,object) # Matriz que almacena las distancias
    for fila in range(len(w)):
        for colum in range(len(w[fila])):
            aux = x-w[fila][colum]
            dist[fila][colum] = np.dot(aux,aux) # dist**2
    return dist

def SOMTrain(x,w,max_epocas,neu_vec,press):
    
    u = 0.85 #const de aprendizaje
    if (neu_vec > 1): trans = 1000/(neu_vec-1) # Cada cuantas epocas debo disminuir la cantidad de 
    else: trans = 1000                         # vecinos (suponiendo que seran 1000 de transicion)

    filas = len(w)
    columnas = len(w[0])
    w_ant = w

    epoca_actual = 0
    while (epoca_actual < max_epocas):

        epoca_actual += 1
        print(epoca_actual)
        # Etapa de transicion
        if epoca_actual >= 1000 and epoca_actual < 2000:
            u -= 0.00075 # ui-0.1/1000
            if (neu_vec > 1 and epoca_actual%trans == 0):
                neu_vec -= 1

        # Etapa de convergencia
        elif epoca_actual == 2000:
            u = 0.01
            neu_vec = 0
        
        # Criterio de parada de la etapa de convergencia
        elif epoca_actual > 2000:
            ## Sumatoria de todos los elementos de una matriz de vectores
            var = np.sum(np.sum((w - w_ant)**2)) 
            print(var)
            tol = 1 - press # Tolerancia

            if var < tol: # Condicion de corte
                print("Todo joya rey")
                return w


        w_ant = w #nos guardamos el w anterior

        for i in range(len(x)):
            dist = distancia(x[i],w) # distancia del patron a cada neurona
            ind_min = np.argmin(dist) # indice plano
            f, c = np.unravel_index(ind_min, dist.shape) # indice en fila y columna
            
            # Indices validos para el entorno cuadrado
            fila_min = max(0, f - neu_vec)
            fila_max = min(filas - 1 , f + neu_vec)
            column_min = max(0, c - neu_vec)
            column_max = min(columnas - 1 , c + neu_vec)

            e = x[i]-w[f][c] # Error cometido

            # Actualizamos la vecindad
            for fil in range(fila_min,fila_max+1):
                for col in range(column_min,column_max+1):
                    w[fil][col] += u*(x[i]-w[fil][col])
    print("Todo mal wacho")
    return w    
                    
            
## Carga de datos
circulo = Datos("circulo.csv")
te = Datos("te.csv")


### ---------------- Circulo ---------------- ###

# Creamos la figura
plt.figure("Circulo")

## --------------- Plot de los datos
plt.scatter(circulo[:,0],circulo[:,1],color="blue")
## ---------------

## Entrenamiento
wc = Pesos([2, 2],2)
wc = SOMTrain(circulo,wc,4000,1,0.99)

# --------------- Plot de los resultados
# Matriz desarmada
wx = np.zeros(wc.shape)
wy = np.zeros(wc.shape)

## Desarmado
for i in range(len(wc)):
    for j in range(len(wc[i])):
        wx[i][j] = wc[i][j][0]
        wy[i][j] = wc[i][j][1]
    
    # Plot de las uniones por fila
    plt.plot(wx[i],wy[i],color="red")

for i in range(len(wx.T)):
    # Plot de las uniones por columna
    plt.plot(wx.T[i],wy.T[i],color="red")

# Plot de las "neuronas"
plt.scatter(wx.ravel(),wy.ravel(),color="red")

# ---------------

# Añadir etiquetas y título
plt.title('Circulo')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel('X1')
plt.ylabel('X2')

# Mostrar el gráfico
plt.grid(True)
plt.show()


### ---------------- Te ---------------- ###

# Creamos la figura
plt.figure("Te")

## --------------- Plot de los datos
plt.scatter(te[:,0],te[:,1],color="blue")
## ---------------

## Entrenamiento
wt = Pesos([5, 5],2)
wt = SOMTrain(te,wt,4000,3,0.99)

# --------------- Plot de los resultados
# Matriz desarmada
wtx = np.zeros(wt.shape)
wty = np.zeros(wt.shape)

## Desarmado
for i in range(len(wt)):
    for j in range(len(wt[i])):
        wtx[i][j] = wt[i][j][0]
        wty[i][j] = wt[i][j][1]
    
    # Plot de las uniones por fila
    plt.plot(wtx[i],wty[i],color="red")

for i in range(len(wtx.T)):
    # Plot de las uniones por columna
    plt.plot(wtx.T[i],wty.T[i],color="red")

# Plot de las "neuronas"
plt.scatter(wtx.ravel(),wty.ravel(),color="red")

# ---------------

# Añadir etiquetas y título
plt.title('Te')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel('X1')
plt.ylabel('X2')

# Mostrar el gráfico
plt.grid(True)
plt.show()