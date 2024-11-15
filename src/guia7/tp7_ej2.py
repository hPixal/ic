import numpy as np

def Datos(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

def next_city(nodos_disponibles, ind_actual, conexiones, feromonas, alpha, beta):
    
    ## Vector sobredimensionado para simplificar la estructura
    N = len(conexiones)
    prob = np.zeros(N)

    ## Suma del denominador
    suma = sum([(feromonas[ind_actual][aux]**alpha)*(1/conexiones[ind_actual][aux])**beta for aux in nodos_disponibles])

    ## Recorremos cada uno de los valores del set
    for next in nodos_disponibles: 
        ## Probabilidad de elegir el camino
        prob[next] = ((feromonas[ind_actual][next]**alpha)*(1/conexiones[ind_actual][next])**beta)/suma
    
    total_prob = sum(prob) # Suma de todas las probabilidades
    num = np.random.rand()*total_prob # Random entre 0 y total_prob
    acum = 0 # Acumulador de probabilidad
    
    ## Recorremos el set
    for next in nodos_disponibles:
        acum += prob[next] # Agregamos la probabilidad del camino next
        
        ## Si el numero es menor al acumulado hasta el momento retornamos el indice actual
        if (num <= acum):
            return next
    return -1

def distancia(camino,conexiones):
    suma = 0
    for i in range(len(camino)-1):
        suma += conexiones[camino[i]][camino[i+1]]
    return suma

def func_delta(funcion, Q, conexiones,camino,longitud):  
    delta = np.zeros_like(conexiones)
    ## Metodo global
    if funcion == 1:
        for i in range(len(camino)-1):
            ## indices
            ind_act = camino[i]
            ind_sig = camino[i+1]

            ## Simetria
            delta[ind_act][ind_sig] += Q/longitud
            delta[ind_sig][ind_act] += Q/longitud

    ## Metodo uniforme
    elif funcion == 2:
        for i in range(len(camino)-1):
            ## indices
            ind_act = camino[i]
            ind_sig = camino[i+1]

            ## Simetria
            delta[ind_act][ind_sig] += Q
            delta[ind_sig][ind_act] += Q
    
    ## Metodo local
    else:
        for i in range(len(camino)-1):
            ## indices
            ind_act = camino[i]
            ind_sig = camino[i+1]

            peso = conexiones[ind_act][ind_sig]
            ## Simetria
            delta[ind_act][ind_sig] += Q/peso
            delta[ind_sig][ind_act] += Q/peso
    return delta
        

def colonia(cant_hormigas,conexiones,hormiguero,alpha,beta,p,Q,d_func,max_epocas):
    # --- Aclaraciones ---
    # hormiguero = nodo inicial
    # d_func (deposito de feromonas):
    #   1 -> Global
    #   2 -> Uniforme
    #   3 -> Local


    N = len(conexiones)
    feromonas = 0.01*np.random.rand(N,N)

    ## Le damos la simetria correspondiente a las feromonas
    for i in range(N):
        feromonas[i][i] = 0
        for j in range(i+1,N):
            feromonas[j][i] = feromonas[i][j]

    hormigas = np.empty((cant_hormigas,N+1),int)
    longitudes = np.zeros(cant_hormigas)

    epoca = 0
    while (epoca < max_epocas):
        
        ## Recorremos las hormigas
        for k in range(cant_hormigas):

            ## Camino tiene la cantidad de nodos mas 1 (al repetir el final)
            camino = np.empty(N+1,int)
            camino[0] = hormiguero # Nodo inicial

            ## Inicializamos un set de nodos
            nodos_disponibles = set(range(0,N))
            nodos_disponibles.remove(hormiguero) # Ya recorrimos el inicial

            for i in range(1,N+1):
                ## Si es la ultima iteracion volvemos al nodo inicial
                if i == N:
                    camino[N] = hormiguero
                else:
                    ## Calculamos el nuevo paso
                    next = next_city(nodos_disponibles,camino[i-1],conexiones,feromonas,alpha,beta)

                    ## Debug
                    if (next == -1):
                        print("--------- Error 203 (Nuevo nodo invalido) ----------")
                        return "Error 203"
                    
                    camino[i] = next # Agregamos el nodo al camino
                    nodos_disponibles.remove(next) # Eliminamos el nodo de los caminos disponibles
            
            ## Guardamos el camino
            hormigas[k] = camino

            ## Calculamos la long del camino recorrido
            longitudes[k] = distancia(camino,conexiones)
            
        ## Depositamos feromonas
        feromonas = (1-p)*feromonas
        for i in range(cant_hormigas):
            delta = func_delta(d_func,Q,conexiones,hormigas[i],longitudes[i])
            feromonas += delta
        #print(feromonas)
        epoca += 1 # Contabilizamos la epoca
        #print(f'min long: {np.min(longitudes)}')

    ## Retornamos el camino de mayor longitud
    print(f'epoca: {epoca}')
    print(f'min long: {np.min(longitudes)}')
    return hormigas[np.argmin(longitudes)]

hormigas = 30
data = Datos('gr17.csv')
hormiguero = 0
alpha = 1
beta = 1
p = 0.3
Q = 1
d_func = 2
max_epocas = 200

print(colonia(hormigas,data,hormiguero,alpha,beta,p,Q,d_func,max_epocas))