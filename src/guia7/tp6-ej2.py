import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import utils

## ---- Codificacion de los datos ---- ##
# Se usan 7129 bits para representar con 1 si el valor se cree relevante
# o 0 en el caso que no se cosidere relevante

def Datos(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

def init(cant_individuos,cant_bits):
    # vector de cadenas de bits totalmente randoms
    poblacion = [''.join(random.choices('01',k=cant_bits)) for _ in range(cant_individuos)]
    return poblacion

def reducir_datos(data,poblacion):
    N = len(poblacion) # Cantidad de individuos
    M = len(data[0]) # Longitud de cada muestra
    K = len(data) # Cantidad de muestras

    # Obtenemos los indices que debemos conservar por cada individuo
    rangos = np.empty(N,object)
    for i in range(N):
        indices = []
        indiv = poblacion[i]
        for j in range(M):
            if indiv[j] == '1':
                indices.append(j)
        rangos[i] = indices
    
    # Obtenemos los datos con la estructura de cada individuo
    conjuntos = []
    for i in range(N):
        indiv = np.empty((K,len(rangos[i])),float) # Muestras redimensionadas
        for j in range(K):
            indiv[j] = np.array([data[j][k] for k in rangos[i]])
        conjuntos.append(indiv)

    return conjuntos

def aptitud(poblacion,yd):
    # Aca hay que entrenar un algoritmo de clasificacion
    # La aptitud es ver a cuanto le pega

    N = len(poblacion) # Cantidad de individuos
    apt = np.empty(N)
    for i in range(N):
        
        print(f'individuo {i}')
        indiv = poblacion[i]

        ## Definimos el perceptron multicapa
        mlp = MLPClassifier(hidden_layer_sizes=(len(indiv[0]), 50),  # Capas ocultas con 32 y 16 neuronas
                activation='logistic',           # Función de activación
                solver='adam',               # Optimizador
                max_iter=500,                  # Número máximo de iteraciones
                random_state=42)             # Semilla para reproducibilidad
        
        x_train, x_test, y_train, y_test = train_test_split(indiv, yd, test_size=0.2, random_state=42)

        # Entrenar el modelo
        mlp.fit(x_train, y_train)
        
        y_pred = mlp.predict(x_test)
        apt[i] = accuracy_score(y_test,y_pred)

    return apt

def ventana(aptitudes,cant_proge):
    # Recibe el vector de aptitudes y devulve un
    # vector de indices a los individuos seleccionados

    indices = np.argsort(aptitudes) # Esto devuelve los indices de las
                                    # aptitudes ordenadas en forma asc.

    reduccion_ventana = round(len(indices)/cant_proge)
    proge = np.empty(cant_proge,int)
    for i in range(cant_proge):
        proge[i] = random.choice(indices[(reduccion_ventana*i):])
    
    return proge

def cruza(padre,madre,prob):
    if (prob > np.random.rand()):
        corte = np.random.randint(len(padre))
        hijo = madre[0:corte] + padre[corte:]
        hija = padre[0:corte] + madre[corte:]
    else:
        hijo = padre
        hija = madre
    return hijo, hija


def mutacion(nueva_poblacion,prob):
    bits = len(nueva_poblacion[0])

    for i in range(len(nueva_poblacion)):
        if (prob > np.random.rand()):
            pj_aux = list(nueva_poblacion[i]) # Auxiliar para poder modificar un char
            ind = np.random.randint(bits)
            if (pj_aux[ind] == '1'):
                pj_aux[ind] = '0'
            else:
                pj_aux[ind] = '1'

            nueva_poblacion[i] = ''.join(pj_aux) # Insertamos el auxiliar modificado

    return nueva_poblacion

def evolucion(max_gen, max_estabilidad, prob_muta, prob_cruza, cant_individuos, data):
    
    # Mezclamos los datos porque vienen ordenados
    data = utils.shuffle(data)

    x = data[:,:-1] # Caracteristicas a filtrar
    yd = np.array(data[:,-1]) # Salida esperada

    bits = len(x[0])

    cant_proge = round(cant_individuos*0.3) # Calculamos la cantidad de progenitores por generacion
    poblacion = init(cant_individuos,bits) # Creamos la generacion inicial completamente al azar

    gen = 0 # Contador de generacion
    max_apt = -1e15 # Aptitud maxima de la generacion anterior
    estabilidad = 0 # Contador de estabilidad
    while (gen < max_gen):
        
        print(gen)

        poblacion_deco = reducir_datos(x,poblacion) # reducimos la dimencion de entrada
        apt = aptitud(poblacion_deco,yd) # Calculamos la aptitud

        max_apt_act = max(apt) # Maxima aptitud actual
        
        # Si no hay casi diferencia entre la aptitud actual y la anterior contabilizamos
        if np.isclose(max_apt_act,max_apt):
            # Contabilizamos generacion estable
            estabilidad += 1
        else:
            # Reiniciamos el contador
            estabilidad = 0

        ## Si van x generaciones estables cortamos
        if estabilidad >= max_estabilidad:
            print(f'EL entrenamiento termino en la generacion: {gen}, con una aptitud de {max_apt_act}')
            return poblacion
        ## -----------
        
        max_apt = max_apt_act # Maxima aptitud de la iteracion anterior

        # Reiniciamos el ciclo
        progenitores = ventana(apt,cant_proge) # Obtenemos los progenitores para la nueva generacion

        # Creamos la nueva generacion
        nuevagen = np.empty(cant_individuos,object)
        for i in range(0,cant_individuos,2):
            # Se elige padre y madre entre los posibles progenitores
            padre = poblacion[random.choice(progenitores)]
            madre = poblacion[random.choice(progenitores)]

            nuevagen[i],nuevagen[i+1] = cruza(padre,madre,prob_cruza) # El delicioso

        poblacion = mutacion(nuevagen,prob_muta) # Delicioso entre primos
        gen += 1


data = Datos('leukemia_train.csv')
p = evolucion(1000,20,0.1,0.7,50,data)
