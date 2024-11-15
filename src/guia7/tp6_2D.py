import numpy as np
import random

## ---- Codificacion de los datos ---- ##
# Se usa el primer bit como bit de signo [0] = positivo
# Se usan 7 bits para la parte entera 2^7 = 128
# Se fuerzan los valores a no ser mayores a 100
# Se usan 5 bits para la parte decimal
# Por lo tanto, genotipo = 13 bits

def init(cant_individuos,cant_bits):
    # vector de cadenas de bits totalmente randoms
    poblacion = [''.join(random.choices('01',k=cant_bits)) for _ in range(cant_individuos)]
    return poblacion

def decodif2D(poblacion):
    N = len(poblacion)
    poblacion_deco = np.empty(N,object)

    for i in range(N):
        # Hay que decofificar las 2 coordenadas
        x = 0
        y = 0

        # Para no repetir
        indiv = poblacion[i]

        ## ------- Coordenada X ------- ##
        # Parte entera
        x = int(indiv[1:8],2)
        
        # Parte fraccionaria
        fracc = 0
        for j in range(5):
            fracc += int(indiv[j+8])*(2**(-j-1))
        x += fracc

        # Extension de signo
        if indiv[0] == '1':
            x *= -1
        ## ---------------------------- ##
        
        ## ------- Coordenada Y ------- ##
        # Parte entera
        y = int(indiv[14:21],2)
        
        # Parte fraccionaria
        fracc = 0
        for j in range(5):
            fracc += int(indiv[j+21])*(2**(-j-1))
        y += fracc

        # Extension de signo
        if indiv[13] == '1':
            y *= -1
        ## ---------------------------- ##

        poblacion_deco[i] = [x, y]
    

    return(poblacion_deco)

def aptitud2D(poblacion_deco):
    N = len(poblacion_deco)
    apt = np.empty(N)
    for i in range(N):
        x = poblacion_deco[i][0]
        y = poblacion_deco[i][1]
        # apt = -f(x,y)
        apt[i] = -((x**2+y**2)**0.25)*(np.sin(50*(x**2+y**2)**0.1)**2+1)
    return apt

def entorno(poblacion):
    for indiv in poblacion:
        pj_aux = list(indiv) # Auxiliar para operar

        # Si la coordenada x es mayor a 99 se hace igual a 99
        if int(indiv[1:8],2) > 99:
            pj_aux[1:8] = '1100011'
        
        # Si la coordenada y es mayor a 99 se hace igual a 99
        if int(indiv[14:21],2) > 99:
            pj_aux[14:21] = '1100011'

        indiv = ''.join(pj_aux)
    return poblacion

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


def evolucion(max_gen, max_estabilidad, prob_muta, prob_cruza, cant_individuos, bits):
    
    cant_proge = round(cant_individuos*0.3) # Calculamos la cantidad de progenitores por generacion
    poblacion = init(cant_individuos,bits) # Creamos la generacion inicial completamente al azar
    poblacion = entorno(poblacion) # Corregimos los limites

    gen = 0 # Contador de generacion
    max_apt = -1e15 # Aptitud maxima de la generacion anterior
    estabilidad = 0 # Contador de estabilidad
    while (gen < max_gen):

        poblacion_deco = decodif2D(poblacion) # Decodificamos el genotipo
        apt = aptitud2D(poblacion_deco) # Calculamos la aptitud

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
        poblacion = entorno(poblacion) # Corregimos los limites
        gen += 1

p = evolucion(1000,20,0.1,0.7,200,26)
print(p[0:15])
# Con individuos > 200 anda perfecto pero no se porque necesita tanto