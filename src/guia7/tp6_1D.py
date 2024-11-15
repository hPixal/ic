import numpy as np
import random

## ---- Codificacion de los datos ---- ##
# Se usa el primer bit como bit de signo [0] = positivo
# Se usan 9 bits para la parte entera 2^9 = 512
# Se usan 5 bits para la parte decimal
# Por lo tanto, genotipo = 15 bits

def init(cant_individuos,cant_bits):
    # vector de cadenas de bits totalmente randoms
    poblacion = [''.join(random.choices('01',k=cant_bits)) for _ in range(cant_individuos)]
    return poblacion


def decodif(poblacion):
    N = len(poblacion)
    poblacion_deco = np.empty(N)

    for i in range(N):
        # Parte entera
        poblacion_deco[i] = int(poblacion[i][1:-5],2)
        
        # Parte fraccionaria
        fracc = 0
        for j in range(5,0,-1):
            fracc += int(poblacion[i][-j])*(2**(j-6))
        poblacion_deco[i] += fracc

        # Extension de signo
        if poblacion[i][0] == '1':
            poblacion_deco[i] *= -1
    
    return(poblacion_deco)

def aptitud(poblacion_deco):
    p = np.array(poblacion_deco) # Auxiliar para soportar la operacion vectorial
    return p*np.sin(np.sqrt(abs(p))) # La funcion f pero son el signo opuesto
        
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

    gen = 0 # Contador de generacion
    max_apt = -1e15 # Aptitud maxima de la generacion anterior
    estabilidad = 0 # Contador de estabilidad
    while (gen < max_gen):
        
        poblacion_deco = decodif(poblacion) # Decodificamos el genotipo
        apt = aptitud(poblacion_deco) # Calculamos la aptitud

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

p = evolucion(1000,20,0.1,0.7,200,15)
print(p[0:15])

# Con individuos > 200 anda perfecto pero no se porque necesita tanto