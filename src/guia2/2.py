import numpy as np
import matplotlib.pyplot as plt

def Datos(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

def Pesos(dim,entradas):
    capas = len(dim)
    # Lista de matrices
    w = np.empty(capas,object)
    w[0] = np.random.rand(dim[0],entradas+1)-0.5
    for n in range(1,capas):
        # Creamos y pushbackeamos matrices de n filas, capas columnas
        w[n] = np.random.rand(dim[n],dim[n-1]+1)-0.5
    return w

def sigmoide(v):
    if (np.isscalar(v)):
        return np.array([(2/(1+np.exp(-v)))-1])
    else:
        y = np.empty(0)
        for x in v:
            y = np.append(y, (2/(1+np.exp(-x)))-1)
        return y

def MultiCapaTrain(w,data,epocas,entradas,salidas):
    capas = len(w) # Se declara la cantidad de capas, para los indices vamos a tener que restarle uno
    x = data[:,0:entradas]  # se las entradas, data es una matriz  te longitud n, se toma de 0 a entradas
    yd = data[:,range(entradas,entradas+salidas)] # y deseada es la ultima columna de la matriz data
    
    # Constante de aprendizaje
    u = 0.05

    ## Epocas

    count = 0
    while (count < epocas): # Mientras no nos pasemos de la cantidad de epocas predefinidas hacer
        count += 1 # Contabilizamos la Epoca
        for i in range(len(x)): # Por cada patron aplicar el forward a cada capa de la red neuronal
            
            ### ----------- Forward ----------- ###
            yi = np.empty(capas,object) # Vector de salidas por capa, tendra el largo de la cantidad de capas
            xi = np.empty(capas,object) # Vector de entradas por capa 

            xi[0] = np.insert(x[i],0,-1) # Entrada capa 0 se añade -1 por los biases
            v0 = np.dot(w[0],xi[0])  # Resultado capa 0 se hace producto punto de los w y las entradas de la capa 0
                                                    # aun no es un forward porque no aplicamos la sigmoidea a los resultados
            
            # Corregimos la salida
            yi[0] = sigmoide(v0) # Salida capa 0 ahora si habiendo aplicado la sigmoidea

            # Demas capas
            for j in range(1,capas): # Repetir el mismo proceso para el resto de capas 
                xi[j] = np.insert(yi[j-1],0,-1) # TODO: CORREGIR
                vj = np.dot(w[j],xi[j])
                yi[j] = sigmoide(vj)

            # Ahora ya tenemos las salidas de las capas, en la matriz, la ultima será la salida de la red
            ### ----------- Backward ----------- ###
            # Calcular los delta
            dW = np.zeros_like(w)   # Inicializamos un vector que va a contener los valores por los 
                                                    # que vamos a corregir los pesos
            dd = np.zeros_like(yi)   # Almacenamos los deltas

            ## Salida
            capaF = capas-1 # Para no repetir
            dd[capaF] = (yd[i]-yi[capaF])*(1+yi[capaF])*(1-yi[capaF])*(1/2) # delta^2 ultima capa

            # Calculamos todos los delta
            for j in range(capas-1,0,-1): #  Recorremos en reversa la red neuronal 
                wj = w[j][:,1:].T                    #  Transponemos el renglon de pesos correspondiente a la capa
                dd[j-1] = np.dot(wj,dd[j])*(1+yi[j-1])*(1-yi[j-1])*(1/2) # Almacenamos los delta^2 calculados
            
            # Calculamos la actualizacion de pesos
            for j in range(capas):
                dW[j] = u*np.outer(dd[j],xi[j]) # Creamos la matriz  para corregir la matriz de pesos original

            ## Actualizar los pesos
            w = w + dW 
        
        ### ----------- Chequeo de sol ----------- ### 
        # Aca hacemos un forward con la data y comparamos la salida esperada con la salida de la red
        correct = 0
        for i in range(len(x)):
            yi = np.empty(capas,object) # Vector de salidas por capa
            xi = np.empty(capas,object) # Vector de entradas a las capas

            xi[0] = np.insert(x[i],0,-1) # Entrada capa 0
            v0 = np.dot(w[0],xi[0]) # Resultado capa 0
            
            # Corregimos la salida
            yi[0] = sigmoide(v0) # Salida capa 0

            # Demas capas
            for j in range(1,capas):
                xi[j] = np.insert(yi[j-1],0,-1)
                vj = np.dot(w[j],xi[j])
                yi[j] = sigmoide(vj)
            
            correct += 1
            for k in range(len(yi[-1])):    # TODO: Corregir WINNER TAKE ALL
                if (yd[i][k]*yi[-1][k] < 0):
                    correct += -1
                    break
        
        presicion = correct/len(x)
        print("iteracion " + str(count) + ": " + str(presicion))
        if (presicion > 0.99):
            print("El algoritmo alcanzo una precision de: " + str(presicion*100) + "'%' en " + str(count) + " epocas, con los siguientes pesos:")
            for aux in w:
                print(aux)
            return w
        
    print("El algoritmo no alcanzo la precision deseada: " + str(presicion))
    for aux in w:
        print(aux)
    return w


def MultiCapaTest(w,data,entradas,salidas):
    capas = len(w)
    x = data[:,0:entradas]
    yd = data[:,range(entradas,entradas+salidas)]

    #prediccion = np.empty(len(x),object)
    correct = 0
    for i in range(len(x)):
        yi = np.empty(capas,object) # Vector de salidas por capa
        xi = np.empty(capas,object) # Vector de entradas a las capas

        xi[0] = np.insert(x[i],0,-1) # Entrada capa 0
        v0 = np.dot(w[0],xi[0]) # Resultado capa 0
        
        # Corregimos la salida
        yi[0] = sigmoide(v0) # Salida capa 0

        # Demas capas
        for j in range(1,capas):
            xi[j] = np.insert(yi[j-1],0,-1)
            vj = np.dot(w[j],xi[j])
            yi[j] = sigmoide(vj)
        
        #prediccion[i] = yi
        correct += 1
        for k in range(len(yi[-1])):
            if (yd[i][k]*yi[-1][k] < 0):
                correct += -1
                break
    
    presicion = correct/len(x)
    print("El algoritmo alcanzo una precision de: " + str(presicion*100) + "'%' en el test")
    #return prediccion

def ej1():
    # Ejercicio 1
    trn = Datos('XOR_trn.csv')
    tst = Datos('XOR_tst.csv')
    estr = [2,1]
    entradas = len(trn[0])-estr[-1]
    w = Pesos(estr,entradas)
    
    print("Pesos iniciales:")
    for x in w:
        print(x)
    print("")
    
    w = MultiCapaTrain(w,trn,100,entradas,estr[-1])
    MultiCapaTest(w,tst,entradas,estr[-1])

def ej2():
    # Ejercicio 2
    trn2 = Datos("concent_trn.csv")
    tst2 = Datos("concent_tst.csv")
    estr2 = [4,1]
    entradas2 = len(trn2[0])-estr2[-1]
    w2 = Pesos(estr2,entradas2)
    MultiCapaTest(w2,trn2,entradas2,estr2[-1])
    w2 = MultiCapaTrain(w2,trn2,3,entradas2,estr2[-1])
        # Crear el gráfico
    plt.figure()
    #
    for x in trn2:
        if (x[2] > 0):
            plt.scatter(x[0], x[1], color='blue', marker='x')
        else:
            plt.scatter(x[0], x[1], color='red')
           
    x_line = np.linspace(-1, 1, 100)
    y_line = - w2[0][0][2]/w2[0][0][1] * x_line + w2[0][0][0]/w2[0][0][1]
    
    plt.plot(x_line,y_line,color='black')
    
    y1_line = - w2[0][1][2]/w2[0][1][1] * x_line + w2[0][1][0]/w2[0][1][1]
    
    plt.plot(x_line,y1_line,color='green')
    y2_line = - w2[0][2][2]/w2[0][2][1] * x_line + w2[0][2][0]/w2[0][2][1]
    
    plt.plot(x_line,y2_line,color='black')
    
    y3_line = - w2[0][3][2]/w2[0][3][1] * x_line + w2[0][3][0]/w2[0][3][1]
    
    plt.plot(x_line,y3_line,color='green')
    #    Añadir etiquetas y título
    plt.title('Gráfico de Puntos de Prueba')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    #plt.legend()
    
    # Mostrar el gráfico
    plt.grid(True)
    plt.show()
    
def ej3():
    ### Ejercicio 3

    trn3 = Datos("irisbin_trn.csv")
    tst3 = Datos("irisbin_tst.csv")
    estr3 = [1,3]
    entradas3 = len(trn3[0])-estr3[-1]
    w3 = Pesos(estr3,entradas3)
    w3 = MultiCapaTrain(w3,trn3,100,entradas3,estr3[-1])

    MultiCapaTest(w3,tst3,entradas3,estr3[-1])
     
ej3()
