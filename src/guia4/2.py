from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
import copy
import numpy as np
from sklearn.model_selection import KFold
import KMeans
import SOM as SOM

def import_circles():
    X = np.genfromtxt('circulo.csv', delimiter=',')
    return X

def import_te():
    X = np.genfromtxt('te.csv', delimiter=',')
    return X

def import_iris():
    data = np.loadtxt("irisbin_trn.csv", delimiter=",")
    X = data[:, :4]  
    Y = data[:, 4:] 
    #data = load_iris()
    #X = data.tada
    
    return X,Y

def ejercicio2():
    # CONFIGURACION ---------------------------------------------------------
    
    # Importar datos para entrenar
    X_iris,_ = import_iris()        # Datos iris
    epochs = 2000                   # Epocas
    lr = 0.1                        # Taza de aprendizaje
    k = 3                           # Cantidad de clusters  
    tolerance=1e-4                  # Tolerancia para actualizar centroides
    # ENTRENAR MODELO IRIS K-MEDIAS -----------------------------------------
    # 1 | Crear modelo
    modelKM = KMeans.KMeans(k=k, max_iters=epochs, tolerance=tolerance)
    
    # 2 | Entrenar modelo
    modelKM.fit(X_iris)
    
    # ENTRENAR MODELO IRIS SOM ---------------------------------------------
    # 1 | Crear modelo
    modelTE = SOM.SOM(6,6,4,lr=lr,radius=None,epochs=epochs)
    
    # 2 | Entrenar modelo
    modelTE.learn(X_iris,visualize=True)
    
    

    
ejercicio2()
    