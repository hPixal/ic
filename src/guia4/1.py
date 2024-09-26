from sklearn.model_selection import KFold
import numpy as np
import SOM as SOM

def import_circles():
    X = np.genfromtxt('circulo.csv', delimiter=',')
    return X

def import_te():
    X = np.genfromtxt('te.csv', delimiter=',')
    return X

def import_iris():
    data = np.loadtxt("your_iris_data.csv", delimiter=",")
    X = data[:, :4]  
    Y = data[:, 4:] 
    return X,Y

def ejercicio1():
    # CONFIGURACION ---------------------------------------------------------
    
    # Importar datos para entrenar
    X_circ = import_circles()   # Datos circulo
    X_te = import_te()          # Datos T
    epochs = 2000               # Epocas
    lr = 0.2                    # Taza de aprendizaje
    
    # ENTRENAR MODELO CIRCULO -----------------------------------------------
    # 1 | Crear modelo
    modelCircle = SOM.SOM(6,6,2,lr=lr,radius=None,epochs=epochs)
    
    # 2 | Entrenar modelo
    modelCircle.learn(X_circ,visualize=True)
    
    # ENTRENAR MODELO T ----------------------------------------------------
    # 1 | Crear modelo
    modelTE = SOM.SOM(6,6,2,lr=lr,radius=None,epochs=epochs)
    
    # 2 | Entrenar modelo
    modelTE.learn(X_te,visualize=True)
    
ejercicio1()