import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import SOM

def import_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data


circulo = import_data("circulo.csv")
t = import_data("te.csv")


def circulo_som(circulo):
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    ACCk = np.zeros(k)
    som = SOM.SOM(10, 10, 2, 0.2)
    
    for fold, (train_index, test_index) in enumerate(kf.split(circulo)):
        # Crear conjuntos de entrenamiento y prueba para esta iteración
        x_train, x_test = circulo[train_index], circulo[test_index]
        _, y_test = circulo[train_index], circulo[test_index]
        
        som.learn(x_train, True)
        y_pred = som.predict(x_test)

        ACCk[fold] = accuracy_score(y_test,y_pred)

    return som


def t_som(t):
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=44)
    # ACCk = np.zeros(k)
    som = SOM.SOM(10, 10, 2, 0.2)
    
    for fold, (train_index, test_index) in enumerate(kf.split(t)):
        # Crear conjuntos de entrenamiento y prueba para esta iteración
        x_train, x_test = t[train_index], t[test_index]
        _, y_test = t[train_index], t[test_index]
        
        
        som.learn(x_train, True)
        # ACCk[fold] = accuracy_score(y_test,y_pred)

    return som

# circulo_som(circulo)
t_som(t)

