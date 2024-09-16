import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def cuadratic_error(y_d, y):
    acum = 0
    for i in range(len(y_d)):
        acum += (y[i]-y_d[i])**2
    return acum / len(y_d)
     
     

# Cargar el conjunto de datos de dígitos
digits = load_digits()

# Dividimos en entradas (X) y salidas esperadas (y_d)
x = digits.data
y_d = digits.target

##############################################
# K-FOLD 5
##############################################

mlp = MLPClassifier(hidden_layer_sizes=(32, 16),  # Capas ocultas con 32 y 16 neuronas
                activation='relu',           # Función de activación
                solver='adam',               # Optimizador
                max_iter=50,                  # Número máximo de iteraciones
                random_state=42)             # Semilla para reproducibilidad

# Crear KFold con 5 particiones
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterar sobre los 5 pliegues generados por KFold

# Crear instancias de los clasificadores
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', max_iter=500, random_state=42)
}

print("-" * 40)
for name, clf in classifiers.items():
    print(f"Clasificador: {name}")
    
    # Iterar sobre los pliegues generados por KFold
    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        # Crear conjuntos de entrenamiento y prueba para esta iteración
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_d[train_index], y_d[test_index]
        
        # Calcular la media y varianza para el conjunto de entrenamiento
        media_train = np.mean(x_train)
        varianza_train = np.var(x_train)

        # Calcular la media y varianza para el conjunto de prueba
        media_test = np.mean(x_test)
        varianza_test = np.var(x_test)
        
        clf.fit(x_train, y_train)
        
    y_pred = clf.predict(x_test)
    print("Error cuadratico medio:", cuadratic_error(y_test, y_pred))
    print("-" * 40)
    
        
