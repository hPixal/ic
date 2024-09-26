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
mlp = MLPClassifier(hidden_layer_sizes=(32, 10),  # Capas ocultas con 32 y 16 neuronas
                activation='logistic',           # Función de activación
                solver='adam',               # Optimizador
                max_iter=5000,                  # Número máximo de iteraciones
                random_state=42)             # Semilla para reproducibilidad

# Crear KFold con k particiones
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
ACCk = np.zeros(k)

# Crear instancias de los clasificadores
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(32, 10), activation='logistic', solver='adam', max_iter=5000, random_state=42)
}

print("-" * 40)
for name, clf in classifiers.items():
    print(f"Clasificador: {name}")
    
    # Iterar sobre los pliegues generados por KFold
    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        # Crear conjuntos de entrenamiento y prueba para esta iteración
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_d[train_index], y_d[test_index]
        
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        ACCk[fold] = accuracy_score(y_test,y_pred)

        # Imprimir los resultados para cada pliegue
        #print(f"Pliegue {fold + 1}:")
        #print(f"Precision: {ACCk[fold]}")
        #print("Error cuadratico medio:", cuadratic_error(y_test, y_pred))
        #print("-" * 40)
    print(f"Media: {np.mean(ACCk)}")
    print(f"Varianza: {np.var(ACCk)}")
    print("-" * 40)
