import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score              # Precisión
import matplotlib.pyplot as plt

     
##############################################
# En esta parte se usara Split
##############################################

# Cargar el conjunto de datos de digitos
digits = load_digits()

# Dividimos en entradas (X) y salidas esperadas (y_d)
x = digits.data
y_d = digits.target

# Dividir los datos en 80% para entrenamiento y 20% para prueba
x_train, x_test, y_train, y_test = train_test_split(x, y_d, test_size=0.2, random_state=42)

# Crear y configurar el MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(32,10),  # Capas ocultas con 32 y 16 neuronas (hacen falta tantas?)
                activation='logistic',           # Función de activación (Usaria logistic capaz)
                solver='adam',               # Optimizador
                max_iter=500,                 # Número máximo de iteraciones
                random_state=42)             # Semilla para reproducibilidad

# Entrenar el modelo
mlp.fit(x_train, y_train)

# Hacer predicciones
y_pred = mlp.predict(x_test)

print("MLPClassifier:")
print("Precision: ", accuracy_score(y_test,y_pred))
print("-" * 40)


#############################################
# Ahora usaremos K-FOLD
#############################################
mlp = MLPClassifier(hidden_layer_sizes=(32, 10),  # Capas ocultas con 32 y 10 neuronas
                activation='logistic',           # Función de activación
                solver='adam',               # Optimizador
                max_iter=5000,                  # Número máximo de iteraciones
                random_state=42)             # Semilla para reproducibilidad

# Crear KFold con k particiones
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
ACCk = np.zeros(k)

# Iterar sobre los 5 pliegues generados por KFold
for fold, (train_index, test_index) in enumerate(kf.split(x)):
    # Crear conjuntos de entrenamiento y prueba para esta iteración
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y_d[train_index], y_d[test_index]
    
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

    ACCk[fold] = accuracy_score(y_test,y_pred)

# Imprimir los resultados
print("KFold 5 Particiones: ")
print(f"Media: {np.mean(ACCk)}")
print(f"Varianza: {np.var(ACCk)}")
print("-" * 40)

# Crear KFold con 2k particiones
kf = KFold(n_splits=2*k, shuffle=True, random_state=42)
ACCk = np.zeros(2*k)

# Iterar sobre los 5 pliegues generados por KFold
for fold, (train_index, test_index) in enumerate(kf.split(x)):
    # Crear conjuntos de entrenamiento y prueba para esta iteración
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y_d[train_index], y_d[test_index]
    
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

    ACCk[fold] = accuracy_score(y_test,y_pred)

# Imprimir los resultados
print("KFold 10 Particiones: ")
print(f"Media: {np.mean(ACCk)}")
print(f"Varianza: {np.var(ACCk)}")
print("-" * 40)

# Mostrar la primera imagen en el conjunto de datos
#plt.axis('off')
plt.matshow(digits.images[1])
plt.gray()
plt.show()
