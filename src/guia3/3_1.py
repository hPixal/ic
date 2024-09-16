import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def cuadratic_error(y_d, y):
    acum = 0
    for i in range(len(y_d)):
        acum += (y[i]-y_d[i])**2
    return acum / len(y_d)
     
##############################################
# En esta parte se usará Split
##############################################

# Cargar el conjunto de datos de dígitos
digits = load_digits()

# Dividimos en entradas (X) y salidas esperadas (y_d)
x = digits.data
y_d = digits.target

# Dividir los datos en 80% para entrenamiento y 20% para prueba
x_train, x_test, y_train, y_test = train_test_split(x, y_d, test_size=0.2, random_state=42)

# Calcular la media y varianza del conjunto de entrenamiento
media_train = np.mean(x_train)
varianza_train = np.var(x_train)

# Calcular la media y varianza del conjunto de prueba
media_test = np.mean(x_test)
varianza_test = np.var(x_test)

# Imprimir los resultados
print("Media del conjunto de entrenamiento:", media_train)
print("Varianza del conjunto de entrenamiento:", varianza_train)
print("Media del conjunto de prueba:", media_test)
print("Varianza del conjunto de prueba:", varianza_test)

# Crear y configurar el MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(32, 16),  # Capas ocultas con 32 y 16 neuronas
                activation='relu',           # Función de activación
                solver='adam',               # Optimizador
                max_iter=50,                 # Número máximo de iteraciones
                random_state=42)             # Semilla para reproducibilidad

# Entrenar el modelo
mlp.fit(x_train, y_train)

# Hacer predicciones
y_pred = mlp.predict(x_test)

print("Error cuadratico medio:", cuadratic_error(y_test, y_pred))


##############################################
# Ahora usaremos K-FOLD
##############################################

mlp = MLPClassifier(hidden_layer_sizes=(32, 16),  # Capas ocultas con 32 y 16 neuronas
                activation='relu',           # Función de activación
                solver='adam',               # Optimizador
                max_iter=50,                  # Número máximo de iteraciones
                random_state=42)             # Semilla para reproducibilidad

# Crear KFold con 5 particiones
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterar sobre los 5 pliegues generados por KFold
print("-" * 40)

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

    # Imprimir los resultados para cada pliegue
    print(f"Pliegue {fold + 1}:")
    print("  Media del conjunto de entrenamiento:", media_train)
    print("  Varianza del conjunto de entrenamiento:", varianza_train)
    print("  Media del conjunto de prueba:", media_test)
    print("  Varianza del conjunto de prueba:", varianza_test)
    print("-" * 40)
    
    mlp.fit(x_train, y_train)
    
y_pred = mlp.predict(x_test)

print("Error cuadratico medio:", cuadratic_error(y_test, y_pred))


# Mostrar la primera imagen en el conjunto de datos
plt.axis('off')
plt.gray()
plt.matshow(digits.images[1])
plt.show()
