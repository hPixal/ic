from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


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
    return X,Y

X,_ = import_iris()

# Valores de k a probar
k_values = range(2, 11)

# Listas para almacenar las métricas
inertia_values = []
silhouette_scores = []

# Probar diferentes valores de k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    # Calcular la inercia (distancia dentro de los clusters)
    inertia = kmeans.inertia_
    inertia_values.append(inertia)
    
    # Calcular el índice de silueta (solo si k > 1)
    if k > 1:
        silhouette = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette)
    else:
        silhouette_scores.append(None)

# Graficar las métricas
plt.figure(figsize=(12, 5))

# Inercia
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Inercia para diferentes valores de k')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')

# Índice de silueta
plt.subplot(1, 2, 2)
plt.plot(k_values[1:], silhouette_scores[1:], marker='o')
plt.title('Índice de silueta para diferentes valores de k')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Índice de silueta')

plt.tight_layout()
plt.show()
