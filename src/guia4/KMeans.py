import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from itertools import combinations

class KMeans:
    def __init__(self, k=3, max_iters=100, tolerance=1e-4):
        self.k = k                  # Numero de clusters
        self.max_iters = max_iters  # Numero maximo de iteraciones
        self.tolerance = tolerance  # Tolerancia para actualizar centroides
        self.centroids = None       # Contenedor de centroides
        self.fig, self.axes = None, None  # To keep track of the same window

    
    def euclidean_distance(self,a, b):
        # Formula de distancia euclidea
        return np.sqrt(np.sum((a - b) ** 2))

    def initialize_centroids(self,X):
        # Devuelve los k puntos que van a servir de centroides
        # en las primeras iteraciones del algoritmo
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]
    
    def assign_clusters(self,X):
        dist = np.zeros(len(self.centroids)-1)
        for i in self.centroids:
            dist[i] = self.euclidean_distance(X,self.centroids[i])
        return np.argmin(dist)
    
    def fit(self, X,plot=True):
        # Paso 1: Inicializo los centroides tomando posiciones aleatorias del
        # vector de puntos
        self.centroids = self.initialize_centroids(X)
        
        # Mientras no llegue a las maximas iteraciones permitidas repito
        for i in range(self.max_iters):
            # Paso 2: Creo los clusters, se asignan los puntos a su centroide mas cercano
            clusters = self.create_clusters(X)
            
            # Paso 3: Ahora que ya se tienen los clusters, muevo la posicion vieja del
            #         centroide a la del centroide del nuevo cluster
            new_centroids = self.calculate_centroids(X, clusters)
            
            # Step 4: Checkeo que la distancia de actualizacion sea mayor a la tolerancia
            #         caso contrario consideramos que ya termino
            if self.is_converged(new_centroids):
                break
                
            self.centroids = new_centroids  # Update centroids
            
            if plot is True:
                self.plot_clusters(X, clusters, iteration=i)
        
        if plot is True:
            self.plot_clusters(X, clusters, iteration=i)
            plt.show()
    
    def predict(self, X):
        return self.assign_cluster(X)
    
    def create_clusters(self, X):
        # Inicializo el vector con K vectores vacios
        # este va a contener los puntos que esten dentro del cluster
        clusters = [[] for _ in range(self.k)]
        
        # Para cada punto dado repetir
        for idx, sample in enumerate(X):
            # Obtengo el centroide mas cercano al punto
            closest_centroid = self.closest_centroid(sample)
            # Agregar el indice del punto al cluster mas cercano
            clusters[closest_centroid].append(idx)
            
        # Retorno los clusters
        # Es un vector de k vectores
        # estos k vectores son los k conjuntos y sus elementos son 
        # los indices de los puntos contenidos por el conjunto
        return clusters
    
    def closest_centroid(self, sample):
        # Para cada punto determino el indice del centroide mas cercano 
        distances = [self.euclidean_distance(sample, centroid) for centroid in self.centroids]
        
        # Retorno el indice de dicho centroide
        return np.argmin(distances)
    
    def calculate_centroids(self, X, clusters):
        # Actulizar la posicion de los centroides
        
        # Creo un arreglo para evitar usar append
        centroids = np.zeros((self.k, X.shape[1]))
        
        # Para para cada cluster 
        for cluster_idx, cluster in enumerate(clusters):
            
            # Calcular el centroide del cluster 
            centroid = np.mean(X[cluster], axis=0)
            
            # Almacenar la nueva posicion del cluster
            centroids[cluster_idx] = centroid
            
        # Retorno los nuevos centroides
        return centroids
    
    def is_converged(self, new_centroids):
        # Checkeo si es que la distancia euclidea del nuevo conjunto de centroides
        # es menor a la tolerancia establecida para establecer una condicion de
        # convergencia
        
        distances = [self.euclidean_distance(self.centroids[i], new_centroids[i]) for i in range(self.k)]
        return sum(distances) < self.tolerance
    
    def plot_clusters(self, X, clusters, iteration=None):
        # Create figure and axes only once
        if self.fig is None or self.axes is None:
            n_features = X.shape[1]
            feature_pairs = list(combinations(range(n_features), 2))
            num_plots = len(feature_pairs)
            
            self.fig, self.axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
            if num_plots == 1:
                self.axes = [self.axes]  # Ensure it's a list for single plot case
        
        # Clear previous plot (but use the same figure)
        plt.clf()
        # Redraw for every feature pair
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        n_features = X.shape[1]
        feature_pairs = list(combinations(range(n_features), 2))
        
        for idx, (i, j) in enumerate(feature_pairs):
            ax = self.fig.add_subplot(1, len(feature_pairs), idx+1)
            
            # Plot clusters for the pair (i, j)
            for cluster_idx, cluster in enumerate(clusters):
                points = X[cluster]
                ax.scatter(points[:, i], points[:, j], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx + 1}')
            
            # Plot the centroids
            for centroid_idx, centroid in enumerate(self.centroids):
                ax.scatter(centroid[i], centroid[j], s=300, c='black', marker='x', label=f'Centroid {centroid_idx + 1}')
            
            # Voronoi diagram for the selected pair of features
            vor = Voronoi(self.centroids[:, [i, j]])
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black')
            
            ax.set_title(f"Feature {i+1} vs Feature {j+1} (Iteration #{iteration})")
            ax.set_xlabel(f'Feature {i+1}')
            ax.set_ylabel(f'Feature {j+1}')
            ax.legend()
        
        plt.suptitle(f"K-Means Clustering (Iteration #{iteration})")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
        
    # def plot_clusters(self, X, clusters, iteration=None):
    #     plt.clf()
    #     colors = ['red', 'blue', 'green', 'yellow', 'purple']
    #     
    #     n_features = X.shape[1]
    #     
    #     # Generate all pairwise combinations of features
    #     feature_pairs = list(combinations(range(n_features), 2))
    #     
    #     num_plots = len(feature_pairs)
    #     fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
# 
    #     if num_plots == 1:
    #         axes = [axes]  # Ensure it's a list for single plot case
    #     
    #     for idx, (i, j) in enumerate(feature_pairs):
    #         ax = axes[idx]
    #         
    #         # Plot clusters for the pair (i, j)
    #         for cluster_idx, cluster in enumerate(clusters):
    #             points = X[cluster]
    #             ax.scatter(points[:, i], points[:, j], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx + 1}')
    #         
    #         # Plot the centroids
    #         for centroid_idx, centroid in enumerate(self.centroids):
    #             ax.scatter(centroid[i], centroid[j], s=300, c='black', marker='x', label=f'Centroid {centroid_idx + 1}')
    #         
    #         # Voronoi diagram for the selected pair of features
    #         vor = Voronoi(self.centroids[:, [i, j]])
    #         voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black')
    #         
    #         ax.set_title(f"Feature {i+1} vs Feature {j+1} (Iteration #{iteration})")
    #         ax.set_xlabel(f'Feature {i+1}')
    #         ax.set_ylabel(f'Feature {j+1}')
    #         ax.legend()
    #     
    #     plt.suptitle(f"K-Means Clustering (Iteration #{iteration})")
    #     plt.tight_layout()
    #     plt.draw()
    #     plt.pause(0.5)