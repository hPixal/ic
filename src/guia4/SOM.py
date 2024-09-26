import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, m, n, dim, lr=0.1, radius=None, epochs=1500):
        self.m = m                                          # Filas
        self.n = n                                          # Columnas
        self.dim = dim                                      # Dimension de la entrada
        self.lr = lr                                        # Taza de aprendizaje
        self.radius = radius if radius else max(m, n) / 2   # Radio de los vecinos
        self.epochs = epochs                                # Epocas
        self.weights = np.random.rand(m, n, dim)            # Pesos iniciales aleatorios
        self.precompute_distances() 

    def find_bmu(self, vector):
        # Para encontrar la mejor neurona (bmu) de un vector de entrada
        # vamos a recorrer por cada neurona (w) de la red y comparar 
        # la distancia en relacion a la distancia euclidea
        
        # Formula para la distancia euclidea
        # distanse(x,w_{ij}) = sqrt(sum((x - w_{ij})^2))
        
        smallest_distance = 0
        
        # BMU es Best Matching Unit
        bmu_i = 0
        bmu_j = 0
        
        for i in range(self.m):
            for j in range(self.n):
                # Recorremos los elementos de la matriz
                dist = np.linalg.norm(vector - self.weights[i, j]) # equivalente a sqrt(sum((x - w_{ij})^2))
                
                # Si la distancia es menor a la mejor distancia encontrada
                if dist < smallest_distance or smallest_distance == 0: 
                    # Actualizo quien es el seleccionado como el bmu
                    smallest_distance = dist
                    bmu_i = i
                    bmu_j = j
        
        return (bmu_i, bmu_j)

    def fast_find_bmu(self, vector):
        # Calculate the Euclidean distance between the input vector and all weights at once
        distances = np.linalg.norm(self.weights - vector, axis=2)
        # Find the index of the minimum distance (BMU)
        bmu_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        return bmu_idx
    
    def precompute_distances(self):
        # Funciones para precomputar las distancias, para que no se calcule en cada iteracion
        # Esto mejora la velocidad
        coords = np.array([[i, j] for i in range(self.m) for j in range(self.n)])
        self.dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2).reshape(self.m, self.n, self.m, self.n)
        
    def fast_update_weights(self, bmu_i, bmu_j, vector, epoch):
        # Decay the learning rate and radius over time
        lr_decay = self.lr * (1 - epoch / self.epochs)
        radius_decay = self.radius * (1 - epoch / self.epochs)

        # Update all neurons in the neighborhood of the BMU
        for i in range(self.m):
            for j in range(self.n):
                dist_to_bmu = self.dist_matrix[bmu_i, bmu_j, i, j]
                if dist_to_bmu < radius_decay:
                    influence = np.exp(-dist_to_bmu**2 / (2 * radius_decay**2))
                    self.weights[i, j] += influence * lr_decay * (vector - self.weights[i, j])

    def update_weights(self, bmu_i, bmu_j, vector,epoch):
        # Para actualizar los pesos de la red
        # tenemos determinar el radio de los vecinos
        # Formula para el radio de los vecinos
        # distanse(bmu,w_{ij}) = sqrt(sum((bmu - w_{ij})^2))
        
        lr_decay = self.lr * (1 - epoch / self.epochs)
        radius_decay = self.radius * (1 - epoch / self.epochs)
        
        bmu = np.array([bmu_i, bmu_j])
        
        for i in range(self.m):
            for j in range(self.n):
                # Recorremos las entradas del mapa
                # Calculamos la distancia euclidea
                if i != bmu_i or j != bmu_j:
                    # Calculamos la distancia de la neurona (w_{ij}) a la mejor neurona (bmu)
                    dist = np.linalg.norm(np.array([i, j]) - bmu)

                    # Si la distancia es menor al radio de los vecinos
                    if dist < self.radius:
                        # Formula para determinar la influencia del vecino
                        influence = np.exp(-dist**2 / (2 * radius_decay**2))
                        # Actualizamos los pesos de la neurona (w_{ij}) a la mejor neurona (bmu)
                        self.weights[i, j] +=  influence * lr_decay * (vector - self.weights[i, j])
                        
    def learn(self, data, visualize=False):
        
        # Configuracion de la visualizacion
        if visualize:
            plt.ion() 
        
        # Para aprender los datos vamos a tomar la mejor 
        # neurona (bmu) de cada vector de entrada y actualizar
        # su peso y el de sus vecinos
        
        # Recorremos el conjunto de datos de entrada
        
        for epoch in range(self.epochs):
            for vector in data:
                # Encuentro el BMU para el conjunto de entrada 
                bmu_i, bmu_j = self.fast_find_bmu(vector)

                # Actualizamos los pesos de la neurona (w_{ij}) a la mejor neurona (bmu)
                self.fast_update_weights(bmu_i, bmu_j, vector,epoch)
                
            # Visualizacion si es necesario
            if visualize and (epoch % 50 == 0 or epoch == self.epochs - 1):
                self.graph(epoch)
                    
        # Configuracion de la visualizacion
        if visualize:
            plt.ioff() 
            
    def predict(self, vector):
        # Para predecir el valor de un vector de entrada
        # vamos a encontrar la mejor neurona (bmu) de ese vector
        # y devolver el valor de la neurona (w_{ij})
        
        bmu_i, bmu_j = self.find_bmu(vector)
        return self.weights[bmu_i, bmu_j]
        
    def graph(self, epoch=None):
            plt.clf()  # Clear the previous plot
            plt.title(f"SOM after {epoch} epochs" if epoch is not None else "SOM")

            # Reduce weights to 2D for visualization if necessary (for multi-dimensional inputs)
            if self.dim > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced_weights = np.array([pca.fit_transform(self.weights[i]) for i in range(self.m)])
            else:
                reduced_weights = self.weights

            # Plot the neurons in the SOM grid
            for i in range(self.m):
                for j in range(self.n):
                    plt.scatter(reduced_weights[i, j, 0], reduced_weights[i, j, 1], color='blue')
                    plt.text(reduced_weights[i, j, 0], reduced_weights[i, j, 1], f'({i},{j})', fontsize=8)

            plt.pause(0.001)  # Pause to update the plot
            plt.draw()


