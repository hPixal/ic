from sklearn.datasets import make_blobs
import numpy as np
import KMeans
import SOM as SOM

def import_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data


def testKMeans():
    # Generate synthetic data (2D for visualization purposes)

    data = import_data('circulo.csv')
    kmeans = KMeans.KMeans(k=3)
    kmeans.fit(data,plot=True)
    
testKMeans()