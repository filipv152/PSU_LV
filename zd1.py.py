from sklearn import datasets
import numpy as np

def generate_data(n_samples, flagc):
    
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generiranje podataka
X = generate_data(n_samples=500, flagc=5)

# K-means grupiranje
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Bojanje točaka ovisno o grupi
colors = ['red', 'blue', 'green']
labels = kmeans.predict(X)
colored = [colors[label] for label in labels]

# Prikazivanje rezultata
plt.scatter(X[:,0], X[:,1], color=colored)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', s=200)
plt.show()
