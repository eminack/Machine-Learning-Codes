import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

## to generate random datasets
X, y_true = make_blobs(n_samples=300, centers=5,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);
 
clf  = KMeans(n_clusters=4)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)

plt.scatter(centroids[:,0],centroids[:,1],c='black',marker='x',s=250,linewidth=5)
plt.show()
