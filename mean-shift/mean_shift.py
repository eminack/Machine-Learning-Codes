import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift
from mpl_toolkits.mplot3d import Axes3D 

## to generate random datasets
centers = [[1,1,1],[5,5,5],[3,10,10]]
X, y_true = make_blobs(n_samples=10000,centers=centers,
                       cluster_std=1)
clf =  MeanShift()
clf.fit(X)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
print(cluster_centers)
colors = ["g","r","c","b","k","o"]

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')

for i in range(len(X)):
    ax.scatter(X[i][0],X[i][1],X[i][2],c=colors[labels[i]],marker = 'o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],marker = "X",color="k",s=150,linewidths=5,zorder=10)
plt.show()
