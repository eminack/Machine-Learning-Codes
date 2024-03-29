import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

## to generate random datasets
X, y_true = make_blobs(n_samples=300, centers=2,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);
plt.show() 
colors = ["g","r","c","b","k","o"]

class k_means:
    def __init__(self,k=2,tol=0.001,max_iter=300): 
        self.k=k
        self.tol=tol
        self.max_iter = max_iter


    def fit(self,data):

        self.centroids = {}

        ## initializing k initial centroids
        for i in range(self.k):
            self.centroids[i]=data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i]=[]

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False

            if optimized:
                break
            
    def predict(self,data):
         distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
         classification = distances.index(min(distances))
         return classification

clf = k_means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],
                marker = "o" , color="k" , s=150,linewidth=5)
    
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker= "x",color=color,s=150,linewidth=5) 
                    
plt.show()
 
