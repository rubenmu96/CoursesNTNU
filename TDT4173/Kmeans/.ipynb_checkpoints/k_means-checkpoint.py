import numpy as np 
import pandas as pd 
import sys
import random
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class KMeans:
    def __init__(self, k, iters, kmeanspp = False):
        self.k = k # number of clusters
        self.iters = iters # number of iterations
        self.kmeanspp = kmeanspp # determine which type of initialization we are using
        

        self.points = {}
        self.centroids = {}
        for i in range(self.k): # make the dictionary the same size as the number of k
            self.centroids[i] = []
            self.points[i] = []

    def fit(self, X):
        # Turn X into an array
        X = np.array(X)
        
        # Determine if we want the clusters to be initialized from random points drawn from X or use Kmeans++
        r = list(range(X.shape[0])); random.shuffle(r)
        centroids_init = Kmpp(X, self.k)
        
        for i,rand_i in enumerate(r[0:self.k]):
            if self.kmeanspp == True:
                self.centroids[i] = centroids_init[i]
            else: 
                # This is not very good way to initalize clusters if we have many centroids, as the initial clusters may not be very spread
                # Using this on dataset 2 gives Silhouette Score between ~0.412-0.573
                self.centroids[i] = X[rand_i]
        
        for i in range(self.iters):
            for x in X:
                # Compute the distance between the points in X and the centroids
                distances = [euclidean_distance(x, self.centroids[j]) for j in self.centroids]
                
                # Find which class the points belongs to by looking at the minimum distance between the point and centroid
                self.points[np.argmin(distances)].append(x)
            
            for k in range(self.k):
                self.centroids[k] = np.average(self.points[k], axis = 0)

    def predict(self, X):
        X = np.array(X) # Turn X into an array
        points = []
        for x in X:
            # Compute the distance between the points in X and the centroids
            distances = [euclidean_distance(x,self.centroids[j]) for j in self.centroids]
            # Find which class the points belongs to by looking at the minimum distance between the point and centroid
            points.append(np.argmin(distances))
        return points
    
    # Remove this function?
    def get_centroids(self):
        return self.centroids
    
    
    
    
# --- Some utility functions 
def euclidean_distance(x, y):
    return np.linalg.norm(x - y, ord=2, axis=-1)

def Kmpp(X,k): # Implementation of Kmeans++
    '''
    This code was made with help from:
    https://www.geeksforgeeks.org/ml-k-means-algorithm/
    '''
    centroids = []; dist = {}
    for i in range(k-1):
        dist[i] = []

    # Choose the first centroid by choosing a random data point
    r = list(range(X.shape[0])); random.shuffle(r)
    centroids.append(X[r[0]])
    
    # find the rest k-1 centroids
    for i in range(k-1):
        # iterate over all values x in X
        for x in X:
            d = np.inf
            
            # iterate through the centroids to find the minimum distance
            for j in range(len(centroids)):
                d = min(d, euclidean_distance(x, centroids[j]))
            # Append the smallest value
            dist[i].append(d)

        # Find the new centroid
        centroid = X[np.argmax(dist[i]),:]
        centroids.append(centroid)
    return centroids



def cross_euclidean_distance(x, y=None):
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum() # removed axis=1, got shape mismatch for Xc
        
    return distortion


def euclidean_silhouette(X, z):
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  
