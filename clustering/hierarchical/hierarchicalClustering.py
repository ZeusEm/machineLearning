#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 12:31:20 2021

@author: shubham
"""

#Hierarchical Clustering on Mall Customers dataset


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, 3:5].values

#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy
dendrogram = scipy.cluster.hierarchy.dendrogram(scipy.cluster.hierarchy.linkage(x, method = "ward"))
import matplotlib.pyplot
matplotlib.pyplot.title("Dendrogram : Mall Customers Problem")
matplotlib.pyplot.xlabel("Customers")
matplotlib.pyplot.ylabel("Euclidean Distances")
matplotlib.pyplot.show()

#Fitting Hierarchical Clustering to the Mall dataset
from sklearn.cluster import AgglomerativeClustering
hierarchicalClustering = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hierarchicalClustering = hierarchicalClustering.fit_predict(x)

#Visualising the clusters in 2D
matplotlib.pyplot.scatter(x[y_hierarchicalClustering == 0, 0], x[y_hierarchicalClustering == 0, 1], s = 60, c = 'red', label = 'Cluster 1 - Careful Customers')
matplotlib.pyplot.scatter(x[y_hierarchicalClustering == 1, 0], x[y_hierarchicalClustering == 1, 1], s = 60, c = 'blue', label = 'Cluster 2 - Standard Customers')
matplotlib.pyplot.scatter(x[y_hierarchicalClustering == 2, 0], x[y_hierarchicalClustering == 2, 1], s = 60, c = 'green', label = 'Cluster 3 - Target Customers')
matplotlib.pyplot.scatter(x[y_hierarchicalClustering == 3, 0], x[y_hierarchicalClustering == 3, 1], s = 60, c = 'cyan', label = 'Cluster 4 - Careless Customers')
matplotlib.pyplot.scatter(x[y_hierarchicalClustering == 4, 0], x[y_hierarchicalClustering == 4, 1], s = 60, c = 'magenta', label = 'Cluster 5 - Sensible Customers') 
matplotlib.pyplot.title("Clusters of Clients")
matplotlib.pyplot.xlabel('Annual Income (k$)')
matplotlib.pyplot.ylabel('Spending Score (1-100)')
matplotlib.pyplot.legend() 
matplotlib.pyplot.show()