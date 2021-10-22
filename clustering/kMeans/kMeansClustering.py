# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:04:55 2021
@author: shubham
"""

#K-Means Clustering on Mall Customers dataset


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, 3:5].values

#Using the Elbow Method to find the optimal number of clusters
from sklearn.cluster import KMeans
withinClusterSumOfSquares = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 0)
    kmeans.fit(x)
    withinClusterSumOfSquares.append(kmeans.inertia_)

#Plotting the Elbow Method graph
import matplotlib.pyplot
matplotlib.pyplot.plot(range(1, 11), withinClusterSumOfSquares)
matplotlib.pyplot.title("Elbow Method to determine the Optimal No. of Clusters")
matplotlib.pyplot.xlabel("No. of Clusters")
matplotlib.pyplot.ylabel("Within Cluster Sum of Squares (WCSS)")
matplotlib.pyplot.show()

#Applying K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters in 2D
matplotlib.pyplot.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster 1 - Careful Customers')
matplotlib.pyplot.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster 2 - Standard Customers')
matplotlib.pyplot.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster 3 - Target Customers')
matplotlib.pyplot.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 60, c = 'cyan', label = 'Cluster 4 - Careless Customers')
matplotlib.pyplot.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 60, c = 'magenta', label = 'Cluster 5 - Sensible Customers') 
matplotlib.pyplot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
matplotlib.pyplot.title("Clusters of Clients")
matplotlib.pyplot.xlabel('Annual Income (k$)')
matplotlib.pyplot.ylabel('Spending Score (1-100)')
matplotlib.pyplot.legend() 
matplotlib.pyplot.show()