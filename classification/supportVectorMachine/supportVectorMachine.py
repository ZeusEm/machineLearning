#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:15:13 2021

@author: shubham
"""

#Support Vector Machine (SVM) Model for Social Network Users dataset


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

#Splitting the dataset into Train and Test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
x_train = standardScaler.fit_transform(x_train)
x_test = standardScaler.transform(x_test)

#Fitting the classifier to the Training Set
from sklearn.svm import SVC
classifier = SVC(random_state = 0)
classifier.fit(x_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(x_test)

#Makign the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :- ")
print(confusionMatrix)
print("Total Correct predictions = ", confusionMatrix[0][0] + confusionMatrix[1][1])
print("Total Incorrect predictions = ", confusionMatrix[0][1] + confusionMatrix[1][0])

#Visualising the Training Set Results
from matplotlib.colors import ListedColormap
x_graph, y_graph = x_train, y_train
import numpy
x_range, y_range = numpy.meshgrid(numpy.arange(start = x_graph[:, 0].min() - 1, stop = x_graph[:, 0].max() + 1, step = 0.01), numpy.arange(start = x_graph[:, 1].min() - 1, stop = x_graph[:, 1].max() + 1, step = 0.01))
import matplotlib.pyplot
matplotlib.pyplot.contourf(x_range, y_range, classifier.predict(numpy.array([x_range.ravel(), y_range.ravel()]).T).reshape(x_range.shape), alpha = 0.75, cmap = ListedColormap(("red", "green")))
matplotlib.pyplot.xlim(x_range.min(), x_range.max())
matplotlib.pyplot.ylim(y_range.min(), y_range.max())
for i, j in enumerate(numpy.unique(y_graph)):
    matplotlib.pyplot.scatter(x_graph[y_graph == j, 0], x_graph[y_graph == j, 1], c = ['red', 'green'][i], label = j)
matplotlib.pyplot.title("Support Vector Machine (Training Set)")
matplotlib.pyplot.xlabel("Age")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

#Visualising the Test Set Results
from matplotlib.colors import ListedColormap
x_graph, y_graph = x_test, y_test
import numpy
x_range, y_range = numpy.meshgrid(numpy.arange(start = x_graph[:, 0].min() - 1, stop = x_graph[:, 0].max() + 1, step = 0.01), numpy.arange(start = x_graph[:, 1].min() - 1, stop = x_graph[:, 1].max() + 1, step = 0.01))
import matplotlib.pyplot
matplotlib.pyplot.contourf(x_range, y_range, classifier.predict(numpy.array([x_range.ravel(), y_range.ravel()]).T).reshape(x_range.shape), alpha = 0.75, cmap = ListedColormap(("red", "green")))
matplotlib.pyplot.xlim(x_range.min(), x_range.max())
matplotlib.pyplot.ylim(y_range.min(), y_range.max())
for i, j in enumerate(numpy.unique(y_graph)):
    matplotlib.pyplot.scatter(x_graph[y_graph == j, 0], x_graph[y_graph == j, 1], c = ['red', 'green'][i], label = j)
matplotlib.pyplot.title("Support Vector Machine (Test Set)")
matplotlib.pyplot.xlabel("Age")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()