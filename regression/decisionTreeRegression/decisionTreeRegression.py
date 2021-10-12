#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:22:23 2021

@author: shubham
"""

#Decision Tree Regression Model on the Salaries problem


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

#Predicting a salary for a given Position Level
import numpy
y_pred = regressor.predict(numpy.array(6.5).reshape(1,-1))

#Visualising the Decision Tree Regression Result
x_grid = numpy.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
import matplotlib.pyplot
matplotlib.pyplot.scatter(x, y, color = "red")
matplotlib.pyplot.plot(x_grid, regressor.predict(x_grid), color = "blue")
matplotlib.pyplot.title("Decision Tree Regression Model on Salaries Problem")
matplotlib.pyplot.xlabel("Position Level")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.show()