#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 23:10:53 2021

@author: shubham
"""

#Using Random Forest Regression Model for Salaries problem


#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Fitting the Random Forest Regressor to our Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x, y)

#Predicting a Salary value against a given Salary Position
import numpy
regressor.predict(numpy.array(6.5).reshape(1,-1))

#Visualising the Regression results (for higher resolution and a smoother curve)
x_grid = numpy.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid)), 1)
import matplotlib.pyplot
matplotlib.pyplot.scatter(x, y, color = "red")
matplotlib.pyplot.plot(x_grid, regressor.predict(x_grid), color = "blue")
matplotlib.pyplot.title("Random Forest Regression Model on Salaries problem")
matplotlib.pyplot.xlabel("Position Level")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.show()