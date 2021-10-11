#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:28:16 2021

@author: shubham
"""

#Solving the earlier polynomial regression problem with Support Vector Regression (SVR)


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Feature scaling
from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
standardScalerY = StandardScaler()
x = standardScalerX.fit_transform(x)
y = standardScalerY.fit_transform(y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x, y)

#Predicting salary for a given position 6.5
import numpy
y_pred = standardScalerY.inverse_transform(regressor.predict(standardScalerX.transform(numpy.array(6.5).reshape(1,-1))))

#Visualising the SVR results
#import matplotlib.pyplot
#matplotlib.pyplot.scatter(x, y, color = "red")
#matplotlib.pyplot.plot(x, regressor.predict(x), color = "blue")
#matplotlib.pyplot.title("Support Vector Regression Model on Salaries Problem")
#matplotlib.pyplot.xlabel("Position Level")
#matplotlib.pyplot.ylabel("Salary")
#matplotlib.pyplot.show()

#Visualising the SVR results (finer curve)
x_grid = numpy.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
import matplotlib.pyplot
matplotlib.pyplot.scatter(x, y, color="red")
matplotlib.pyplot.plot(x_grid, regressor.predict(standardScalerX.fit_transform(x_grid)), color="blue")
matplotlib.pyplot.title("Support Vector Regression Model on Salaries Problem")
matplotlib.pyplot.xlabel("Position Level")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.show()