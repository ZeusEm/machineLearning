#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 16:50:35 2021

@author: shubham
"""

#Predict the salary of a new employee at a given "Level" with a given matrix of position levels and their corresponding salaries


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values    #define x as a range of columns to have it as a matrix instead of a single column array
y = dataset.iloc[:, 2].values

#Very small dataset, hence no splitting into test/ train set required.
#No need for feature scaling as Linear Regression libraries already take care of that

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 4)    #Define a polynomial regressor of 4th degree
x_poly = polynomial_regressor.fit_transform(x)    #Define a polynomial matrix of features for the regressor to fit
polynomial_regressor.fit(x_poly, y)    #Fit the polynomial matrix onto the polynomial regressor

#Fitting polynomial matrix of features into a linear regression model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_poly, y)

#Visualising the Polynomial Regression results
import numpy
x_grid = numpy.arange(min(x), max(x), 0.1)    #defines a range to x_grid, from 1 to 10 with a step interval of 0.1, resulting in 90 values
x_grid = x_grid.reshape((len(x_grid), 1))    #converts/ reshapes the above array into a matrix of (90, 1)
import matplotlib.pyplot
matplotlib.pyplot.scatter(x, y, color="red")
#matplotlib.pyplot.plot(x, linear_regressor.predict(x_poly), color="blue")
matplotlib.pyplot.plot(x_grid, linear_regressor.predict(polynomial_regressor.fit_transform(x_grid)), color="blue")    #plots x_grid with 90 points along x-axis against predicted values of x_grid after conversion to an equivalent polynomial matrix
matplotlib.pyplot.title("Predictive Employee Salary Polynomial Regression Model")
matplotlib.pyplot.xlabel("Position Level")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.show()

#Predicting a Salary from the given Position 6.5
linear_regressor.predict(polynomial_regressor.fit_transform(numpy.array(6.5).reshape(1,-1)))