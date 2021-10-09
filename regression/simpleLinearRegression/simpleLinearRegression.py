#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 00:42:09 2021

@author: shubham
"""

#Simple Linear Regression excercise on predicting employee salaries based on their job experience


#Data Preprocessing

#Importing the libraries
import pandas    #datasets

#Importing the datasets
dataset = pandas.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test Set results
y_pred = regressor.predict(x_test)

#Visualising the Training Set results
import matplotlib.pyplot
matplotlib.pyplot.scatter(x_train, y_train, color = "red")
matplotlib.pyplot.plot(x_train, regressor.predict(x_train), color = "blue")
matplotlib.pyplot.title("Salary vs Experience (Training Dataset)")
matplotlib.pyplot.xlabel("Years of Experience")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.show()

#Visualising the Test Set results
import matplotlib.pyplot
matplotlib.pyplot.scatter(x_test, y_test, color = "red")
matplotlib.pyplot.plot(x_train, regressor.predict(x_train), color = "blue")
matplotlib.pyplot.title("Salary vs Experience (Testing Dataset)")
matplotlib.pyplot.xlabel("Years of Experience")
matplotlib.pyplot.ylabel("Salary")
matplotlib.pyplot.show()