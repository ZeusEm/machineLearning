#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 16:15:33 2021

@author: shubham
"""

#Multiple Linear Regression excercise on predicting the most profitable company for a venture captalist fund to invest in, based on its R&D Spend, Administration, Marketing Spend, State and Net Profit. Check if there is any correlation between the profit incurred by a company vis-a-vis to the corresponding expenses on R&D, Administration, Marketing and the State of operation 

#Before applying Linear Regression towards modelling data, first check whether the data confirms to the assumptions/ requirements of linear regression itself - https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/

#Ensure that using dummy variables does not lead to multicollinearity - https://www.geeksforgeeks.org/ml-dummy-variable-trap-in-regression-models/

#p-value - https://towardsdatascience.com/null-hypothesis-and-the-p-value-fdc129db6502

#backward alimination - https://towardsdatascience.com/backward-elimination-for-feature-selection-in-machine-learning-c6a3a8f8cef4

#forward selection - https://towardsdatascience.com/using-forward-selection-to-filter-out-unnecessary-features-in-a-machine-learning-dataset-e36c62431781

#numpy.ones - https://numpy.org/doc/stable/reference/generated/numpy.ones.html

#Dependent variable - Profit
#Independent variable - R&D Spend, Admin, Marketing, State


#Data Preprocessing

#Importing the libraries
import pandas
import numpy

#Importing the dataset
dataset = pandas.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
#Encoding independent variable "State"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransformer = ColumnTransformer(transformers = [('oneHotEncoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = numpy.array(columnTransformer.fit_transform(x), dtype=numpy.float)

#Avoiding the Dummy Variable Trap
x = x[:, 1:]

#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test Set results
y_pred = regressor.predict(x_test)

#Building the optimal model using Backward Elimination
#The thumbrule is to keep eliminating independent variables till you see the corresponding p>0.05, which is your chosen significance level. However, do so only until you see that the adjusted R2 value continues to increase and stop eliminating the moment you see that eliminating variables in turn lead up to a decreasing adjusted R2 value
#Interpreting coefficients - Look at the Estimate column of the Coefficients section - A positive number against a given variable indicates a positive correlation and vice versa, also, the magntiude indicates a "per unit" significance the resective independent variable has on the performance of the dependent variable
import statsmodels.api
x = numpy.append(numpy.ones(shape = (50, 1), dtype = "int"), x, axis = 1)    #Adding a column of 1s at the beginning of the independent variable matrix to correspond to b0x0 of the multilinear regression equation
x_optimalMatrixOfFeatures = x[:, [0, 1, 2, 3, 4, 5]] #initialising the optimal feature matrix with the constant + independent variable columns
regressor_OLS = statsmodels.api.OLS(endog = y, exog = x_optimalMatrixOfFeatures).fit()
regressor_OLS.summary()

#Looping again, eliminating x2 with p=0.990>0.05 and fitting x_optimalMatrixOfFeatures
x_optimalMatrixOfFeatures = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = statsmodels.api.OLS(endog = y, exog = x_optimalMatrixOfFeatures).fit()
regressor_OLS.summary()

#Looping again, eliminating x1 with p=0.940>0.05 and fitting x_optimalMatrixOfFeatures
x_optimalMatrixOfFeatures = x[:, [0, 3, 4, 5]]
regressor_OLS = statsmodels.api.OLS(endog = y, exog = x_optimalMatrixOfFeatures).fit()
regressor_OLS.summary()

#Looping again, eliminating x2 with p=0.602>0.05 and fitting x_optimalMatrixOfFeatures
x_optimalMatrixOfFeatures = x[:, [0, 3, 5]]
regressor_OLS = statsmodels.api.OLS(endog = y, exog = x_optimalMatrixOfFeatures).fit()
regressor_OLS.summary()

#Looping again, eliminating x2 with p=0.060>0.05 and fitting x_optimalMatrixOfFeatures
x_optimalMatrixOfFeatures = x[:, [0, 3]]
regressor_OLS = statsmodels.api.OLS(endog = y, exog = x_optimalMatrixOfFeatures).fit()
regressor_OLS.summary()
