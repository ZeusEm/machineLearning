#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:42:29 2021

@author: shubham
"""

#Data Preprocessing

#Importing the libraries
import numpy    #mathemtics
import matplotlib.pyplot    #plot graphs/ charts  
import pandas    #datasets

#Importing the datasets
dataset = pandas.read_csv('/home/shubham/Music/P14-Part1-Data-Preprocessing/Section 3 - Data Preprocessing in Python/Python/Data.csv')
x = dataset.iloc[:, :-1].values    #fetches values by integer locations from first to second last column
y = dataset.iloc[:, 3].values    #fetches values from last column

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = numpy.nan, strategy = "mean")    #from sklearn.impute import SimpleImputer will work because of the following DeprecationWarning: Class Imputer is deprecated; Imputer was deprecated in version 0.20 and will be removed in 0.22. Import impute.SimpleImputer from sklearn instead.
imputer = imputer.fit(x[:, 1:3])
x[: , 1:3] = imputer.transform(x[:, 1:3])

#Encoding categorical data
#Encoding independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransformer = ColumnTransformer(transformers = [('Countries', OneHotEncoder(), [0])], remainder = 'passthrough')    #Creates individual columns for all the different countries and assigns 1 if a row corresponds to a particular country and 0 if vice versa; this is required because strings cannot be used for numerical/ statistical analysis
x = numpy.array(columnTransformer.fit_transform(x))
#Encoding dependent variables
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)    #Transforms y and replaces Yes and No (again as textual data isnt relevant for numerical/ statistical analysis) with 0/1 as labels

#Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature scaling - https://towardsdatascience.com/feature-scaling-and-normalisation-in-a-nutshell-5319af86f89b
from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
#Model Fitting - https://www.educative.io/edpresso/definition-model-fitting
x_train = standardScalerX.fit_transform(x_train)
x_test = standardScalerX.transform(x_test)