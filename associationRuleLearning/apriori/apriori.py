#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:38:19 2021

@author: shubham
"""

#Apriori Association Rule Learning on Market Basket Optimisation dataset


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
#Training Apriori on the dataset
from libraryApriori import apriori
rules = apriori(transactions, min_support = ((3*7)/7500), min_confidence = 0.2, min_lift = 3, min_length = 2)   #min_support for items purchased atleast thrice a day, everyday for a week, divided by the total number of transactions, min_confidence means that the rules need to apply to transactions atleast 20% of the time in all transactions

#Visualising the results
results = list(rules)
i = 0;
for item in results:  
    pair = item[0]   
    items = [x for x in pair]  
    print("Rule: " + items[0] + " -> " + items[1])  
  
    print("Support: " + str(item[1]))  
    print("Confidence: " + str(item[2][0][2]))  
    print("Lift: " + str(item[2][0][3]))  
    print("=====================================")
    i+=1;
    if(i == 5):    #displays only the top 5 rules
        break;