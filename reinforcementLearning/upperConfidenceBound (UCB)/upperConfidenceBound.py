#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:20:01 2021

@author: shubham
"""

#Upper Confidence Bound Reinforcement Learning on a Social Network Click-Through-Rate Optimisation dataset

#https://www.aionlinecourse.com/tutorial/machine-learning/upper-confidence-bound-%28ucb%29


#Data Preprocessing

#Importing the libraries
import pandas

#Importing the dataset
dataset = pandas.read_csv("Ads_CTR_Optimisation.csv")
#This dataset comprises the response of 10,000 visitors to 10 advertisements displayed on a web platform. These 10 advertisements are actually the 10 ad versions of the same product. The responses are represented in terms of rewards given to those 10 ads by visitors. If the visitor has clicked on an ad, the reward is 1 and if the visitor has ignored the ad, the reward is 0. Now, based on these rewards, the task is to identify which among the 10 ads has the highest CTR so that the ad with the highest conversion rate should be placed on the web platform.


#In this reinforcement learning in python implementation, we will compare two approaches – Random selection of ads and selection using UCB method so that we would be able to conclude the effectiveness of UCB method.

#Random Selection
rows = 10000   #No. of rows (users)
columns = 10  #No. of columns (advertisement variants)
totalReward = 0
adsSelected = []
import random
for rows in range(0, rows):
    ad = random.randrange(columns)
    adsSelected.append(ad)
    reward = dataset.values[rows, ad] # if n th row is 1 then prize is 1
    totalReward += reward
print("Total Prize:", totalReward)

#Random Selection Visualisation
import matplotlib.pyplot
matplotlib.pyplot.hist(adsSelected)
matplotlib.pyplot.title("Random Selection Histogram of ad selections")
matplotlib.pyplot.xlabel('Ads')
matplotlib.pyplot.ylabel('Number of times each ad was selected')
matplotlib.pyplot.show()


#Given below are the intuitive steps behind UCB for maximizing the rewards:

#Step 1: Each machine is assumed to have a uniform Confidence Interval and a success distribution. This Confidence Interval is a margin of success rate distributions which is the most certain to consist of the actual success rate distribution of each machine which we are unaware of in the beginning.

#Step 2: A machine is randomly chosen to play, as initially, they have all the same confidence Intervals.

#Step 3: Based on whether the machine gave a reward or not, the Confidence Interval shifts either towards or away from the actual success distribution and the also converges or shrinks as it has been explored thus resulting in the Upper bound value of the confidence Interval to also be reduced.

#Step 4: Based on the current Upper Confidence bounds of each of the machines, the one with the highest is chosen to explore in the next round.

#Step 5: Steps 3 and 4 are continued until there are sufficient observations to determine the upper confidence bound of each machine. The one with the highest upper confidence bound is the machine with the highest success rate.

#Implementing UCB
rows = 10000   #No. of rows (users)
columns = 10  #No. of columns (advertisement variants)
adsSelected = []
#Variable Initialisation
numbersOfSelections = [0] * columns
sumsOfRewards = [0] * columns
totalReward = 0
import math

#Algorithm

#We will iterate over each ad in each row starting with index 0 and with a maximum upper bound value of zero.

#At each round, we will check if a ad has been selected before or not. If yes, the algorithm proceeds to calculate the average rewards of the ad, the delta and the upper confidence. If not, that is if the ad is being selected for the first time then it sets a default upper bound value of 1e400.

#After each round, the ad with the highest upper bound value is selected, the number of selections along with the actual reward and sum of rewards for the selected ad is updated.

#After all the rounds are completed, we will have a ad with a maximum upper bound value.

for rows in range(0, rows):
    ad = 0
    maxUpperBound = 0
    for i in range(0, columns):
        if (numbersOfSelections[i] > 0):
            averageReward = sumsOfRewards[i] / numbersOfSelections[i]
            deltaI = math.sqrt(3/2 * math.log(rows) / numbersOfSelections[i])
            upperBound = averageReward + deltaI
        else:
            upperBound = 1e400
            #Here we applied a trick in the else condition by taking the variable upperBound to a huge number. This is because we want the first 10 rounds as trial rounds where the 10 ads are selected at least once. This trick will help us to do so.
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i
    adsSelected.append(ad)
    numbersOfSelections[ad] += 1
    reward = dataset.values[rows, ad]
    sumsOfRewards[ad] += reward
    totalReward += reward
print("Total Prize:" , totalReward)

# Visualising UCB results
matplotlib.pyplot.hist(adsSelected)
matplotlib.pyplot.title('Histogram of ad selections')
matplotlib.pyplot.xlabel('Ads')
matplotlib.pyplot.ylabel('Number of times each ad was selected')
matplotlib.pyplot.show()