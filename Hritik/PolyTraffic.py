#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:28:46 2019

@author: hritik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#reading the Traffic.csv file created by HRITIK
dataset = pd.read_csv('Traffic.csv')
X = dataset.iloc[:,:1].values #Independent Variable
y = dataset.iloc[:,1].values #Dependent Variable

#changing y into numpy array using reshape function
y = y.reshape((len(y), 1))

LinearReg = LinearRegression()
Polynomial = PolynomialFeatures(degree=15)

X_poly = Polynomial.fit_transform(X)
LinearReg.fit(X_poly, y)
#Predicting for all the time values
Y_Predicted = LinearReg.predict(X_poly)

X_points = np.arange(min(X), max(X)+0.1, 0.1)
# converting the X_points into numpy array
X_points = X_points.reshape((len(X_points), 1))
# using the newly got grid points and predicting the corresponding Y
temp_predicted = LinearReg.predict(Polynomial.fit_transform(X_points))
plt.scatter(X, y, color = 'red')
plt.plot(X_points, temp_predicted, color = 'blue')
plt.title('Traffic Prediction')
plt.xlabel('Time')
plt.ylabel('Traffic Percentage')
plt.show()

def PolynomialML(Time):    
    #printing the traffic predicted based on already predicted value above
    print("Traffic precentage predicted at entered time is {}".format(list(Y_Predicted[Time])[0]))
    
Time=int(input("Enter time:"))
#Checking for valid input
if Time>=0 and Time<=23:
    PolynomialML(Time)
else:
    print("Wrong Time entered")
