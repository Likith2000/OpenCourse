#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:28:46 2019

@author: hritik
"""
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#reading the Traffic.csv file created by HRITIK
dataset = pd.read_csv('Traffic.csv')
X = dataset.iloc[:,:1].values #Independent Variable
y = dataset.iloc[:,1].values #Dependent Variable


LinearReg = LinearRegression()
Polynomial = PolynomialFeatures(degree = 15)

def PolynomialML(Time):
    X_poly = Polynomial.fit_transform(X)
    LinearReg.fit(X_poly, y)
    #Predicting for all the time
    Y_Predicted = LinearReg.predict(X_poly)
    #printing the traffic predicted based on already predicted value above
    print("Traffic precentage predicted at entered time is "+ str(Y_Predicted[Time]))
    
Time=int(input("Enter time:"))
#Checking for valid input
if Time>=0 and Time<=23:
    PolynomialML(Time)
else:
    print("Wrong Time entered")
