#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:37:17 2019

@author: root
"""
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
data=pd.read_csv("traffic.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6,random_state=0)

from sklearn.linear_model import LinearRegression
LR=LinearRegression()


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=5)

x_poly=poly.fit_transform(x_test)
LR.fit(x_poly,y_test)

z=int(input("Enter The time\n"))
time=np.matrix([z])

l=poly.fit_transform(time)
traffic_pred=LR.predict(l)

print("The Predicted traffic at time ",z, " is",traffic_pred)

