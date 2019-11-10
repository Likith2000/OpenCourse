# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:57:53 2019

@author: nitya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Traffic.csv")

x = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values
y = y.reshape((len(y), 1))

from sklearn.model_selection import train_test_split
x_Train,x_Test,y_Train,y_Test=train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_Train,y_Train)

from sklearn.preprocessing import PolynomialFeatures
Poly = PolynomialFeatures(degree = 1)
x_poly = Poly.fit_transform(x_Train)
regressor.fit(x_poly, y_Train)
reg = LinearRegression()

hr = int(input("Enter the hour of the day : "))
hr = np.matrix([hr])
hour = Poly.fit_transform(hr)
traff = regressor.predict(hour)
print("Traffic Density in % is : ",traff)

y_pred = regressor.predict(x_poly) 
x_points = np.arange(min(x), max(x)+0.1, 0.1) 
x_points = x_points.reshape((len(x_points),-1))
temp_pred = regressor.predict(Poly.fit_transform(x_points))

plt.scatter(x, y, color = 'red')
plt.plot(x_points, temp_pred, color = 'blue')
plt.title('Traffic Predictor')
plt.xlabel('Time')
plt.ylabel('Traffic Density')
plt.show() 