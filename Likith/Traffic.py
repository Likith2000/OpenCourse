# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1:].values

y = y.reshape((len(y), 1))


from sklearn.model_selection import  train_test_split
x_Train,x_Test,y_Train,y_Test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(x_Train,y_Train)

from sklearn.preprocessing import PolynomialFeatures
Polynomial = PolynomialFeatures(degree = 4)
x_poly = Polynomial.fit_transform(x_Train)
regressor.fit(x_poly, y_Train)
y_predicted = regressor.predict(x_poly)

x_points = np.arange(min(x_Train), max(x_Train)+0.1, 0.1)
x_points = x_points.reshape((len(x_points), 1))
traffic_predicted = regressor.predict(Polynomial.fit_transform(x_points))


timeinput=int(input("Enter The time\n"))
time=np.matrix([timeinput])
l=Polynomial.fit_transform(time)
traffic_pred=regressor.predict(l)
print("The Predicted traffic at time ",timeinput, " is",traffic_pred)


plt.scatter(x, y, color = 'blue')
plt.plot(x_points, traffic_predicted, color = 'yellow')
plt.title('Traffic Prediction')
plt.xlabel('Time')
plt.ylabel('Traffic')
plt.show()

