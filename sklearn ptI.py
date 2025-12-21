# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:37:51 2020

@author: DELL
"""
import sklearn.datasets as data
import numpy as np
import random

np.random.seed(100)
random.seed(100)

    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()

X = housing['data']
Y= housing['target']
print(housing['DESCR'])

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(10,8))

t = np.arange(0.0, Y.shape[0], 1)

fig, ax = plt.subplots()
ax.plot(t, Y)

ax.set(xlabel='MedInc', ylabel=housing['target'],
       title='Housing Prices')
ax.grid()
plt.show()

#scatter

plt.scatter(housing.Medinc, housing.target)

#scatterplot en seaborn

sns.set()
plt.scatter(iris.sepal_length, iris.sepal_width)

#agregamos set_style. Probar con  darkgrid, dark, white, and ticks

sns.set_style('dark')
plt.scatter(iris.sepal_length, iris.sepal_width)
plt.show()

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, Y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
prediction = regressor.predict(X_test)
prediction_2 = regressor.predict(X_train)
mean_squared_error = mean_squared_error(Y_test, prediction)
mean_absolute_error = mean_absolute_error(Y_test, prediction)
r2_sc = r2_score(Y_test, prediction)
r2_sc_2 = r2_score(Y_train, prediction_2)
 
print("R2:{:.3f},R2_2:{:.3f}, Mean Squared Error:{:.3f}, Mean Absolute Error:{:.3f}". \
      format(r2_sc,r2_sc_2, mean_squared_error, mean_absolute_error))