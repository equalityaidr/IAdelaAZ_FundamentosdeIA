#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 09:54:21 2025

@author: agtavarez
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Datos simulados
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

modelo = LinearRegression()
modelo.fit(X, y)

predicciones = modelo.predict(X)

plt.scatter(X, y, color='blue')
plt.plot(X, predicciones, color='red')
plt.title("Regresión Lineal")
plt.xlabel("X")
plt.ylabel("y")
plt.show()