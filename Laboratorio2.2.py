#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 08:38:37 2025

@author: agtavarez
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Dataset simulado
data = pd.DataFrame({
    'edad': [25, 45, 32, 50, 23, 35],
    'imc': [22.4, 27.5, 30.1, 31.2, 21.3, 29.9],
    'sobrepeso': [0, 1, 1, 1, 0, 1]
})

X = data[['edad', 'imc']]
y = data['sobrepeso']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
print("Exactitud:", accuracy_score(y_test, predicciones))