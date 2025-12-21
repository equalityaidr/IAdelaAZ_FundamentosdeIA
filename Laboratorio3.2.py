#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 09:55:40 2025

@author: agtavarez
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Dataset simple
data = pd.DataFrame({
    'edad': [22, 25, 47, 52, 46, 56],
    'imc': [21.0, 23.5, 30.1, 31.2, 28.3, 33.1],
    'sobrepeso': [0, 0, 1, 1, 1, 1]
})

X = data[['edad', 'imc']]
y = data['sobrepeso']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
print(classification_report(y_test, predicciones))