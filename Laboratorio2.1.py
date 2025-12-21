#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 08:27:44 2025

@author: agtavarez
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Cargar dataset
df = sns.load_dataset("tips")

# Estadísticas básicas
print(df.describe())

# Histograma de propinas
sns.histplot(df['tip'])
plt.show()

# Diagrama de caja por día
sns.boxplot(x='day', y='total_bill', data=df)
plt.show()

# Mapa de calor de correlaciones
sns.heatmap(df.corr(), annot=True)
plt.show()