#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 08:18:36 2025

@author: agtavarez
"""

import pandas as pd

# Dataset artificial
df = pd.DataFrame({
    'Nombre': ['Ana', 'Luis', 'Pedro', 'Luis', None],
    'Edad': [25, 30, None, 30, 22],
    'Ingreso': ['5000', '6000', '7000', '6000', '8000']
})

# Eliminar duplicados
df = df.drop_duplicates()

# Rellenar valores nulos
df['Edad'] = df['Edad'].fillna(df['Edad'].mean())

# Cambiar tipo de ingreso a numérico
df['Ingreso'] = df['Ingreso'].astype(int)

print(df)