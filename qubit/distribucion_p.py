#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:06:43 2025

@author: nelson
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

m = 0.1

def p(beta):
    p = np.sqrt(-(2*m/beta)*np.log(np.random.uniform(0,0.999)))
    return p

T = np.array([1, 100, 1000])

beta = 1/T

# Probamos la distribución para cada T

prob = []

for i in range(0,len(beta)):
    prob_i = []
    for j in range(0,100000):
        prob_i = np.append(p(beta[i]), prob_i)
    prob.append(prob_i)
    
prob = np.matrix(prob)

# # Crear figura con 3 subgráficos en una fila (1 fila, 3 columnas)

fig, axes = plt.subplots(len(T), 1, figsize=(15, 5))  # 1 fila, 3 columnas

for i in range(0, len(T)):
    
    dist = np.array(prob[i,:]).flatten()
    
    axes[i].hist(dist, bins = 1000, color = 'r', density = True)
    axes[i].set_title(r'$T =$' f'{T[i]}', fontsize = 10)
    axes[i].grid(True)
    axes[i].set_xlim(right=max(dist))

axes[1].set_ylabel('Cuentas')
axes[1].set_xlabel('Momento de la partícula')
plt.tight_layout()
plt.show()
    