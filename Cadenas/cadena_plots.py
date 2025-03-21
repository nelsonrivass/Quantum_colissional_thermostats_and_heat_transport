#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:58:04 2025

@author: nelson
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

# Perfil de temperatura y flujo

N_norm, T_WVO, T_RIT, J_WVO, J_RIT = np.loadtxt('cadena_N3_T0100_TN10.txt', skiprows=1, dtype = float, unpack = True)

plt.figure(2)
plt.plot(N_norm, T_WVO, 'rv', label = 'WVO', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(N_norm, T_RIT, 'bo', label = 'RIT', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.xlabel('índice de espín', fontsize = 15)
plt.ylabel('Temperatura', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
plt.grid()
plt.legend(fontsize=12)

plt.figure(3)
plt.plot(N_norm, J_WVO, 'rv', label = 'WVO', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(N_norm, J_RIT, 'bo', label = 'RIT', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.xlabel('índice de espín', fontsize = 15)
plt.ylabel('Flujo de calor', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
plt.grid()
plt.legend(fontsize=12)