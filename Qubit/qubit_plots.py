#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:19:56 2025

@author: nelson
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

# Probabilidad de transición

Ep, WVO, RIT, Exact = np.loadtxt('prob_qubit.txt', skiprows=1, dtype = float, unpack = True)

plt.figure(1)
plt.plot(Ep, WVO, 'r-', label = 'WVO', linewidth = 1.5)
plt.plot(Ep, RIT, 'b-', label = 'RIT', linewidth = 1.5)
plt.plot(Ep, Exact, 'g--', label = 'Exacto', linewidth = 1)
plt.xlabel(r'$E = \frac{p^2}{2m}$', fontsize = 15)
plt.ylabel(r'$P_{00 -> 11}$', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=12)

# Probabilidad de transición para bajas energías

Ep_l, WVO_l, RIT_l, Exact_l = np.loadtxt('prob_qubit_lowE.txt', skiprows=1, dtype = float, unpack = True)

plt.figure(2)
plt.plot(Ep_l, WVO_l, 'r-', label = 'WVO', linewidth = 1.5)
plt.plot(Ep_l, RIT_l, 'b-', label = 'RIT', linewidth = 1.5)
plt.plot(Ep_l, Exact_l, 'g--', label = 'Exacto', linewidth = 1)
plt.xlabel(r'$E = \frac{p^2}{2m}$', fontsize = 15)
plt.ylabel(r'$P_{00 -> 11}$', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=12)

# Termalización con operadores de Kraus

T, p00_WVO_jym1, p00_RIT_jym1 = np.loadtxt('termalizacion_kraus_jy-1_e4.txt', skiprows=1, dtype = float, unpack = True)
T, p00_WVO_jy0, p00_RIT_jy0 = np.loadtxt('termalizacion_kraus_jy0_e4.txt', skiprows=1, dtype = float, unpack = True)
T, p00_WVO_jy1, p00_RIT_jy1 = np.loadtxt('termalizacion_kraus_jy1_e4.txt', skiprows=1, dtype = float, unpack = True)
T_can, p00_can = np.loadtxt('termalizacion_can.txt', skiprows=1, dtype = float, unpack = True)

plt.figure(3)
plt.plot(T, p00_WVO_jy0, 'rv', label = 'WVO', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jy0, 'bo', label = 'RIT', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_WVO_jym1, 'rv', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jym1, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_WVO_jy1, 'rv', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jy1, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T_can, p00_can, 'k-', label = 'Población canónica')
plt.xscale('log')
plt.xlabel('Temperatura', fontsize = 15)
plt.ylabel(r'$\rho_{00}$', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
#plt.grid()
plt.legend(fontsize=12)

# Sin operadores de Kraus, Además añadimos el método exacto para el caso jy = 0

T, p00_WVO_jym1, p00_RIT_jym1 = np.loadtxt('termalizacion_nokraus_jy-1_1e4.txt', skiprows=1, dtype = float, unpack = True)
T, p00_WVO_jy0, p00_RIT_jy0, p00_exact_jy0 = np.loadtxt('termalizacion_nokraus_full_jy0_1e4.txt', skiprows=1, dtype = float, unpack = True)
T, p00_WVO_jy1, p00_RIT_jy1 = np.loadtxt('termalizacion_nokraus_jy1_1e4.txt', skiprows=1, dtype = float, unpack = True)
T_can, p00_can = np.loadtxt('termalizacion_can.txt', skiprows=1, dtype = float, unpack = True)

plt.figure(4)
plt.plot(T, p00_WVO_jy0, 'rv', label = 'WVO', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jy0, 'bo', label = 'RIT', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_WVO_jym1, 'rv', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jym1, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_WVO_jy1, 'rv', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jy1, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_exact_jy0, 'sg', label = 'Exacto', markersize = 7,  fillstyle='none', markeredgewidth=2)
plt.plot(T_can, p00_can, 'k-', label = 'Población canónica')
plt.xscale('log')
plt.xlabel('Temperatura', fontsize = 15)
plt.ylabel(r'$\rho_{00}$', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
#plt.grid()
plt.legend(fontsize=12)


# # Sin operadores de Kraus y utilizando los ritmos promediados

T, p00_WVO_jym1, p00_RIT_jym1 = np.loadtxt('termalizacion_nokraus_prom_jy-1_1e4.txt', skiprows=1, dtype = float, unpack = True)
T, p00_WVO_jy0, p00_RIT_jy0,= np.loadtxt('termalizacion_nokraus_prom_jy0_1e4.txt', skiprows=1, dtype = float, unpack = True)
T, p00_WVO_jy1, p00_RIT_jy1 = np.loadtxt('termalizacion_nokraus_prom_jy1_1e4.txt', skiprows=1, dtype = float, unpack = True)
T_can, p00_can = np.loadtxt('termalizacion_can.txt', skiprows=1, dtype = float, unpack = True)

plt.figure(5)
plt.plot(T, p00_WVO_jy0, 'rv', label = 'WVO', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jy0, 'bo', label = 'RIT', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_WVO_jym1, 'rv', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jym1, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_WVO_jy1, 'rv', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T, p00_RIT_jy1, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
plt.plot(T_can, p00_can, 'k-', label = 'Población canónica')
plt.xscale('log')
plt.ylim(None, 0.90)
plt.xlabel('Temperatura', fontsize = 15)
plt.ylabel(r'$\rho_{00}$', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
#plt.grid()
plt.legend(fontsize=12)