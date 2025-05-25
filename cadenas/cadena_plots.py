#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:58:04 2025

@author: nelson
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.ticker import MaxNLocator

###############################################################################

def load_data(file_name, num_columns):
    try:
        data = np.loadtxt(file_name, usecols=range(num_columns))
        return data
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None 
    
###############################################################################
    
# # Empezamos con una cadena con un único termostato

# # Estudiamos la población del nivel fundamental de la cadena en función de la temperatura

# T, p00_WVO_N2, p00_RIT_N2 = np.loadtxt('termalizacion_N2_XXX_1e3.txt', dtype = float, unpack = True)
# T_can, p00_can_N2 = np.loadtxt('termalizacion_N2_XXX_can.txt', dtype = float, unpack = True)
# T, p00_WVO_N3, p00_RIT_N3 = np.loadtxt('termalizacion_N3_XXZ_1e4.txt', dtype = float, unpack = True)
# T_can, p00_can_N3 = np.loadtxt('termalizacion_N3_XXZ_can.txt', dtype = float, unpack = True)
# T, p00_WVO_N4, p00_RIT_N4 = np.loadtxt('termalizacion_N4_XYZ_1e5.txt', dtype = float, unpack = True)
# T_can, p00_can_N4 = np.loadtxt('termalizacion_N4_XYZ_can.txt', dtype = float, unpack = True)

# plt.figure(1)
# plt.plot(T, p00_WVO_N2, 'ro', markersize = 10,  fillstyle='none', markeredgewidth=2, label=r'$N = 2 \ (XXX)$')
# #plt.plot(T, p00_RIT_N2, 'ro', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(T_can, p00_can_N2, 'k-')
# plt.plot(T, p00_WVO_N3, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2,  label=r'$N = 3 \ (XXZ)$')
# #plt.plot(T, p00_RIT_N3, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(T_can, p00_can_N3, 'k-')
# plt.plot(T, p00_WVO_N4, 'go', markersize = 10,  fillstyle='none', markeredgewidth=2, label=r'$N = 4 \ (XYZ)$')
# #plt.plot(T, p00_RIT_N4, 'go', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(T_can, p00_can_N4, 'k-', label = 'Población canónica')
# plt.xscale('log')
# plt.ylim(0,0.8)
# plt.xlabel(r'$T_U$', fontsize = 15)
# plt.ylabel(r'$\rho_{S}^{GS}$', fontsize = 15)
# plt.tick_params(axis='both', labelsize=12)

# # Crear handles personalizados solo con el color
# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label=r'$N = 2 \ (XXX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$N = 3 \ (XXZ)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label=r'$N = 4 \ (XYZ)$'),
# #     Line2D([0], [0], color='k', linewidth = 2, label='Población canónica')
# # ]

# plt.legend(fontsize = 12)
# plt.show()


# # Ploteamos varios perfiles de temperatura

# N_norm_3, perfil_WVO_N3, perfil_RIT_N3 = np.loadtxt('T10_perfil_N3_XX_3e2.txt', dtype = float, unpack = True)
# N_norm_4, perfil_WVO_N4, perfil_RIT_N4 = np.loadtxt('T15_perfil_N4_XXX_3e2.txt', dtype = float, unpack = True)
# N_norm_5, perfil_WVO_N5, perfil_RIT_N5 = np.loadtxt('T20_perfil_N5_XXZ_3e2.txt', dtype = float, unpack = True)
# N_norm_6, perfil_WVO_N6, perfil_RIT_N6 = np.loadtxt('T25_perfil_N6_XYZ_3e2.txt', dtype = float, unpack = True)


# plt.figure(2)
# plt.plot(N_norm_3, perfil_WVO_N3 , 'bo',  markersize = 10, label = r'$(3,10,XX)$')
# #plt.plot(N_norm_3, perfil_RIT_N3 , 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N_norm_4, perfil_WVO_N4 , 'go',  markersize = 10, label = r'$(4,15,XXX)$')
# #plt.plot(N_norm_4, perfil_RIT_N4 , 'go', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N_norm_5, perfil_WVO_N5 , 'yo',  markersize = 10, label = r'$(5,20,XXZ)$')
# #plt.plot(N_norm_5, perfil_RIT_N5 , 'yo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N_norm_6, perfil_WVO_N6 , 'mo',   markersize = 10, label = r'$(6,25,XYZ)$')
# #plt.plot(N_norm_6, perfil_RIT_N6 , 'mo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.xlabel(r'$i/N$', fontsize = 20)
# plt.ylabel(r'$T_i$', fontsize = 20)
# plt.tick_params(axis='both', labelsize=15)

# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$N = 3 \ (XX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label=r'$N = 4 \ (XXX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='y', markersize=8, label=r'$N = 5 \ (XXZ)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=8, label=r'$N = 6 \ (XYZ)$')
# # ]

# plt.legend(fontsize = 14, loc=(0.095, 0.025))
# plt.ylim(8,26)
# plt.show()


# # Mostamos el proceso de termalización en función de las colisiones

# N = 6

# C_array1 = np.loadtxt('colisiones_3e2.txt', dtype = float, unpack = True)
# # T_WVO1 = load_data('T20_T_WVO_N5_XXZ_3e2.txt', N)
# T_WVO1 = load_data('T25_T_WVO_N6_XYZ_3e2.txt', N)


# for l in range(N):
#     plt.figure(3)
#     plt.plot(C_array1[:250], T_WVO1[:,l][:250], '.', alpha = 1, label = f'Espín {l+1}')
 
# plt.figure(3)
# plt.xlabel('Colisiones', fontsize = 15)
# plt.ylabel(r'$T_i$', fontsize = 15)
# plt.tick_params(axis='both', labelsize=12)
# plt.legend(fontsize = 10, loc=(0.015, 0.65))


###############################################################################

# # Cadena con dos termostátos

# # Ploteamos un par de perfiles de temperatura 

# N_norm_3, perfil_WVO_N3, perfil_RIT_N3 = np.loadtxt('T15T10_perfil_N3_XYZ_3e2.txt', dtype = float, unpack = True)
# N_norm_4, perfil_WVO_N4, perfil_RIT_N4 = np.loadtxt('T15T10_perfil_N4_XYZ_3e2.txt', dtype = float, unpack = True)
# N_norm_5, perfil_WVO_N5, perfil_RIT_N5 = np.loadtxt('T15T10_perfil_N5_XYZ_3e2.txt', dtype = float, unpack = True)
# N_norm_6, perfil_WVO_N6, perfil_RIT_N6 = np.loadtxt('T15T10_perfil_N6_XYZ_3e2.txt', dtype = float, unpack = True)

# plt.figure(4)
# plt.plot(N_norm_3, perfil_WVO_N3 , 'bo', markersize = 10, label=r'$N = 3$')
# #plt.plot(N_norm_3, perfil_RIT_N3 , 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N_norm_4, perfil_WVO_N4 , 'go', markersize = 10, label=r'$N = 4$')
# #plt.plot(N_norm_4, perfil_RIT_N4 , 'go', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N_norm_5, perfil_WVO_N5 , 'yo', markersize = 10, label=r'$N = 5$')
# #plt.plot(N_norm_5, perfil_RIT_N5 , 'yo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N_norm_6, perfil_WVO_N6 , 'mo', markersize = 10, label=r'$N = 6$')
# #plt.plot(N_norm_6, perfil_RIT_N6 , 'mo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.xlabel(r'$i/N$', fontsize = 20)
# plt.ylabel(r'$T_i$', fontsize = 20)
# plt.ylim(10, 13)
# plt.tick_params(axis='both', labelsize=15)

# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$N = 3 \ (XX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label=r'$N = 4 \ (XXX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='y', markersize=8, label=r'$N = 5 \ (XXZ)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=8, label=r'$N = 6 \ (XYZ)$')
# # ]

# plt.legend(fontsize = 14)
# plt.ylim(11.25,12.5)
# plt.yticks([11.25, 11.75, 12.25, 12.75])
# plt.show()

# # Proceso de termalización

# N = 6

# C_array2 = np.loadtxt('colisiones_3e2.txt', dtype = float, unpack = True)
# T_WVO2 = load_data('T13T10_T_WVO_N6_XX_3e2.txt', N)


# for l in range(N):
#     plt.figure(5)
#     plt.plot(C_array2, T_WVO2[:,l], '.', label = f'Espín {l}')
 
# plt.figure(5)
# plt.xlabel('Colisiones', fontsize = 15)
# plt.ylabel(r'$T_i$', fontsize = 15)
# plt.tick_params(axis='both', labelsize=12)
# plt.legend(fontsize=12)

# C_array,  Q_1_WVO, Q_1_RIT, Q_N_WVO, Q_N_RIT = np.loadtxt('T13T10_Q_N6_XX_3e2.txt', dtype = float, unpack = True)

# plt.figure(6)
# plt.plot(C_array, Q_1_WVO, 'r.', markersize = 5, label=r'$\delta Q_{in}$')
# #plt.plot(C_array, Q_1_RIT, 'ro', markersize = 3,  fillstyle='none', markeredgewidth=2)
# plt.plot(C_array, Q_N_WVO, 'b.', markersize = 5, label=r'$\delta Q_{out}$')
# #plt.plot(C_array, Q_N_RIT, 'bo', markersize = 3,  fillstyle='none', markeredgewidth=2)
# plt.xlabel('Colisiones', fontsize = 15)
# plt.ylabel(r'$\delta Q$', fontsize = 15)
# plt.ylim(-0.009, 0.1)
# plt.tick_params(axis='both', labelsize=12)
# plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label=r'$\Delta Q_{in}$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$\Delta Q_{out}$')
# # ]

# plt.legend(fontsize = 15, loc=(0.70, 0.15))
# plt.show()

# # Ley de Fourier

# # DeltaQ v N

# N, Q_N_XX_WVO, Q_N_XX_RIT =  np.loadtxt('QvN_XX_T11T10.txt', dtype = float, unpack = True)
# N, Q_N_XYZ_WVO, Q_N_XYZ_RIT =  np.loadtxt('QvN_XYZ_T11T10.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N6_XX_WVO, Q_T_N6_XX_RIT =  np.loadtxt('QvDeltaT_XX_N6.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N5_XX_WVO, Q_T_N5_XX_RIT =  np.loadtxt('QvDeltaT_XX_N5.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N4_XX_WVO, Q_T_N4_XX_RIT =  np.loadtxt('QvDeltaT_XX_N4.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N3_XX_WVO, Q_T_N3_XX_RIT =  np.loadtxt('QvDeltaT_XX_N3.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N6_XYZ_WVO, Q_T_N6_XYZ_RIT =  np.loadtxt('QvDeltaT_XYZ_N6.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N5_XYZ_WVO, Q_T_N5_XYZ_RIT =  np.loadtxt('QvDeltaT_XYZ_N5.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N4_XYZ_WVO, Q_T_N4_XYZ_RIT =  np.loadtxt('QvDeltaT_XYZ_N4.txt', dtype = float, unpack = True)
# DeltaT, Q_T_N3_XYZ_WVO, Q_T_N3_XYZ_RIT =  np.loadtxt('QvDeltaT_XYZ_N3.txt', dtype = float, unpack = True)

# # DeltaQ v N

# # Modelo de ley de potencias
# def power_law(N, c, a):
#     return c * N**a

# fit_XX, covarianceXX = curve_fit(power_law, N, Q_N_XX_WVO)
# c_XX, a_XX = fit_XX

# fit_XYZ, covarianceXYZ = curve_fit(power_law, N, Q_N_XYZ_WVO)
# c_XYZ, a_XYZ = fit_XYZ

# N_fit = np.linspace(min(N), max(N), 100)
# Q_fit_XX = power_law(N_fit, *fit_XX)
# Q_fit_XYZ = power_law(N_fit, *fit_XYZ)

# # Errores estándar: sqrt de la diagonal de la matriz de covarianza
# errorsXX = np.sqrt(np.diag(covarianceXX))
# c_errXX, a_errXX = errorsXX

# errorsXYZ = np.sqrt(np.diag(covarianceXYZ))
# c_errXYZ, a_errXYZ = errorsXYZ

# # Mostrar resultados
# print(f"aXX = {a_XX:.4f} ± {a_errXX:.4f}")
# print(f"aXYZ = {a_XYZ:.4f} ± {a_errXYZ:.4f}")

# plt.figure(7)
# plt.plot(N, Q_N_XX_WVO, 'ro', markersize = 8, label=r'$XX$')
# plt.plot(N_fit, Q_fit_XX, 'r--', label=f'$\delta Q = {c_XX:.2e} N^{{{a_XX:.2f}}}$')
# #plt.plot(1/N, Q_N_XX_RIT, 'ro', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N, Q_N_XYZ_WVO, 'bo', markersize = 8, label=r'$XYZ$')
# plt.plot(N_fit, Q_fit_XYZ, 'b--', label=f'$\delta Q = {c_XYZ:.2e} N^{{{a_XYZ:.2f}}}$')
# #plt.plot(1/N, Q_N_XYZ_RIT, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.xlabel(r'$N$', fontsize = 15)
# plt.ylabel(r'$\delta Q$', fontsize = 15)
# plt.ylim(None, 2.6e-3)
# plt.tick_params(axis='both', labelsize=12)
# ax = plt.gca() 
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-3, 3))
# ax.yaxis.set_major_formatter(formatter)

# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label=r'$XXZ$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$XYZ$')
# # ]

# plt.legend(fontsize = 10, loc=(0.02, 0.025))
# plt.show()

# # DeltaQ v Delta T

# # Hacemos los ajustes Y calculamos conductividades. Caso XX

# RXX6WVO = np.polyfit(DeltaT, Q_T_N6_XX_WVO, 1)
# #RXX6RIT = np.polyfit(DeltaT, Q_T_N6_XX_RIT, 1)
# RXX5WVO = np.polyfit(DeltaT, Q_T_N5_XX_WVO, 1)
# #RXX5RIT = np.polyfit(DeltaT, Q_T_N5_XX_RIT, 1)
# RXX4WVO = np.polyfit(DeltaT, Q_T_N4_XX_WVO, 1)
# #RXX4RIT = np.polyfit(DeltaT, Q_T_N4_XX_RIT, 1)
# RXX3WVO = np.polyfit(DeltaT, Q_T_N3_XX_WVO, 1)
# #RXX3RIT = np.polyfit(DeltaT, Q_T_N3_XX_RIT, 1)

# kXXWVO = np.array([RXX3WVO[0]*3,
#                     RXX4WVO[0]*4,
#                     RXX5WVO[0]*5,
#                     RXX6WVO[0]*6])
# #kXXRIT = np.array([RXX6RIT[0]*6,
# #                     RXX5RIT[0]*5,
# #                     RXX4RIT[0]*4,
# #                     RXX3RIT[0]*3])

# plt.figure(8)
# plt.plot(DeltaT, Q_T_N6_XX_WVO, 'mo', markersize = 8, markeredgewidth=2, label=r'$N = 6$')
# plt.plot(DeltaT, np.polyval(RXX6WVO, DeltaT), 'm--')
# #plt.plot(DeltaT, Q_T_N6_XX_RIT, 'mo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXX6RIT, DeltaT), 'm--')
# plt.plot(DeltaT, Q_T_N5_XX_WVO, 'yo', markersize = 8, markeredgewidth=2, label=r'$N = 5$')
# plt.plot(DeltaT, np.polyval(RXX5WVO, DeltaT), 'y--')
# #plt.plot(DeltaT, Q_T_N5_XX_RIT, 'yo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXX5RIT, DeltaT), 'y--')
# plt.plot(DeltaT, Q_T_N4_XX_WVO, 'go', markersize = 8, markeredgewidth=2, label=r'$N = 4$')
# plt.plot(DeltaT, np.polyval(RXX4WVO, DeltaT), 'g--')
# #plt.plot(DeltaT, Q_T_N4_XX_RIT, 'go', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXX4RIT, DeltaT), 'g--')
# plt.plot(DeltaT, Q_T_N3_XX_WVO, 'bo', markersize = 8, markeredgewidth=2, label=r'$N = 3$')
# plt.plot(DeltaT, np.polyval(RXX3WVO, DeltaT), 'b--')
# #plt.plot(DeltaT, Q_T_N3_XX_RIT, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXX3RIT, DeltaT), 'b--')

# plt.xlabel(r'$\Delta T$', fontsize = 15)
# plt.ylabel(r'$\delta Q$', fontsize = 15)
# plt.tick_params(axis='both', labelsize=12)
# ax = plt.gca() 
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-3, 3))
# ax.yaxis.set_major_formatter(formatter)

# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$N = 3 \ (XX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label=r'$N = 4 \ (XXX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='y', markersize=8, label=r'$N = 5 \ (XXZ)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=8, label=r'$N = 6 \ (XYZ)$')
# # ]

# plt.legend(fontsize = 12)
# plt.show()

# # Caso XYZ

# RXYZ6WVO = np.polyfit(DeltaT, Q_T_N6_XYZ_WVO, 1)
# #RXYZ6RIT = np.polyfit(DeltaT, Q_T_N6_XYZ_RIT, 1)
# RXYZ5WVO = np.polyfit(DeltaT, Q_T_N5_XYZ_WVO, 1)
# #RXYZ5RIT = np.polyfit(DeltaT, Q_T_N5_XYZ_RIT, 1)
# RXYZ4WVO = np.polyfit(DeltaT, Q_T_N4_XYZ_WVO, 1)
# #RXYZ4RIT = np.polyfit(DeltaT, Q_T_N4_XYZ_RIT, 1)
# RXYZ3WVO = np.polyfit(DeltaT, Q_T_N3_XYZ_WVO, 1)
# #RXYZ3RIT = np.polyfit(DeltaT, Q_T_N3_XYZ_RIT, 1)

# kXYZWVO = np.array([RXYZ3WVO[0]*3,
#                     RXYZ4WVO[0]*4,
#                     RXYZ5WVO[0]*5,
#                     RXYZ6WVO[0]*6])
# # kXYZRIT = np.array([RXYZ6RIT[0]*6,
# #                     RXYZ5RIT[0]*5,
# #                     RXYZ4RIT[0]*4,
# #                     RXYZ3RIT[0]*3])

# plt.figure(9)
# plt.plot(DeltaT, Q_T_N6_XYZ_WVO, 'mo', markersize = 8, markeredgewidth=2, label=r'$N = 6$')
# plt.plot(DeltaT, np.polyval(RXYZ6WVO, DeltaT), 'm--')
# #plt.plot(DeltaT, Q_T_N6_XYZ_RIT, 'mo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXYZ6RIT, DeltaT), 'm--')
# plt.plot(DeltaT, Q_T_N5_XYZ_WVO, 'yo', markersize = 8, markeredgewidth=2, label=r'$N = 5$')
# plt.plot(DeltaT, np.polyval(RXYZ5WVO, DeltaT), 'y--')
# #plt.plot(DeltaT, Q_T_N5_XYZ_RIT, 'yo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXYZ5RIT, DeltaT), 'y--')
# plt.plot(DeltaT, Q_T_N4_XYZ_WVO, 'go', markersize = 8, markeredgewidth=2, label=r'$N = 4$')
# plt.plot(DeltaT, np.polyval(RXYZ4WVO, DeltaT), 'g--')
# #plt.plot(DeltaT, Q_T_N4_XYZ_RIT, 'go', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXYZ4RIT, DeltaT), 'g--')
# plt.plot(DeltaT, Q_T_N3_XYZ_WVO, 'bo', markersize = 8, markeredgewidth=2, label=r'$N = 3$')
# plt.plot(DeltaT, np.polyval(RXYZ3WVO, DeltaT), 'b--')
# #plt.plot(DeltaT, Q_T_N3_XYZ_RIT, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)
# #plt.plot(DeltaT, np.polyval(RXYZ3RIT, DeltaT), 'b--')

# plt.xlabel(r'$\Delta T$', fontsize = 15)
# plt.ylabel(r'$\delta Q$', fontsize = 15)
# plt.tick_params(axis='both', labelsize=12)
# ax = plt.gca() 
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-3, 3))
# ax.yaxis.set_major_formatter(formatter)

# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$N = 3 \ (XX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label=r'$N = 4 \ (XXX)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='y', markersize=8, label=r'$N = 5 \ (XXZ)$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=8, label=r'$N = 6 \ (XYZ)$')
# # ]

# plt.legend(fontsize = 12)
# plt.show()

# # Graficamos las conductividades 

# plt.figure(10)
# plt.plot(N, kXXWVO, 'ro', markersize = 8,label=r'$XX$')
# #plt.plot(N, kXXRIT, 'ro', markersize = 10,  fillstyle='none', markeredgewidth=2)
# plt.plot(N, kXYZWVO, 'bo', markersize = 8, label=r'$XYZ$')
# #plt.plot(N, kXYZRIT, 'bo', markersize = 10,  fillstyle='none', markeredgewidth=2)

# plt.xlabel(r'$N$', fontsize = 15)
# plt.ylabel(r'$\kappa$', fontsize = 15)
# plt.tick_params(axis='both', labelsize=12)
# ax = plt.gca() 
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-3, 3))
# ax.yaxis.set_major_formatter(formatter)

# # legend_handles = [
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label=r'$XX$'),
# #     Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label=r'$XYZ$')
# # ]

# plt.legend(fontsize = 12)
# plt.show()