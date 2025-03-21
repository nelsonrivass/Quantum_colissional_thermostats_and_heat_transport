#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:35:08 2025

@author: nelson
"""

import qutip as qt 
import numpy as np
import time

inicio = time.time()

###############################################################################

# Definimos el problema. Espacios, hamiltonianos, estados iniciales, etc

dim_u = 2
dim_s = 2
dim_total = dim_u * dim_s

# Constantes

ws = 1
wu = 1
jx = 1
jy = 0
m = 0.1
k = 1
L = 50

# Distribución de momentos de las partículas a una temperatura T

def p(beta):
    p = np.sqrt(-(2*m/beta)*np.log(np.random.uniform(0,0.999)))
    return p

#  Matrices de Pauli

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
I = qt.qeye(2)

# Hamiltonianos

Hs = ws*sz
Hu = wu*sz
Ho = qt.tensor(Hs,I) + qt.tensor(I,Hu)
Hus = jx * qt.tensor(sx,sx) + jy * qt.tensor(sy,sy)
H = Ho + Hus

# Energías y autoestados de Ho y H

eigenvaluesHo, eigenstatesHo = Ho.eigenstates()
eigenvaluesH, eigenstatesH = H.eigenstates()

# Energías de los Hamltonianos individuales

eigenvaluesHs, eigenstatesHs = Hs.eigenstates() 
eigenvaluesHu, eigenstatesHu = Hu.eigenstates() 

# Función de partición del sistema

def Z_s(beta):
    Z_s = 0
    for i in range(len(eigenvaluesHs)):
        Z_s += np.exp(-beta*eigenvaluesHs[i])
    return Z_s

# Función de partición del reservorio

def Z_u(beta):
    Z_u = 0
    for i in range(len(eigenvaluesHu)):
        Z_u += np.exp(-beta*eigenvaluesHu[i])
    return Z_u

# Estado inicial del sistema: estado fundamental de Hs. Lo expresamos en la base de Hs

system_state = eigenstatesHs[0] * eigenstatesHs[0].dag()

Us = qt.Qobj([state.full().flatten() for state in eigenstatesHs], dims = [[dim_s],[dim_s]]).dag()

# Estado del reservorio. Viene dado por la temperatura del mismo

def reservoirstate(beta):
    
    reservoirstate = 0
    
    for i in range(len(eigenvaluesHu)):
        state = eigenstatesHu[i]
        reservoirstate += np.exp(-beta*eigenvaluesHu[i]) * state * state.dag()
        
    reservoirstate = (1/Z_u(beta))*reservoirstate
    
    return reservoirstate

# Energía máxima del problema necesaria par los termostátos

e_max = np.max([eigenvaluesH, eigenvaluesHo])

# Frecuencias de Bohr únicas de Ho

bohr_frequencies = np.unique([eigenvaluesHo[i] - eigenvaluesHo[j] 
                              for i in range(len(eigenvaluesHo)) 
                              for j in range(len(eigenvaluesHo)) if i != j])

###############################################################################

# Operador numero de onda y termostatos

def K(E):
    K = (2*m*(E-H)).sqrtm()
    return K

def WVO(Ep, ei, ef, ket, bra):
    E = Ep + ei
    if E > e_max:
        ki = np.sqrt(2*m*(E-ei))
        kf = np.sqrt(2*m*(E-ef))
        op = (1j*L*K(E)).expm()
        t = op.matrix_element(bra,ket) * np.exp(-1j*L*(ki+kf)/2)
        tt = (t * t.conjugate()).real
    else:
        t = (bra.dag() * ket).real
        tt = t
    return t, tt

def RIT(Ep, ei, ef, ket, bra):
    E = Ep + ei
    if E > e_max:
        tau = L/(np.sqrt(2*E/m))
        op = (-1j*tau*H).expm()
        t = op.matrix_element(bra,ket) * np.exp(1j*L*(ei+ef)/2)
        tt = (t * t.conjugate()).real
    else:
        t = (bra.dag() * ket).real
        tt = t
    return t, tt

###############################################################################

# Calculamos los operadores de Kraus

def kraus(Ep):
    
    # Operadores Ml para cada modelo 
    M_WVO = []
    M_RIT = []
    
    for Delta_l in bohr_frequencies:
        M_l_WVO = 0
        M_l_RIT = 0 
        for J in range(len(eigenvaluesHo)):
            for J_prime in range(len(eigenvaluesHo)):
                if np.isclose(eigenvaluesHo[J] - eigenvaluesHo[J_prime], Delta_l):  # Kronecker delta check
                    ket = eigenstatesHo[J]
                    bra = eigenstatesHo[J_prime]
                    ei = eigenvaluesHo[J]
                    ef = eigenvaluesHo[J_prime]
                    t_JJ_WVO = WVO(Ep, ei, ef, ket, bra)[0]
                    t_JJ_RIT = RIT(Ep, ei, ef, ket, bra)[0]
                    M_l_WVO += t_JJ_WVO * bra * ket.dag()
                    M_l_RIT += t_JJ_RIT * bra * ket.dag()
                
        M_WVO.append(M_l_WVO)           
        M_RIT.append(M_l_RIT)
        
    return  M_WVO, M_RIT

# Función para comporbar su condición

def krauscomp(Ep):
    
    M_WVO, M_RIT = kraus(Ep)
                            
    comp_WVO = 0
    comp_RIT = 0
        
    for i in range(0,len(bohr_frequencies)):
        comp_WVO += M_WVO[i].dag() * M_WVO[i]
        comp_RIT += M_RIT[i].dag() * M_RIT[i]
    
    return comp_WVO, comp_RIT

# Evolución del operador densidad con una colisión

def evol(beta, rho_WVO, rho_RIT):
    
    WVO_initial = rho_WVO
    RIT_initial = rho_RIT
   
    Ep = (p(beta)**2) / (2*m)
    
    M_WVO_op, M_RIT_op = kraus(Ep)
    
    WVO = 0
    RIT = 0
    
    for i in range(0,len(bohr_frequencies)):
        WVO += M_WVO_op[i] * WVO_initial * M_WVO_op[i].dag()
        RIT += M_RIT_op[i] * RIT_initial * M_RIT_op[i].dag()

    return WVO, RIT

###############################################################################

# Hacemos 1 sola trayectoria de  term colisiones y promediamos en las últimas prom
# estudiamos la población del nivel 0 del qubit

# Población canónica

def p00_can(beta):
    
    p_00 = (1/Z_s(beta))*np.exp(-beta*eigenvaluesHs[0])
    
    return p_00

T = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

beta = 1 / (k*T)

# Para el canónico
T_can = np.linspace(1, 1000, 100000)
beta_can = 1 / (k*T_can)

terma = 1e4
prom = 1e4
    
N_total = int(terma + prom) # Número de colisiones de termalización y de promediado

p00_WVO = np.zeros(len(T))
p00_RIT = np.zeros(len(T))

for i in range(len(T)):
    
    rho_u = reservoirstate(beta[i])
    
    rho_WVO = qt.tensor(system_state, rho_u)
    
    rho_RIT = qt.tensor(system_state, rho_u)
    
    for j in range(N_total):
    
        rho_WVO, rho_RIT = evol(beta[i], rho_WVO, rho_RIT)
        
        rho_WVO_s = qt.ptrace(rho_WVO, 0)
        
        rho_RIT_s = qt.ptrace(rho_RIT, 0)
        
        rho_WVO = qt.tensor(rho_WVO_s, rho_u)
        
        rho_RIT = qt.tensor(rho_RIT_s, rho_u)
        
        if j > terma:

            # Cambias de base
            
            rho_WVO_s = Us * rho_WVO_s * Us.dag()
            
            rho_RIT_s = Us * rho_RIT_s * Us.dag()
            
            # Población del nivel fundamental
    
            p00_WVO[i] += rho_WVO_s[0,0].real
    
            p00_RIT[i] += rho_RIT_s[0,0].real
            
    
    p00_WVO[i] = p00_WVO[i]/prom
    
    p00_RIT[i] = p00_RIT[i]/prom
    
###############################################################################

# Función para escribir en un txt

def txt3(arr1, arr2, arr3, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        f.write("T\p00_WVO\p00_RIT\n")  # Encabezados
        for valores in zip(arr1, arr2, arr3):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

def txt2(arr1, arr2, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        f.write("T\p00_can\n")  # Encabezados
        for valores in zip(arr1, arr2):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")
    
txt3(T, p00_WVO, p00_RIT, 'termalizacion_kraus_jy0_e4.txt')
txt2(T_can, p00_can(beta_can), 'termalizacion_can.txt')
    
###############################################################################

fin = time.time()
print('Tiempo de ejecución en segundos')
print(fin-inicio)