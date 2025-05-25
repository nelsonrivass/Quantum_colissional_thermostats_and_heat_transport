#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:34:10 2025

@author: nelson
"""


import qutip as qt 
import numpy as np
import scipy.integrate as spi
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
N = 1

# Distribución de momentos de las partículas a una temperatura T

def u(beta, p):
    u = (beta*p/m) * np.exp(-(beta * p**2) / (2*m))
    return u

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

rho_s = eigenstatesHs[0] * eigenstatesHs[0].dag()

# Matriz cambio de base de Ho

Uo = qt.Qobj([state.full().flatten() for state in eigenstatesHo], dims = [[dim_s for i in range(N+1)],[dim_s for i in range(N+1)]])

def Uot(rho):
    return Uo * rho * Uo.dag()

def Uotinv(rho):
    return Uo.dag() * rho * Uo

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

###############################################################################

# Operador numero de onda y termostatos

def K(E):
    K = (2*m*(E-H)).sqrtm()
    return K

def WVO(p, ei, ef, ket, bra):
    E = p**2/(2*m) + ei
    if E > e_max:
        ki = np.sqrt(2*m*(E-ei))
        kf = np.sqrt(2*m*(E-ef))
        op = (1j*L*K(E)).expm()
        t = op.matrix_element(bra,ket) * np.exp(-1j*L*(ki+kf)/2)
        tt = (t * t.conjugate()).real
    else:
        t = (bra.dag() * ket).real
        tt = t
    return tt

def RIT(p, ei, ef, ket, bra):
    E = p**2/(2*m) + ei
    if E > e_max:
        tau = L/(np.sqrt(2*E/m))
        op = (-1j*tau*H).expm()
        t = op.matrix_element(bra,ket) * np.exp(1j*L*(ei+ef)/2)
        tt = (t * t.conjugate()).real
    else:
        t = (bra.dag() * ket).real
        tt = t
    return tt

def RIT_mal(p, ei, ef, ket, bra):
    E = p**2/(2*m) + ei
    if E > e_max:
        tau = L*m/p
        op = (-1j*tau*H).expm()
        t = op.matrix_element(bra,ket) * np.exp(1j*L*(ei+ef)/2)
        tt = (t * t.conjugate()).real
    else:
        t = (bra.dag() * ket).real
        tt = t
    return tt

###############################################################################

# Función para calcular la matriz de ritmos/probabilidades

def ritmos(beta):
    
    ritmos_WVO = np.zeros((dim_total, dim_total))
    ritmos_RIT = np.zeros((dim_total, dim_total))
    
    for j_prima in range(dim_total):
        
        ef = eigenvaluesHo[j_prima]
        bra = eigenstatesHo[j_prima]
        
        for j in range(dim_total):
            
            ei = eigenvaluesHo[j]
            ket = eigenstatesHo[j]
            
            ritmos_WVO[j_prima,j], error = spi.quad(lambda p: WVO(p, ei, ef, ket, bra) * u(beta, p), 0, np.inf)
            ritmos_RIT[j_prima,j], error = spi.quad(lambda p: RIT(p, ei, ef, ket, bra) * u(beta, p), 0, np.inf)
            
    return ritmos_WVO, ritmos_RIT
    
    
# Evolución del operador densidad con los ritmos promediados

def evol(ritmos_WVO, ritmos_RIT, rho_WVO, rho_RIT):
    
    WVO_initial = rho_WVO
    RIT_initial = rho_RIT
    
    rho_WVO = np.zeros((dim_total, dim_total))
    rho_RIT = np.zeros((dim_total, dim_total))
    
    for j_prima in range(dim_total):
        
        for j in range(dim_total):
            
            rho_WVO[j_prima,j_prima] += ritmos_WVO[j_prima,j] * WVO_initial[j,j].real
            
            rho_RIT[j_prima,j_prima] += ritmos_RIT[j_prima,j] * RIT_initial[j,j].real
            
    rho_WVO = qt.Qobj(rho_WVO, dims = [[2] * (N+1), [2] * (N+1)])
    
    rho_RIT = qt.Qobj(rho_RIT, dims = [[2] * (N+1), [2] * (N+1)])

    return rho_WVO, rho_RIT

def comp(beta):
   
    ritmos_WVO, ritmos_RIT = ritmos(beta)
    
    comp_WVO = np.zeros((dim_total, dim_total))
    comp_RIT = np.zeros((dim_total, dim_total))
    
    for j in range(dim_total):
        
        for j_prima in range(dim_total):
            
            comp_WVO[j,j] += ritmos_WVO[j_prima,j]
            
            comp_RIT[j,j] += ritmos_RIT[j_prima,j]
    
    return comp_WVO, comp_RIT

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
C_total = int(terma + prom) # Número de colisiones de termalización y de promediado

p00_WVO = np.zeros(len(T))
p00_RIT = np.zeros(len(T))

for i in range(len(T)):
    
    rho_u = reservoirstate(beta[i])
    
    ritmos_WVO, ritmos_RIT = ritmos(beta[i])
    
    rho_WVO = Uot(qt.tensor(rho_s, rho_u))
    
    rho_RIT = Uot(qt.tensor(rho_s, rho_u))
    
    
    for j in range(C_total):
    
        rho_WVO, rho_RIT = evol(ritmos_WVO, ritmos_RIT, rho_WVO, rho_RIT)
        
        rho_WVO = Uotinv(rho_WVO)
        
        rho_RIT = Uotinv(rho_RIT)
        
        rho_WVO_s = qt.ptrace(rho_WVO, 0)
        
        rho_RIT_s = qt.ptrace(rho_RIT, 0)
        
        rho_WVO = Uot(qt.tensor(rho_WVO_s, rho_u))
        
        rho_RIT = Uot(qt.tensor(rho_RIT_s, rho_u))
              
        if j > terma:
            
            # Población del nivel fundamental
    
            p00_WVO[i] += rho_WVO_s[1,1].real
    
            p00_RIT[i] += rho_RIT_s[1,1].real
            
            
    p00_WVO[i] = p00_WVO[i]/prom
    
    p00_RIT[i] = p00_RIT[i]/prom

###############################################################################

# Estudiamos la termalización del espín en función de las colisiones

C_total = 50

C_array = np.array([i+1 for i in range(C_total)]) 

T_ind = 10
beta_ind = 1/T_ind

ritmos_WVO, ritmos_RIT = ritmos(beta_ind)

T_espin_WVO = np.zeros(C_total)
T_espin_RIT = np.zeros(C_total)

# Matrices densidad de cada uno de los reservorios

rho_u = reservoirstate(beta_ind)

# Matrices densidad iniciales de la cadena

rho_s_WVO = rho_s
rho_s_RIT = rho_s

# Empezamos las colisiones

for i in range(C_total):
    
    rho_WVO = Uot(qt.tensor(rho_s_WVO, rho_u))
    rho_RIT = qt.tensor(rho_s_RIT, rho_u)
    
    rho_WVO, rho_RIT = evol(ritmos_WVO, ritmos_RIT, rho_WVO, rho_RIT)

    rho_s_WVO = qt.ptrace(rho_WVO, 0)

    rho_s_RIT = qt.ptrace(rho_RIT, 0)
        
    T_espin_WVO[i] = (-2*ws) / (np.log(rho_s_WVO[1,1].real/rho_s_WVO[0,0].real))
    
    T_espin_RIT[i] = (-2*ws) / (np.log(rho_s_RIT[1,1].real/rho_s_RIT[0,0].real))

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
    
# txt3(T, p00_WVO, p00_RIT,'termalizacion_nokraus_prom_jy0_1e4.txt') 
# txt2(T_can, p00_can(beta_can), 'termalizacion_can.txt')
# txt3(C_array, T_espin_WVO, T_espin_RIT, 'termalizacion_colisiones_jy0_5e1.txt')
    
###############################################################################

fin = time.time()
print('Tiempo de ejecución en segundos')
print(fin-inicio)