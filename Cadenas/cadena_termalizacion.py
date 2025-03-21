#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:57:23 2025

@author: nelson
"""

import qutip as qt 
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import time

inicio = time.time()

###############################################################################

#  Matrices de Pauli

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
I2 = qt.qeye(2)

# Constantes

wu = 1 # Frecuencia de las unidades

ws = 1 # Parámetros de la cadena
J = 1
delta = 0.5*J

jx = 1  # Acoplos entre unidades y el extremo de la cadena
jy = 0

m = 0.1 
k = 1
L = 50

# Hamiltoniano individual

Hind = ws * sz

eigenvaluesHind, eigenstatesHind = Hind.eigenstates()

# Distribución de momentos de las partículas a una temperatura T

def u(beta, p):
    u = (beta*p/m) * np.exp(-(beta * p**2) / (2*m))
    return u

# Conmutador

def comm(A, B):
    return A * B - B * A

###############################################################################

# Definimos el problema. Espacios, hamiltonianos, estados iniciales, etc.

# Se trata de cadenas de N espines S = 1/2

s = 1/2
N = 4

# La dimensión del espacio de Hilbert de cada uno de estos espines es

dim = 2*s + 1

# Por tannto, la dimensión del sistema será

dim_s = dim**N

# Las unidades con las que bombardearemos el sistema serán qubits igual que antes

dim_u = dim

# La dimensión total en cada colisión será

dim_total = int(dim_u * dim_s)

# Hamiltonianos individuales de espines i e i+1

def e(i):
    
    e = 0
    e += J * (qt.tensor([I2]* i + [sx,sx] + [I2] * (N - i - 2)))
    e += J * (qt.tensor([I2]* i + [sy,sy] + [I2] * (N - i - 2)))
    e += delta * (qt.tensor([I2]* i + [sz,sz] + [I2] * (N - i - 2)))
    e += ws * (qt.tensor([I2] * i + [sz] + [I2] * (N - i - 1)))
               
    return e

# Contruimos el Hamiltoniano del sistema 

def Hs_func(N):
    
    # Inicializar el Hamiltoniano en cero
    Hs = 0 
    
    # Construcción del Hamiltoniano sumando los términos de acoplamiento entre vecinos
    for j in range(N - 1): 
        Hs += J * qt.tensor([I2] * j + [sx, sx] + [I2] * (N - j - 2))
        Hs += J * qt.tensor([I2] * j + [sy, sy] + [I2] * (N - j - 2))

    # Término de energía (espín total)
    for j in range(N):
        Hs += ws * qt.tensor([I2] * j + [sz] + [I2] * (N - j - 1))
    
    return Hs

# Contruimos el Hamiltoniano de interacción del primer y último espin

def Hus_func_1(N):

    # Inicializar el Hamiltoniano en cero
    Hus_0 = 0
    
    # Construcción del Hamiltoniano
    
    Hus_0 += jx * qt.tensor([sx,sx] + [I2] * (N-1))
    Hus_0 += jy * qt.tensor([sy,sy] + [I2] * (N-1))

    return Hus_0

def Hus_func_N(N):

    # Inicializar el Hamiltoniano en cero
    Hus_N = 0
    
    # Construcción del Hamiltoniano
    Hus_N += jx * qt.tensor([I2] * (N-1) +[sx,sx])
    Hus_N += jy * qt.tensor([I2] * (N-1) +[sy,sy])
    
    return Hus_N

# Ahora añadimos el hamiltoniano libre de las unidades Hu para formar Ho

I = qt.tensor(I2 for i in range(N))
Hs = Hs_func(N)
Hu = wu*sz

# Hamiltomianos libres

Ho_1 = qt.tensor(Hu,I) + qt.tensor(I2,Hs)

Ho_N = qt.tensor(Hs,I2) + qt.tensor(I,Hu)

# Hamiltonianos de interacción

Hus_1 = Hus_func_1(N)

Hus_N = Hus_func_N(N)

H_1 = Ho_1 + Hus_1

H_N = Ho_N + Hus_N

# Será util luego crear una lista para las trazas

N_s_1 = N_s = np.array([i+1 for i in range(N)])

N_s_N = np.array([i for i in range(N)])

N_1 = np.append(0, N_s_1)

N_N = np.append(N_s_1, N+1)

# Ahora hacemos un array pensado solo para graficar

N_norm = np.linspace(0, 1, N)

###############################################################################

# Diagonalizamos Ho y H y definimos el estado inicial del sistema y del reservorio

# Energías y autoestados de Ho y H

eigenvaluesHo_1, eigenstatesHo_1 = Ho_1.eigenstates()
eigenvaluesHo_N, eigenstatesHo_N = Ho_N.eigenstates()
eigenvaluesH_1, eigenstatesH_1 = H_1.eigenstates()
eigenvaluesH_N, eigenstatesH_N = H_N.eigenstates()

# Lo mismo con Hs y Hu

eigenvaluesHs, eigenstatesHs = Hs.eigenstates()
eigenvaluesHu, eigenstatesHu = Hu.eigenstates()

# Estado inicial del sistema. Estado fundamental puro

rho_s = eigenstatesHs[0] * eigenstatesHs[0].dag() 

# La expresamos en la base de Hs

Us = qt.Qobj([state.full().flatten() for state in eigenstatesHs], dims = [[dim for i in range(N)],[dim for i in range(N)]])

rho_s = Us * rho_s * Us.dag()

Uu = qt.Qobj([state.full().flatten() for state in eigenstatesHu], dims = [[dim],[dim]])

Uind = qt.Qobj([state.full().flatten() for state in eigenstatesHind], dims = [[dim],[dim]])

###############################################################################

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


###############################################################################

# Estado del reservorio que depende de la temperatura

def reservoirstate(beta):
    
    reservoirstate = 0
    
    for i in range(len(eigenvaluesHu)):
        state = eigenstatesHu[i]
        reservoirstate += np.exp(-beta*eigenvaluesHu[i]) * state * state.dag()
        
    reservoirstate = (1/Z_u(beta))*reservoirstate
    
    reservoirstate = Uu * reservoirstate * Uu.dag()
    
    return reservoirstate

# Energía máxima del problema necesaria par los termostátos

e_max = np.max([eigenvaluesH_1, eigenvaluesHo_1])

# e_max = np.max([eigenvaluesH_N, eigenvaluesHo_N])

###############################################################################
    
# Operador numero de onda y termostatos

def K(E, H):
    K = (2*m*(E-H)).sqrtm()
    return K

def WVO(p, ei, ef, ket, bra, H):
    E = p**2/(2*m) + ei
    if E > e_max:
        ki = np.sqrt(2*m*(E-ei))
        kf = np.sqrt(2*m*(E-ef))
        op = (1j*L*K(E, H)).expm()
        t = op.matrix_element(bra,ket) * np.exp(-1j*L*(ki+kf)/2)
        tt = (t * t.conjugate()).real
    else:
        t = (bra.dag() * ket).real
        tt = t
    return tt

def RIT(p, ei, ef, ket, bra, H):
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

###############################################################################

# Evolución del operador densidad con una colisión

def ritmos(beta, eigenvaluesHo, eigenstatesHo, H):
    
    ritmos_WVO = np.zeros((dim_total, dim_total))
    ritmos_RIT = np.zeros((dim_total, dim_total))
    
    for j_prima in range(dim_total):
        
        ef = eigenvaluesHo[j_prima]
        bra = eigenstatesHo[j_prima]
        
        for j in range(dim_total):
            
            ei = eigenvaluesHo[j]
            ket = eigenstatesHo[j]
            
            ritmos_WVO[j_prima,j], error = spi.quad(lambda p: WVO(p, ei, ef, ket, bra, H) * u(beta, p), 0, np.inf)
            ritmos_RIT[j_prima,j], error = spi.quad(lambda p: RIT(p, ei, ef, ket, bra, H) * u(beta, p), 0, np.inf)
            
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

def comp(beta, eigenvaluesHo, eigenstatesHo, H):
   
    ritmos_WVO, ritmos_RIT = ritmos(beta, eigenvaluesHo, eigenstatesHo, H)
    
    comp_WVO = np.zeros((dim_total, dim_total))
    comp_RIT = np.zeros((dim_total, dim_total))
    
    for j in range(dim_total):
        
        for j_prima in range(dim_total):
            
            comp_WVO[j,j] += ritmos_WVO[j_prima,j]
            
            comp_RIT[j,j] += ritmos_RIT[j_prima,j]
    
    return comp_WVO, comp_RIT


###############################################################################

# Nos enfocaremos en calcular perfiles de temperatura y flujos
# i índice del espín

# Temperatura

def T_espin(rho, i):
    
    rho_i = qt.ptrace(rho, i)
    
    T = (-2*ws) / (np.log(rho_i[1,1].real/rho_i[0,0].real))
    
    T = np.abs(T)
    
    return T

# Energía local

def E_espin(rho, i):
    
    if i == N - 1:
        E = (rho*e(i-1)).tr()
    else:
        E = (rho*e(i)).tr()
    
    return E.real

# Flujo

def J_espin(rho, i):
    
    if i == 1:
        
        rho_s = qt.ptrace(rho, N_1)
        
        op = comm(Hus_1,qt.tensor(I2,e(1)))
        
        J = ((rho_s*op).tr()).real
        
    if i > 1 and i < N:
        
        rho_s = qt.ptrace(rho, N_s)
    
        op = comm(e(i-1),e(i))
    
        J = (1j*(rho_s*op).tr()).real
    
    if i == N:
        
        rho_s = qt.ptrace(rho, N_N)
        
        op = comm(Hus_N,qt.tensor(e(N-1),I2))
        
        J = ((rho_s*op).tr()).real
      
    return J

###############################################################################

# Ahora simulamos. Definimos las temperaturas de cada termostáto. El 1 es el caliente 
# y el N+1 el frío.

T1 = 100
beta1 = 1/T1
TN = 10
betaN = 1/TN

# Matrices densidad de cada uno de los reservorios

rho_u1 = reservoirstate(beta1)
rho_uN = reservoirstate(betaN)

rho_s_WVO = rho_s
rho_s_RIT = rho_s

# Colisiones de termalización y de promediado

terma = int(1e2)
prom = int(1e2)
C_total = int(terma + prom)

# Array con número de colisiones

C_array = np.array([i+1 for i in range(C_total)])

# Matriz con las temperaturas, energías locales y flujos

T_espines_WVO = np.zeros((N,C_total))
T_espines_RIT = np.zeros((N,C_total))

E_espines_WVO = np.zeros((N,C_total))
E_espines_RIT = np.zeros((N,C_total))

J_espines_WVO = np.zeros((N,C_total))
J_espines_RIT = np.zeros((N,C_total))

# Calculamos los ritmos promediados por las colisiones calientes y frías

ritmos_WVO_1, ritmos_RIT_1 = ritmos(beta1, eigenvaluesHo_1, eigenstatesHo_1, H_1)
ritmos_WVO_N, ritmos_RIT_N = ritmos(betaN, eigenvaluesHo_N, eigenstatesHo_N, H_N)

# Arrays para las poblaciones del nivel fundamental de cada espín

rho00_WVO = np.zeros(N)
rho00_RIT = np.zeros(N)

# En cada colisión hay relmente dos: una por la izquierda y una por la derecha

for i in range(C_total):
    
    rho_WVO = qt.tensor(rho_u1, rho_s_WVO)
    rho_RIT = qt.tensor(rho_u1, rho_s_RIT)
    
    rho_WVO_1, rho_RIT_1 = evol(ritmos_WVO_1, ritmos_RIT_1, rho_WVO, rho_RIT)

    rho_s_WVO_1 = qt.ptrace(rho_WVO_1, N_s_1)

    rho_s_RIT_1 = qt.ptrace(rho_RIT_1, N_s_1)
    
    # Colisión por la derecha
    
    rho_WVO = qt.tensor(rho_s_WVO_1, rho_uN)
    rho_RIT = qt.tensor(rho_s_RIT_1, rho_uN)
    
    rho_WVO_N, rho_RIT_N = evol(ritmos_WVO_N, ritmos_RIT_N, rho_WVO, rho_RIT)
    
    rho_s_WVO = qt.ptrace(rho_WVO_N, N_s_N)
    
    rho_s_RIT = qt.ptrace(rho_RIT_N, N_s_N)
    
    for l in range(N):
        
        T_espines_WVO[l,i] = T_espin(rho_s_WVO,l)
        
        T_espines_RIT[l,i] = T_espin(rho_s_RIT,l)
        
for l in range(N):
    plt.figure(1)
    plt.plot(C_array, T_espines_WVO[l,:], 'o', label = f'Espín {l}')
    plt.figure(2)
    plt.plot(C_array, J_espines_WVO[l,:], 'o', label = f'Espín {l}')
    plt.figure(3)
    plt.plot(C_array, E_espines_WVO[l,:], 'o', label = f'Espín {l}')
 
plt.figure(1)
plt.xlabel('Número de colisión', fontsize = 15)
plt.ylabel('Temperatura', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=12)

plt.figure(2)
plt.xlabel('Número de colisión', fontsize = 15)
plt.ylabel('Flujo', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=12)

plt.figure(3)
plt.xlabel('Número de colisión', fontsize = 15)
plt.ylabel('Energía local', fontsize = 15)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=12)
        

# for i in range(N_total):
    
#     rho_WVO = qt.tensor(rho_u1, rho_s_WVO)
#     rho_RIT = qt.tensor(rho_u1, rho_s_RIT)
    
#     rho_WVO_1, rho_RIT_1 = evol(ritmos_WVO_1, ritmos_RIT_1, rho_WVO, rho_RIT)

#     rho_s_WVO_1 = qt.ptrace(rho_WVO_1, N_s_1)

#     rho_s_RIT_1 = qt.ptrace(rho_RIT_1, N_s_1)
    
#     # Colisión por la derecha
    
#     rho_WVO = qt.tensor(rho_s_WVO_1, rho_uN)
#     rho_RIT = qt.tensor(rho_s_RIT_1, rho_uN)
    
#     rho_WVO_N, rho_RIT_N = evol(ritmos_WVO_N, ritmos_RIT_N, rho_WVO, rho_RIT)
    
#     rho_s_WVO = qt.ptrace(rho_WVO_N, N_s_N)
    
#     rho_s_RIT = qt.ptrace(rho_RIT_N, N_s_N)
    
#     if i > terma: 
    
#         for l in range(N):
            
#             rho00_WVO[l] += qt.ptrace(rho_s_WVO,l)[0,0].real
        
#             rho00_RIT[l] += qt.ptrace(rho_s_RIT,l)[0,0].real
        
# rho00_WVO = rho00_WVO/prom
# rho00_RIT = rho00_RIT/prom

###############################################################################

# Función para escribir en un txt

def txt2(arr1, arr2, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        f.write("T\p00_can\n")  # Encabezados
        for valores in zip(arr1, arr2):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

def txt3(arr1, arr2, arr3, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        f.write("T\p00_WVO\p00_RIT\n")  # Encabezados
        for valores in zip(arr1, arr2, arr3):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

def txt5(arr1, arr2, arr3, arr4, arr5, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        f.write("i/N\T_WVO\T_RIT\J_WVO\J_RIT\n")  # Encabezados
        for valores in zip(arr1, arr2, arr3, arr4,arr5):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

# txt5(N_norm, T_WVO, T_RIT, J_WVO, J_RIT, 'cadena_N3_T0100_TN10.txt')

###############################################################################

fin = time.time()
print('Tiempo de ejecución en segundos')
print(fin-inicio)