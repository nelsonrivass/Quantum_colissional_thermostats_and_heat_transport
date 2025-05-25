#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:57:23 2025

@author: nelson
"""

import qutip as qt 
import numpy as np
import scipy.integrate as spi
from scipy.optimize import fsolve
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

# Parámetros de la cadena

ws = 1 # Campo magnético o energía de cada uno de los sitios
Jx = 0.1 # Este lo dejamos fijo
Jy = 0.05 # Si es igual a Jx entonces la cadena es  XX? si son distintos XY?. Si lo cambiamos es a 0.05
Jz = 0.05 # Si es 0 entonces la cadena es XX, si es igual a los anteriores XXX, si es distinto XXZ y si son todos distintos XYZ. Si lo cambiamos es a 0.05 a menos que sea XYZ entonces es 0.02

jx = 1  # Acoplos entre unidades y el extremo de la cadena. Usaremos siempre esta configuración porque ahora nos interesa la cadena
jy = 0

m = 0.1 
k = 1
L = 50

# Distribución de momentos de las partículas a una temperatura T

def u(beta, p):
    u = (beta*p/m) * np.exp(-(beta * p**2) / (2*m))
    return u

###############################################################################

# Definimos el problema. Espacios, hamiltonianos, estados iniciales, etc.

# Se trata de cadenas de N espines S = 1/2

s = 1/2
N = 5

# La dimensión del espacio de Hilbert de cada uno de estos espines es

dim = 2*s + 1

# Por tannto, la dimensión del sistema será

dim_s = int(dim**N)

# Las unidades con las que bombardearemos el sistema serán qubits igual que antes

dim_u = dim

# La dimensión total en cada colisión será

dim_total = int(dim_u * dim_s)

# Hamiltonianos individuales. 

def h(i):
    
    h = ws * (qt.tensor([I2] * i + [sz] + [I2] * (N - i - 1)))
        
    return h

def hh(i):
    
    if i == -1 or i == N-1 :
        
        hh = 0
    
    else:
    
        hh = 0
        hh += Jx * (qt.tensor([I2]* i + [sx,sx] + [I2] * (N - i - 2)))
        hh += Jy * (qt.tensor([I2]* i + [sy,sy] + [I2] * (N - i - 2)))
        hh += Jz * (qt.tensor([I2]* i + [sz,sz] + [I2] * (N - i - 2)))
    
    return hh

def hs(i):
    
    hs = 0
    hs += (1/2)*hh(i-1)
    hs += h(i)
    hs += (1/2)*hh(i)
    
    return hs
             
def hb(i):      

    hb = 0
    hb += (1/2)*h(i)
    hb += hh(i)
    hb += (1/2)*h(i+1)
    
    return hb

# Hamiltoniano completo

def Hs_func(N):
    
    # Inicializar el Hamiltoniano en cero
    Hs = 0 
    
    # Construcción del Hamiltoniano sumando los términos de acoplamiento entre vecinos
    for j in range(N - 1): 
        Hs += Jx * qt.tensor([I2] * j + [sx, sx] + [I2] * (N - j - 2))
        Hs += Jy * qt.tensor([I2] * j + [sy, sy] + [I2] * (N - j - 2))
        Hs += Jz * qt.tensor([I2] * j + [sz, sz] + [I2] * (N - j - 2))

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
    Hus_N += jx * qt.tensor([I2] * (N-1) + [sx,sx])
    Hus_N += jy * qt.tensor([I2] * (N-1) + [sy,sy])
    
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

# Matrices cambio de base de Ho1 y HoN

Uo1 = qt.Qobj([state.full().flatten() for state in eigenstatesHo_1], dims = [[dim for i in range(N+1)],[dim for i in range(N+1)]])

UoN = qt.Qobj([state.full().flatten() for state in eigenstatesHo_N], dims = [[dim for i in range(N+1)],[dim for i in range(N+1)]])

def Uo1t(rho):
    return Uo1 * rho * Uo1.dag()

def Uo1tinv(rho):
    return Uo1.dag() * rho * Uo1

def UoNt(rho):
    return UoN * rho * UoN.dag()

def UoNtinv(rho):
    return UoN.dag() * rho * UoN

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
    
    return reservoirstate

# Estado térmico del sistema que depende de la temperatura

def systemstate(beta):
    
    systemstate = 0
    
    for i in range(len(eigenvaluesHs)):
        state = eigenstatesHs[i]
        systemstate += np.exp(-beta*eigenvaluesHs[i]) * state * state.dag()
        
    systemstate = (1/Z_s(beta))*systemstate
    
    return systemstate

# Energía máxima del problema necesaria par los termostátos

e_max = np.max([eigenvaluesH_1, eigenvaluesHo_1]) # Es igual si utilizamos H_N y Ho_N

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

# Temperatura de cada espín 

def T_espin(rho, i):
    
    rho_i = qt.ptrace(rho, i)
    
    T = (-2*ws) / (np.log(rho_i[1,1].real/rho_i[0,0].real))
    
    T = np.abs(T)
    
    return T

def T_new(rho, i, guess):
    
    guess = 1/guess
    
    goal = (rho*hs(i)).tr().real
    
    def f(beta):
        
        return (systemstate(beta)*hs(i)).tr().real
    
    def g(beta):
        
        return f(beta) - goal
    
    sol = fsolve(g, guess)
    
    return 1/sol[0]

###############################################################################

# Ahora simulamos. Definimos las temperaturas de cada termostáto. El 1 es el caliente 
# y el N+1 el frío.

T1 = 15
beta1 = 1/T1
TN = 10
betaN = 1/TN

# Matrices densidad de cada uno de los reservorios

rho_u1 = reservoirstate(beta1)
rho_uN = reservoirstate(betaN)

rho_s_WVO = rho_s
rho_s_RIT = rho_s

# Colisiones de termalización y de promediado

C_total = int(3e2)

# Array con número de colisiones

C_array = np.array([i+1 for i in range(C_total)])

# Matriz con las temperaturas, energías locales y flujos

T_espines_WVO = np.zeros((C_total,N))
T_espines_RIT = np.zeros((C_total,N))

Q_1_WVO = np.zeros(C_total)
Q_1_RIT = np.zeros(C_total)
Q_N_WVO = np.zeros(C_total)
Q_N_RIT = np.zeros(C_total)

perfil_WVO = np.zeros(N)
perfil_RIT = np.zeros(N)

# Calculamos los ritmos promediados por las colisiones calientes y frías

ritmos_WVO_1, ritmos_RIT_1 = ritmos(beta1, eigenvaluesHo_1, eigenstatesHo_1, H_1)
ritmos_WVO_N, ritmos_RIT_N = ritmos(betaN, eigenvaluesHo_N, eigenstatesHo_N, H_N)

# Matrices y energías iniciales

rho_WVO = Uo1t(qt.tensor(rho_u1, rho_s))
rho_RIT = Uo1t(qt.tensor(rho_u1, rho_s))

E0_WVO = (rho_s_WVO*Hs).tr()
E0_RIT = (rho_s_RIT*Hs).tr()


# En cada colisión hay relmente dos: una por la izquierda y una por la derecha

for i in range(C_total):
    
    # Colisión por la izquierda (caliente)
    
    rho_WVO_1, rho_RIT_1 = evol(ritmos_WVO_1, ritmos_RIT_1, rho_WVO, rho_RIT)
    
    rho_WVO_1 = Uo1tinv(rho_WVO_1)

    rho_RIT_1 = Uo1tinv(rho_RIT_1)
    
    rho_s_WVO_1 = qt.ptrace(rho_WVO_1, N_s_1)

    rho_s_RIT_1 = qt.ptrace(rho_RIT_1, N_s_1)
    
    E1_WVO = (rho_s_WVO_1*Hs).tr()
    
    E1_RIT = (rho_s_RIT_1*Hs).tr()
    
    Q_1_WVO[i] = E1_WVO - E0_WVO
    
    Q_1_RIT[i] = E1_RIT - E0_RIT
    
    # Colisión por la derecha (frío)
    
    rho_WVO = UoNt(qt.tensor(rho_s_WVO_1, rho_uN))
    rho_RIT = UoNt(qt.tensor(rho_s_RIT_1, rho_uN))
    
    rho_WVO_N, rho_RIT_N = evol(ritmos_WVO_N, ritmos_RIT_N, rho_WVO, rho_RIT)
    
    rho_WVO_N = UoNtinv(rho_WVO_N)

    rho_RIT_N = UoNtinv(rho_RIT_N)
    
    rho_s_WVO = qt.ptrace(rho_WVO_N, N_s_N)
    
    rho_s_RIT = qt.ptrace(rho_RIT_N, N_s_N)
    
    EN_WVO = (rho_s_WVO*Hs).tr()
    
    EN_RIT = (rho_s_RIT*Hs).tr()
        
    Q_N_WVO[i] = EN_WVO - E1_WVO
    
    Q_N_RIT[i] = EN_RIT - E1_RIT
    
    E0_WVO = EN_WVO
    
    E0_RIT = EN_RIT
    
    rho_WVO = Uo1t(qt.tensor(rho_u1, rho_s_WVO))
    
    rho_RIT = Uo1t(qt.tensor(rho_u1, rho_s_RIT))
    
    for l in range(N):
        
        T_espines_WVO[i,l] = T_new(rho_s_WVO,l, 1)
        T_espines_RIT[i,l] = T_new(rho_s_RIT,l, 1)
    
for l in range(N):
    
    perfil_WVO[l] = T_espines_WVO[-1,l]
    
    perfil_RIT[l] = T_espines_RIT[-1,l]
    

###############################################################################

# Función para escribir en un txt

def txt1(arr1, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        for valores in zip(arr1):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

def txt2(arr1, arr2, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        for valores in zip(arr1, arr2):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

def txt3(arr1, arr2, arr3, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        for valores in zip(arr1, arr2, arr3):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

def txt5(arr1, arr2, arr3, arr4, arr5, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        for valores in zip(arr1, arr2, arr3, arr4,arr5):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")
    
def matrix(matrix, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

matrix(T_espines_WVO, 'T5T10_T_WVO_N5_XXZ_3e2.txt')
txt1(C_array, 'colisiones_3e2.txt')
txt3(N_norm, perfil_WVO, perfil_RIT, 'T15T10_perfil_N5_XXZ_3e2.txt')
txt5(C_array, Q_1_WVO, Q_1_RIT, Q_N_WVO, Q_N_RIT, 'T15T10_Q_N5_XXZ_3e2.txt')

###############################################################################

fin = time.time()
print('Tiempo de ejecución en segundos')
print(fin-inicio)