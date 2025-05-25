#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:41:29 2025

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
Jy = 0.1 # Si es igual a Jx entonces la cadena es  XX? si son distintos XY?. Si lo cambiamos es a 0.05
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

# Hamiltoniano de interacción 

def Hus_func(N):

    # Inicializar el Hamiltoniano en cero
    Hus = 0
    
    # Construcción del Hamiltoniano
    
    Hus += jx * qt.tensor([sx,sx] + [I2] * (N-1))
    Hus += jy * qt.tensor([sy,sy] + [I2] * (N-1))

    return Hus

# Ahora añadimos el hamiltoniano libre de las unidades Hu para formar Ho

I = qt.tensor(I2 for i in range(N))
Hs = Hs_func(N)
Hu = wu*sz

# Hamiltomianos libres

Ho = qt.tensor(I2,Hs) + qt.tensor(Hu,I)

# Hamiltonianos de interacción

Hus = Hus_func(N)

H = Ho + Hus

# Será util luego crear una lista para las trazas

N_s = np.array([i+1 for i in range(N)])

# Ahora hacemos un array pensado solo para graficar

N_norm = np.linspace(0, 1, N)

###############################################################################

# Diagonalizamos Ho y H y definimos el estado inicial del sistema y del reservorio

# Energías y autoestados de Ho y H

eigenvaluesHo, eigenstatesHo = Ho.eigenstates()
eigenvaluesH, eigenstatesH = H.eigenstates()

# Lo mismo con Hs y Hu

eigenvaluesHs, eigenstatesHs = Hs.eigenstates()
eigenvaluesHu, eigenstatesHu = Hu.eigenstates()

# Estado inicial del sistema. Estado fundamental puro

rho_s = eigenstatesHs[0] * eigenstatesHs[0].dag() 

# Matriz cambio de base de Ho

Uo = qt.Qobj([state.full().flatten() for state in eigenstatesHo], dims = [[dim for i in range(N+1)],[dim for i in range(N+1)]])


def Uot(rho):
    return Uo * rho * Uo.dag()

def Uotinv(rho):
    return Uo.dag() * rho * Uo


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

###############################################################################

# Evolución del operador densidad con una colisión

def ritmos(beta):
    
    ritmos_WVO = np.zeros((dim_total, dim_total))
    ritmos_RIT = np.zeros((dim_total, dim_total))
    
    for j_prima in range(dim_total):
        
        ef = eigenvaluesHo[j_prima]
        bra = eigenstatesHo[j_prima]
        
        for j in range(dim_total):
            
            ei = eigenvaluesHo[j]
            ket = eigenstatesHo[j]
            
            ritmos_WVO[j_prima,j], error = spi.quad(lambda p: WVO(p, ei, ef, ket, bra) * u(beta, p), 0, np.inf, limit = 50)
            ritmos_RIT[j_prima,j], error = spi.quad(lambda p: RIT(p, ei, ef, ket, bra) * u(beta, p), 0, np.inf, limit = 50)
            
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
    
# Energia local

def E_espin(rho, i):
    
    E = (rho*hs(i)).tr()
    
    return E.real

###############################################################################

# Estudiamos la población del nivel fundamental de la cadena en comparación con
# la población canónica del mismo, en similitud a lo hecho con el qubit.

# Población canónica

def p00_can(beta):
    
    p_00 = (1/Z_s(beta))*np.exp(-beta*eigenvaluesHs[0])
    
    return p_00

T = np.array([1, 2, 5, 10, 20, 50, 100, 500, 1000])

beta = 1 / (k*T)

# Para el canónico
T_can = np.linspace(1, 1000, 100000)
beta_can = 1 / (k*T_can)

terma = 1e5
prom = 1e5

C_total = int(terma + prom) # Número de colisiones de termalización y de promediado

p00_WVO = np.zeros(len(T))
p00_RIT = np.zeros(len(T))

for i in range(len(T)):
    
    rho_u = reservoirstate(beta[i])
    
    ritmos_WVO, ritmos_RIT = ritmos(beta[i])
    
    rho_WVO = Uot(qt.tensor(rho_u, rho_s))
    
    rho_RIT = Uot(qt.tensor(rho_u, rho_s))
    
    for j in range(C_total):
        
        rho_WVO, rho_RIT = evol(ritmos_WVO, ritmos_RIT, rho_WVO, rho_RIT)
        
        rho_WVO = Uotinv(rho_WVO)
        
        rho_RIT = Uotinv(rho_RIT)
        
        rho_WVO_s = qt.ptrace(rho_WVO, N_s) 
        
        rho_RIT_s = qt.ptrace(rho_RIT, N_s)
    
        rho_WVO = qt.tensor(rho_u, rho_WVO_s)
        
        rho_RIT = qt.tensor(rho_u, rho_RIT_s)
        
        rho_WVO = Uot(rho_WVO)
        
        rho_RIT = Uot(rho_RIT)
        
        if j > terma:
            
            # Población del nivel fundamental
    
            p00_WVO[i] += rho_WVO_s[dim_s-1,dim_s-1].real
    
            p00_RIT[i] += rho_RIT_s[dim_s-1,dim_s-1].real
    
    p00_WVO[i] = p00_WVO[i]/prom
    
    p00_RIT[i] = p00_RIT[i]/prom


###############################################################################

# Ahora estudiamos el proceso de termalización

T_ind = 20
beta_ind = 1/T_ind

# Estado del reservorio

rho_u = reservoirstate(beta_ind)

# Matrices densidad iniciales de la cadena

rho_s_WVO = rho_s
rho_s_RIT = rho_s

# Colisiones

C_total = int(3e2)

# Array con número de colisiones

C_array = np.array([i+1 for i in range(C_total)])

# Matriz con las temperatura

T_espines_WVO = np.zeros((C_total,N))
T_espines_RIT = np.zeros((C_total,N))

Q_WVO = np.zeros(C_total)
Q_RIT = np.zeros(C_total)

perfil_WVO = np.zeros(N)
perfil_RIT = np.zeros(N)

# Calculamos los ritmos promediados para la colisión

ritmos_WVO, ritmos_RIT = ritmos(beta_ind)

# Matrices  y energías iniciales

rho_WVO = Uot(qt.tensor(rho_u, rho_s))
rho_RIT = Uot(qt.tensor(rho_u, rho_s))

E1_WVO = (rho_s_WVO*Hs).tr()
E1_RIT = (rho_s_RIT*Hs).tr()

# Empezamos las colisiones

for i in range(C_total):
    
    rho_WVO, rho_RIT = evol(ritmos_WVO, ritmos_RIT, rho_WVO, rho_RIT)
    
    rho_WVO = Uotinv(rho_WVO)
    
    rho_RIT = Uotinv(rho_RIT)

    rho_s_WVO = qt.ptrace(rho_WVO, N_s) 
    
    rho_s_RIT = qt.ptrace(rho_RIT, N_s)
    
    E2_WVO = (rho_s_WVO*Hs).tr()
    
    E2_RIT = (rho_s_RIT*Hs).tr()
    
    Q_WVO[i] = E2_WVO - E1_WVO
    
    Q_RIT[i] = E2_RIT - E1_RIT
    
    E1_WVO = E2_WVO
    
    E1_RIT = E2_RIT
    
    rho_WVO = qt.tensor(rho_u, rho_s_WVO)
    
    rho_RIT = qt.tensor(rho_u, rho_s_RIT)
    
    rho_WVO = Uot(rho_WVO)
    
    rho_RIT = Uot(rho_RIT)
    
    for l in range(N):
        
        T_espines_WVO[i,l] = T_new(rho_s_WVO,l, T_ind)
        T_espines_RIT[i,l] = T_new(rho_s_RIT,l, T_ind)


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

# txt3(T, p00_WVO, p00_RIT,'termalizacion_N4_XYZ_1e5.txt') 
# txt2(T_can, p00_can(beta_can), 'termalizacion_N4_XYZ_can.txt')
matrix(T_espines_WVO, 'T20_T_WVO_N5_XXZ_3e2.txt')
txt1(C_array, 'colisiones_3e2.txt')
txt3(N_norm, perfil_WVO, perfil_RIT, 'T20_perfil_N5_XXZ_3e2.txt')

###############################################################################

fin = time.time()
print('Tiempo de ejecución en segundos')
print(fin-inicio)