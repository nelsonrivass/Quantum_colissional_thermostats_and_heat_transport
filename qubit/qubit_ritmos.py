#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:09:39 2025

@author: nelson
"""

import matplotlib.pyplot as plt
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

wu = 1
ws = 1
jx = 1
jy = 0
m = 0.1
k = 1
L = 50

#  Matrices de Pauli

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
I = qt.qeye(2)

# Hamiltonianos

Ho = ws * qt.tensor(sz,I) + wu * qt.tensor(I,sz)
Hus = jx * qt.tensor(sx,sx) + jy * qt.tensor(sy,sy)
H = Ho + Hus

# Energías de Ho y H

eigenvaluesHo, eigenstatesHo = Ho.eigenstates()
eigenvaluesH, eigenstatesH = H.eigenstates()

# Energía máxima y estados de mínima y máxima energía

e_max = np.max([eigenvaluesH, eigenvaluesHo])

initial_state = eigenstatesHo[0]
final_state = eigenstatesHo[-1]

###############################################################################

# Empezamos a definir los termostatos: Exacto, WVO y RIT

#Función para proyectar en estados abiertos

def P(Op, Popen):
    return np.matmul(np.matmul(Popen, Op), Popen)

# Operadores numero de onda

def K(x,E):
    K =  (2*m*(E-H)).sqrtm()
    Kexpn = (-1j*x*K).expm()
    Kexpp = (1j*x*K).expm()
    return K, Kexpn, Kexpp

def Ko(x,E):
    Ko = (2*m*(E-Ho)).sqrtm()
    Koexpn = (-1j*x*Ko).expm()
    Koexpp = (1j*x*Ko).expm()
    return Ko, Koexpn, Koexpp

# Método exacto. Usamos numpy para el espacio suma directa

def K_np(x,E):
    K_np = K(x,E)[0].full()
    Kexpn_np = K(x,E)[1].full()
    Kexpp_np = K(x,E)[2].full()
    return K_np, Kexpn_np, Kexpp_np

def Ko_np(x,E):
    Ko_np = Ko(x,E)[0].full()
    Koexpn_np = Ko(x,E)[1].full()
    Koexpp_np = Ko(x,E)[2].full()
    return Ko_np, Koexpn_np, Koexpp_np

def Mm(x, E, K):
    A = K(x,E)[2]
    B = K(x,E)[1]
    C = np.matmul(K(x,E)[0],K(x,E)[2])
    D = -np.matmul(K(x,E)[0],K(x,E)[1])
    Mm = np.block([[A, B], [C, D]])
    return Mm

def M(x, E):
    prod1 = np.matmul(np.linalg.inv(Mm(x,E,Ko_np)),Mm(x,E,K_np))
    prod2 = np.matmul(np.linalg.inv(Mm(-x,E,K_np)),Mm(-x,E,Ko_np))
    M = np.matmul(prod1,prod2)
    return M

# Termostatos

def exact(x,Ep, ei, ef, ket, bra):
    
    E = Ep + ei
    
    if E >= ef:
        
        # Filtrar canales abiertos
        open_channels = [j for j, e in enumerate(eigenvaluesHo) if e <= E]
        Hopen = [eigenstatesHo[j] for j in open_channels]

        # Proyector sobre Hopen
        Popen = qt.Qobj(sum([ket * ket.dag() for ket in Hopen]))
        
        op = M(x,E)
        m00_matrix = op[:dim_total,:dim_total]
        m01_matrix = op[:dim_total,dim_total:]
        m10_matrix = op[dim_total:,:dim_total]
        m11_matrix = op[dim_total:,dim_total:]
        s00_matrix = - np.matmul(np.linalg.pinv(m11_matrix),m10_matrix)
        s10_matrix = m00_matrix - np.matmul(m01_matrix, np.matmul(np.linalg.pinv(m11_matrix), m10_matrix)) 
        s01_matrix = np.linalg.pinv(m11_matrix)
        s11_matrix = np.matmul(m01_matrix,np.linalg.pinv(m11_matrix))

        s00 = P(s00_matrix, Popen.full())
        s01 = P(s01_matrix, Popen.full())
        s10 = P(s10_matrix, Popen.full())
        s11 = P(s11_matrix, Popen.full())

        S = qt.Qobj(np.block([[s00, s01], [s10, s11]]))
        zero = np.zeros((dim_total, dim_total))
        
        Kop = P(Ko(x,E)[0].sqrtm().full(), Popen.full())
        Kom = P(np.linalg.inv(Ko(x,E)[0].sqrtm().full()), Popen.full())
        
        KKo_p = qt.Qobj(np.block([[Kop, zero], [zero, Kop]]))
        KKo_m = qt.Qobj(np.block([[Kom, zero], [zero, Kom]]))
        
        SS =  KKo_p * S * KKo_m
        
        t_matrix = qt.Qobj(SS[:dim_total,dim_total:])
        r_matrix = qt.Qobj(SS[:dim_total,:dim_total])
        
        t = t_matrix.matrix_element(bra,ket) 
        r = r_matrix.matrix_element(bra,ket)
        
        exact = (t * t.conjugate()).real + (r * r.conjugate()).real
    else:
        exact = 0
    return exact

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

def WVO(Ep, ei, ef, ket, bra):
    E = Ep + ei
    if E > e_max:
        ki = np.sqrt(2*m*(E-ei))
        kf = np.sqrt(2*m*(E-ef))
        op = (1j*L*K(0,E)[0]).expm()
        t = op.matrix_element(bra,ket) * np.exp(-1j*L*(ki+kf)/2)
        tt = (t * t.conjugate()).real
    else:
        t = (bra.dag() * ket).real
        tt = t
    return t, tt

###############################################################################

# Probabilidades de transición en función de la energía

ket = initial_state
ei = eigenvaluesHo[0]
bra = final_state
ef = eigenvaluesHo[-1]

space = int(1e2)
Ep_array = np.linspace(6,60,space)
WVO_array = np.zeros(space)
RIT_array = np.zeros(space)
exact_array = np.zeros(space)


for i in range(0,len(Ep_array)):
    WVO_array[i] = WVO(Ep_array[i], ei, ef, ket, bra)[1]
    RIT_array[i] = RIT(Ep_array[i], ei, ef, ket, bra)[1]
    try:
        exact_array[i] = exact(L/2, Ep_array[i], ei, ef, ket, bra)
    except:
        pass
  
# Probabilidades de transición para bajas energías

Ep_l_array = np.linspace(4,6.5,space)
WVO_l_array = np.zeros(space)
RIT_l_array = np.zeros(space)
exact_l_array = np.zeros(space)

for i in range(0,len(Ep_l_array)):
    WVO_l_array[i] = WVO(Ep_l_array[i], ei, ef, ket, bra)[1]
    RIT_l_array[i] = RIT(Ep_l_array[i], ei, ef, ket, bra)[1]
    try:
        exact_l_array[i] = exact(L/2, Ep_l_array[i], ei, ef, ket, bra)
    except:
        pass

###############################################################################

# Función para escribir en un tx

def txt4(arr1, arr2, arr3, arr4, nombre_archivo):
    with open(nombre_archivo, "w") as f:
        f.write("Ep\tWVO\tRIT\tExact\n")  # Encabezados
        for valores in zip(arr1, arr2, arr3, arr4):
            f.write("\t".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {nombre_archivo}")

txt4(Ep_array, WVO_array, RIT_array, exact_array, 'prob_qubit.txt')
txt4(Ep_l_array, WVO_l_array, RIT_l_array, exact_l_array, 'prob_qubit_lowE.txt')

###############################################################################

fin = time.time()
print('Tiempo de ejecución en segundos')
print(fin-inicio)
    