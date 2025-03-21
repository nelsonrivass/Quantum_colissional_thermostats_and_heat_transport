# Master's Degree Final Project Code

This repository contains the Python code for my master's degree final project, organized into two main folders:

- **`/Qubit`**: Simulations of the thermalization of a single 1/2-spin (qubit) through repeated interactions with 1/2-spin units from a thermal reservoir. The scattering formalism is fully explained in [Jorge Tabanera et al. 2022, *New J. Phys.* 24 023018](https://iopscience.iop.org/article/10.1088/1367-2630/ac4923).
- **`/Cadenas`**: Programs for simulating the thermalization and heat transport in 1/2-spin chains coupled to one or two thermal reservoirs at their ends.

## **Contents of `/Qubit`**

Most scripts generate `.txt` files with simulation results, which can be visualized using the plotting scripts.

- **`qubits_plots.py`**: Plots the simulation results from the generated `.txt` files.
- **`qubit_ritmos.py`**: Computes transition probabilities using three methods:  
  - Exact method  
  - Wave-vector-operator model (WvO)  
  - Random interaction time model (RIT)  
  These probabilities are also calculated in the `_termalization_` scripts.
- **`distribucion.py`**: Plots the effusion distribution of reservoir units at different temperatures.
- **`qubit_termalization_kraus.py`**: Simulates the thermalization process using Kraus operators (CPTP maps).
- **`qubit_termalization_nokraus.py`**: Simulates thermalization without Kraus operators, considering only the evolution of the diagonal terms of the density matrix.
- **`qubit_termalization_nokraus_full.py`**: Extends the previous approach by including the exact method. However, this method is unstable at low energies due to the use of multiple inverse matrices. **I do not recommend** to use my implementation.
- **`qubit_termalization_nokraus_prom.py`**: Instead of using a stochastic approach, this script computes a weighted average of the transition probabilities based on the effusion distribution and simulates bombardment with these averaged units.

## **Contents of `/Cadenas`**

This folder contains scripts for simulating the thermalization and heat transport in 1/2-spin chains coupled to one or two thermal reservoirs at their extremities. We only use the weighted average approch from now on.

- **`cadena_plot.py`**: Plots the simulation results.
- **`cadena1_termalization.py`**: Simulates a spin chain where the outermost spin on the left is bombarded with units from a thermal bath at temperature **T**.
- **`cadena2_termalization.py`**: Simulates a spin chain where the outermost spins on both ends are bombarded with units from thermal baths at temperatures **T₁** and **T₂**, respectively.
