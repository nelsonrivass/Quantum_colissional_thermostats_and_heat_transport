# Master's Degree Final Project Code

This repository contains the Python code for my master's degree final project, organized into two main folders:

- **`/Qubit`**: Simulations of the thermalization of a single spin-1/2 system (qubit) through repeated interactions with spin-1/2 units from a thermal reservoir. The scattering formalism is fully explained in [Jorge Tabanera et al. 2022, *New J. Phys.* 24 023018](https://iopscience.iop.org/article/10.1088/1367-2630/ac4923).
- **`/Cadenas`**: Programs for simulating thermalization and heat transport in spin-1/2 chains coupled to one or two thermal reservoirs at their ends.

---

## Contents of `/Qubit`

Most scripts generate `.txt` files with simulation results, which can be visualized using the provided plotting scripts.

- **`qubits_plots.py`**: Visualizes the simulation results from the generated `.txt` files.
- **`qubit_ritmos.py`**: Computes transition probabilities using three different models:  
  - Exact method  
  - Wave-vector-operator model (WvO)  
  - Random interaction time model (RIT)  
  These probabilities are also calculated in the `_termalization_` scripts.
- **`distribucion.py`**: Plots the effusion distribution of reservoir units at various temperatures.
- **`qubit_termalization_kraus.py`**: Simulates the thermalization process using Kraus operators (CPTP maps).
- **`qubit_termalization_nokraus.py`**: Simulates thermalization without using Kraus operators, evolving only the diagonal terms of the density matrix.
- **`qubit_termalization_nokraus_full.py`**: Extends the previous approach by including the exact method. **Not recommended**, as the method is unstable at low energies due to the use of multiple matrix inversions.
- **`qubit_termalization_nokraus_prom.py`**: Simulates bombardment using a weighted average of transition probabilities based on the effusion distribution, instead of a stochastic approach.

---

## Contents of `/Cadenas`

This folder contains scripts for simulating thermalization and heat transport in spin-1/2 chains coupled to one or two thermal reservoirs at their ends. From this point on, only the weighted average approach is used.

- **`cadena_plot.py`**: Visualizes the simulation results.
- **`cadena1_termalization.py`**: Simulates a spin chain where the leftmost spin is bombarded by thermal units from a bath at temperature **T**.
- **`cadena2_termalization.py`**: Simulates a spin chain where both ends are bombarded by thermal units from baths at temperatures **T₁** and **T₂**, respectively.

---

For completeness, each folder includes the `.txt` files generated from all simulations, allowing readers to easily reproduce the plots and visualize the results without rerunning the simulations.
