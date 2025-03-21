Hi!
This is the python code I wrote for my master's degree final project

On `/Qubit` you'll find several `.py` files.  The goal is to perform the termalization of a single 1/2-spin (qubit) through its bombardment with 1/2-spin units coming from a thermal reservoir. The scattering formalism is fully explained on the article: [Jorge Tabanera et al 2022 New J. Phys. 24 023018](https://iopscience.iop.org/article/10.1088/1367-2630/ac4923)

- `qubits_plots.py`: most of the `.py` files create `.txt` with the results of the simulations. This file plot those .txt.
- `qubit.ritmos.py`: calculation of the transition probabilities for the three methods: exact, wave-vector-operator model (WvO) and random interaction time model (RIT).
- `distribucion.py`: plots the effusion distribution of the units coming from the resevoirs at different temperatures
- `qubit_termalization_kraus`: termalization process perfomed through Kraus operators (CPTP map)
- `qubit_termalization_nokraus`: termalization process without Kraus operators. It only takes into account the evolution of the diagonal terms of the density matrix.
- `qubit_termalization_nokraus_full`: the same file as before but it includes the exact method. This method is quite unstable at low energies due to the several inverse matrices needed. I do not recommned using my code to do this.
- 'qubit_termalization_nokraus_prom`: we now leave behind the sthocastic process and we perform a weighted average of the transition probabilities using the effusion distribution and bombard the system with this 'averaged' units.

On `/Cadenas`
