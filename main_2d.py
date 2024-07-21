from Izing_2d import metropolis, plot_magnetization_energy, get_lattice, get_kernel, get_energy, multi_metropolis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#ising model
L = 5
h = 0
init_up_rate = 0.5
Temp = 1.5
N=10**6

kernel = get_kernel()
lattice = get_lattice(L, init_up_rate)
energy = get_energy(lattice, kernel, h)
magnetization = np.sum(lattice)

lattice, energies, magnetizations = metropolis(lattice, energy, magnetization, Temp, L, h, N)
print(abs(np.mean(magnetizations)), np.mean(energies))
plot_magnetization_energy(magnetizations, energies)
