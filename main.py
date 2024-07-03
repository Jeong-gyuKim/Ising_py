from Izing_2d import metropolis, plot_magnetization_energy, get_lattice, get_kernel, get_energy

#ising model
L = 10
h = 0
init_up_rate = 1
Temp = 1
N=10**6

kernel, lattice = get_kernel(), get_lattice(L, init_up_rate)
energy = get_energy(lattice, kernel, h)

magnetizations, energies = [], []
for i, x in enumerate(metropolis(lattice, energy, Temp, L, h)):
    lattice, energy, spin = x
    magnetizations.append(spin)
    energies.append(energy)
    if i >= N:
        break

plot_magnetization_energy(magnetizations, energies)