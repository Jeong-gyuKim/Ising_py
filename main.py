from Izing_2d import metropolis, plot_spin_energy, get_lattice, get_kernel, get_energy

#ising model
L = 10
init_up_rate = 0.25
Temp = 100
N=10**6

kernel, lattice = get_kernel(), get_lattice(L, init_up_rate)
energy = get_energy(lattice, kernel)

spins, energies = [], []
for i, x in enumerate(metropolis(lattice, energy, Temp, L)):
    lattice, energy, spin = x
    spins.append(spin)
    energies.append(energy)
    if i >= N:
        break

plot_spin_energy(spins, energies)