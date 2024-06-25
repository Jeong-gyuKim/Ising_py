from Izing import metropolis
import numpy as np
import matplotlib.pyplot as plt

#ising model
L = 10
dim = 2
init_up_rate = 0.25
Temp = 1

kernel = np.array([[False,  True, False],
                   [ True, False,  True],
                   [False,  True, False]])

spins, energies = [], []
N=1000000

lattice = np.where(np.random.random([L]*dim)<init_up_rate,1,-1)
energy = 0

for i, x in enumerate(metropolis(lattice, energy, Temp)):
    lattice, energy, spin = x
    spins.append(spin)
    energies.append(energy)
    if i >= N:
        break

fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins)#[N//2:])
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Average Spin $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(energies)#[N//2:])
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Energy $E$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Average Spin and Energy', y=1.07, size=18)
plt.show()