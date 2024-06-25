import numpy as np

def metropolis(lattice, energy, Temp, L=10, dim=2):
    lattice = lattice.copy()
    spin = np.sum(lattice)/L**dim
    while True:
      coord = np.random.randint(L, size=dim)
      s_i = lattice[tuple(coord)]
      s_f = - s_i

      E_i = 0
      E_f = 0
      for d in range(dim):
        new_coords = list(coord)
        if new_coords[d]>0:
          real_coords = list(new_coords)
          real_coords[d] -= 1
          E_i += -s_i*lattice[tuple(real_coords)]
          E_f += -s_f*lattice[tuple(real_coords)]
        if new_coords[d]<L-1:
          real_coords = list(new_coords)
          real_coords[d] += 1
          E_i += -s_i*lattice[tuple(real_coords)]
          E_f += -s_f*lattice[tuple(real_coords)]

      dE = E_f - E_i
      if dE < 0 or np.exp(-dE/Temp) > np.random.random():
          lattice[tuple(coord)] = s_f
          energy += dE
          spin = np.sum(lattice)/L**dim
      yield lattice, energy, spin