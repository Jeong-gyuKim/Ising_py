import numpy as np
import numba

#Metropolis algorithm
#@numba.njit("UniTuple(f8[:], 2, 2)(f8[:,:], f8, f8, i8, i8)", nopython=True, nogil=True)
def metropolis(lattice, energy, Temp, L=10):
    lattice = lattice.copy()
    spin = np.sum(lattice)/L**2
    while True:
      x = np.random.randint(0,L)
      y = np.random.randint(0,L)
      s_i = lattice[x, y]
      s_f = - s_i

      E_i = 0
      E_f = 0
      if x>0:
            E_i += -s_i*lattice[x-1,y]
            E_f += -s_f*lattice[x-1,y]
      if x<L-1:
            E_i += -s_i*lattice[x+1,y]
            E_f += -s_f*lattice[x+1,y]
      if y>0:
            E_i += -s_i*lattice[x,y-1]
            E_f += -s_f*lattice[x,y-1]
      if y<L-1:
            E_i += -s_i*lattice[x,y+1]
            E_f += -s_f*lattice[x,y+1]

      dE = E_f - E_i
      if dE < 0 or np.exp(-dE/Temp) > np.random.random():
          lattice[x,y] = s_f
          energy += dE
          spin = np.sum(lattice)/L**2
      yield lattice, energy, spin