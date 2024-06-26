import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import numba

#Metropolis algorithm
@numba.njit("UniTuple(i8[:], 2)(i8[:,:], f8, f8, i8)", nopython=True, nogil=True)
def metropolis(lattice, energy, Temp, L):
    lattice = lattice.copy()
    magnetization = np.sum(lattice)/L**2
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
          magnetization = np.sum(lattice)/L**2
      yield lattice, energy, magnetization

def plot_magnetization_energy(magnetization, energies):
  fig, axes = plt.subplots(1, 2, figsize=(12,4))
  ax = axes[0]
  ax.plot(magnetization)#[N//2:])
  ax.set_xlabel('Algorithm Time Steps')
  ax.set_ylabel(r'Average Magnetization $\bar{m}$')
  ax.grid()
  ax = axes[1]
  ax.plot(energies)#[N//2:])
  ax.set_xlabel('Algorithm Time Steps')
  ax.set_ylabel(r'Energy $E$')
  ax.grid()
  fig.tight_layout()
  fig.suptitle(r'Evolution of Average Magnetization and Energy', y=1.07, size=18)
  plt.show()

def get_lattice(L, init_up_rate):
     return np.where(np.random.random([L,L])<init_up_rate,1,-1)

def get_kernel():
     return np.array([[False,  True, False], [ True, False,  True], [False,  True, False]])

def get_energy(lattice, kernel):
     return (-lattice * convolve(lattice, kernel, mode='constant', cval=0)).sum()