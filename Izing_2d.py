import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import numba

#Metropolis algorithm
@numba.njit(nopython=True, nogil=True)
def metropolis(lattice, energy, magnetization, Temp, L, h, N):
     lattice = lattice.copy()
     energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
     for i in range(N):
          x = np.random.randint(0,L)
          y = np.random.randint(0,L)
          
          s_i = lattice[x, y]
          
          E_i = -h
          E_i -= lattice[(x-1)%L,y]
          E_i -= lattice[(x+1)%L,y]
          E_i -= lattice[x,(y-1)%L]
          E_i -= lattice[x,(y+1)%L]

          dE = -2 * s_i * E_i
          
          if dE < 0 or np.exp(-dE/Temp) > np.random.random():
               lattice[x,y] = - s_i
               energy += dE
               magnetization += -2 * s_i
          energies[i] = energy
          magnetizations[i] = magnetization
     return lattice, energies/L**2, magnetizations/L**2

@numba.njit(nopython=True, nogil=True)
def multi_metropolis(Temp_range, L_range, N, n):
     kernel = np.array([[False,  True, False], [ True, False,  True], [False,  True, False]])
     results = []
     for L in L_range:
          init_up_rate = 1
          for Temp in Temp_range:
               mean_data = np.zeros(shape=(7,n), dtype=np.float64)
               for j in range(n):
                    lattice = np.where(np.random.random_sample((L,L))<init_up_rate,1,-1)
                    magnetization = np.sum(lattice)
                    
                    padded = np.zeros(shape=(L+2,L+2))
                    padded[1:-1,1:-1] = lattice
                    padded[:,0] = padded[:,-2]
                    padded[:,-1] = padded[:,1]
                    padded[0,:] = padded[-2,:]
                    padded[-1,:] = padded[1,:]
                    
                    result = np.zeros(shape=(L,L))
                    for y in range(0,L):
                         for x in range(0,L):
                              result[y,x] = np.sum(padded[y:y+3,x:x+3]*kernel)
                                        
                    energy = result.sum()/2
                    
                    energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
                    for i in range(N):
                         x = np.random.randint(0,L)
                         y = np.random.randint(0,L)
                         
                         s_i = lattice[x, y]
                         
                         E_i = 0
                         E_i -= lattice[(x-1)%L,y]
                         E_i -= lattice[(x+1)%L,y]
                         E_i -= lattice[x,(y-1)%L]
                         E_i -= lattice[x,(y+1)%L]

                         dE = -2 * s_i * E_i
                         
                         if dE < 0 or np.exp(-dE/Temp) > np.random.random():
                              lattice[x,y] = - s_i
                              energy += dE
                              magnetization += -2 * s_i
                         energies[i] = energy
                         magnetizations[i] = magnetization
                    energies /= L**2
                    magnetizations /= L**2
               
                    mean_data[:,j] = np.array([
                         abs(np.mean(magnetizations)),
                         np.mean(energies),
                         np.mean(magnetizations**2),
                         np.mean(energies**2),
                         np.mean(magnetizations**4),
                         np.mean(energies * magnetizations),
                         np.mean(energies * magnetizations**2)
                    ])
               results.append([L, Temp, np.mean(mean_data[0]), np.std(mean_data[0]), np.mean(mean_data[1]), np.std(mean_data[1]), np.mean(mean_data[2]), np.mean(mean_data[3]), np.mean(mean_data[4]), np.mean(mean_data[5]), np.mean(mean_data[6])])
     print("L, T, m, sm, e, se, m2, e2, m4, em, em2")
     return results

def plot_magnetization_energy(magnetization, energies):
     fig, axes = plt.subplots(1, 2, figsize=(12,4))
     ax = axes[0]
     ax.plot(magnetization)
     ax.set_xlabel('Algorithm Time Steps')
     ax.set_ylabel(r'Average Magnetization $\bar{m}$')
     ax.set_ylim(-1.1,1.1)
     ax.grid()
     ax = axes[1]
     ax.plot(energies)
     ax.set_xlabel('Algorithm Time Steps')
     ax.set_ylabel(r'Energy $E$')
     ax.set_ylim(-2.1,2.1)
     ax.grid()
     fig.tight_layout()
     fig.suptitle(r'Evolution of Average Magnetization and Energy', y=1.07, size=18)
     plt.show()

def get_lattice(L, init_up_rate):
     return np.where(np.random.random([L,L])<init_up_rate,1,-1)

def get_kernel():
     return np.array([[False,  True, False], [ True, False,  True], [False,  True, False]])

def get_energy(lattice, kernel, h):
     return (-lattice * convolve(lattice, kernel, mode='wrap')).sum()/2 - h * (lattice).sum()