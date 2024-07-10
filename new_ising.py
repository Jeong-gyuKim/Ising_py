import numpy as np
import numba

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
                                        
                    energy = -result.sum()
                    
                    energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
                    for i in range(N):
                         x = np.random.randint(0,L)
                         y = np.random.randint(0,L)
                         
                         s_i = lattice[x, y]
                         
                         E_i = 0
                         E_i -= lattice[x-1 if x>0 else x+L-1,y]
                         E_i -= lattice[x+1 if L-1>x else x-L+1,y]
                         E_i -= lattice[x,y-1 if y>0 else y+L-1]
                         E_i -= lattice[x,y+1 if L-1>y else y-L+1]

                         dE = -2 * s_i * E_i
                         
                         if dE < 0 or np.exp(-dE/Temp) > np.random.random():
                              lattice[x,y] = - s_i
                              energy += dE
                              magnetization += -2 * s_i
                         energies[i] = energy
                         magnetizations[i] = magnetization
                    
                    init_up_rate = (abs(np.mean(magnetizations))+1)/2
               
                    mean_data[:,j] = np.array([
                         abs(np.mean(magnetizations))/L**2,
                         np.mean(energies),
                         np.mean(magnetizations**2),
                         np.mean(energies**2),
                         np.mean(magnetizations**4),
                         np.mean(energies * magnetizations),
                         np.mean(energies * magnetizations**2)
                    ])
               results.append([L, Temp, np.mean(mean_data[0]), np.std(mean_data[0]), np.mean(mean_data[1]), np.std(mean_data[1]), np.mean(mean_data[2]), np.mean(mean_data[3]), np.mean(mean_data[4]), np.mean(mean_data[5]), np.mean(mean_data[6])])
               print(results[-1][0:2])
     #print("L, T, m, sm, e, se, m2, e2, m4, em, em2")
     return results