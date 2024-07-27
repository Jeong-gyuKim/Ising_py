import pandas as pd
import numpy as np
import datetime
import numba

@numba.njit(nopython=True, nogil=True)
def set_init_state(L, init_up_rate, kernel):
    lattice = np.where(np.random.random_sample((L,L))<init_up_rate,1,-1)
    magnetization = np.sum(lattice)
    energy = get_energy(lattice, kernel, L)
    return lattice,magnetization,energy

@numba.njit(nopython=True, nogil=True)
def get_energy(lattice, kernel, L):
     padded = np.zeros(shape=(L+2,L+2))
     padded[1:-1,1:-1] = lattice
     padded[:,0] = padded[:,-2]
     padded[:,-1] = padded[:,1]
     padded[0,:] = padded[-2,:]
     padded[-1,:] = padded[1,:]

     convolution = np.zeros(shape=(L,L))
     for y in range(0,L):
          for x in range(0,L):
               convolution[y,x] = np.sum(padded[y:y+3,x:x+3]*kernel)

     return -convolution.sum()

@numba.njit(nopython=True, nogil=True)
def metropolis(L, init_up_rate, kernel, N, Temp):
     lattice, magnetization, energy = set_init_state(L, init_up_rate, kernel)
     
     energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
     cnt = 0
     for i in range(N):
          x = np.random.randint(0,L)
          y = np.random.randint(0,L)
          
          s_i = lattice[x, y]
          
          E_i = 0
          E_i -= lattice[x-1 if x>0 else L-1,y]
          E_i -= lattice[x+1 if L-1>x else 0,y]
          E_i -= lattice[x,y-1 if y>0 else L-1]
          E_i -= lattice[x,y+1 if L-1>y else 0]

          dE = -2 * s_i * E_i
          
          if dE <= 0 or np.exp(-dE/Temp) > np.random.random():
               cnt += 1
               lattice[x,y] = - s_i
               energy += dE
               magnetization += -2 * s_i
          
          energies[i] = energy
          magnetizations[i] = magnetization
          
     data = np.array([
               #np.mean(np.absolute(magnetizations)),#M
               np.mean(np.absolute(magnetizations))/L**2,#m
               np.mean(energies)/2,#E
               #np.mean(energies)/2/L**2,#e
               (1/Temp)/(L**2)*(np.mean(magnetizations**2)-np.mean(magnetizations)**2),#X
               (1/Temp)**2/L**2*(np.mean(energies**2)-np.mean(energies)**2),#c
               1-np.mean(magnetizations**4)/(3*np.mean(magnetizations**2)**2),#u
          ])
     #print(L,round(Temp,2),cnt/N)
     return data

@numba.njit(nopython=True, nogil=True)
def multi_metropolis(Temp_range, L_range, N, n):
     kernel = np.array([[False,  True, False], [ True, False,  True], [False,  True, False]])
     results = []
     print("READY!!")
     for L in L_range:
          init_up_rate = 1
          for Temp in Temp_range:
               mean_data = np.zeros(shape=(5,n), dtype=np.float64)
               for j in range(n):
                    mean_data[:,j] = metropolis(L, init_up_rate, kernel, N, Temp)
               result = [L, Temp]
               for data in mean_data:
                    result.extend([np.mean(data), np.std(data)])
               results.append(result)
               #print(results[-1][0:2])
     return results

def calc_time(T, L, N, n):
     calc_time = round(np.dot([1.75387668e-07, -2.96140896e-01], [T*L*N*n, 1]),2)/2
     print(f'약 {str(datetime.timedelta(seconds = calc_time))} 예상')
     print(f'시작: {datetime.datetime.now()}')
     print(f'종료: {str(datetime.datetime.now() + datetime.timedelta(seconds = calc_time))}')
     
def save(result):
     print(f'종료: {datetime.datetime.now()}')
     df = pd.DataFrame(result, columns=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])
     df.to_csv("result.csv", index=False, encoding='utf-8', header=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])