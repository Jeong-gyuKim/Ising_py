import pandas as pd
import numpy as np
import datetime
import numba

@numba.njit(nopython=True, nogil=True)
def set_init_state(L, init_up_rate):
    lattice = np.where(np.random.random_sample((L,L))<init_up_rate,1,-1)
    magnetization = np.sum(lattice)
    energy = get_energy(lattice, L)
    return lattice, magnetization, energy

@numba.njit(nopython=True, nogil=True)
def get_energy(lattice, L):
     kernel = np.array([[False,  True, False], 
                        [ True, False,  True], 
                        [False,  True, False]])
     
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

     return (-lattice * convolution).sum()

@numba.njit(nopython=True, nogil=True)
def metropolis(L, init_up_rate, sweep, Temp):
     lattice, magnetization, energy = set_init_state(L, init_up_rate)
     N = sweep*L**2
     
     energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
     for i in range(sweep):
          for j in range(L**2):
               x = np.random.randint(0,L)
               y = np.random.randint(0,L)
               
               s_i = lattice[x, y]
               dE = deltaE(lattice, x, y, L)
               
               if dE <= 0 or np.exp(-dE/Temp) > np.random.random():
                    lattice[x,y] = - s_i
                    energy += dE * 2
                    magnetization += -2 * s_i
               
               energies[i*L**2 + j] = energy
               magnetizations[i*L**2 + j] = magnetization
          
     return physics(L, Temp, energies, magnetizations)

@numba.njit(nopython=True, nogil=True)
def deltaE(lattice, x, y, L):
    E_i = 0
    E_i += lattice[x-1 if x>0 else L-1,y]
    E_i += lattice[x+1 if L-1>x else 0,y]
    E_i += lattice[x,y-1 if y>0 else L-1]
    E_i += lattice[x,y+1 if L-1>y else 0]

    dE = 2 * lattice[x, y] * E_i
    return dE

@numba.njit(nopython=True, nogil=True)
def physics(L, Temp, energies, magnetizations):
    data = np.array([
               #np.mean(np.absolute(magnetizations)),#M
               np.mean(np.absolute(magnetizations))/L**2,#m
               #np.mean(energies)/2,#E
               np.mean(energies)/2/L**2,#e
               (np.mean(magnetizations**2)-np.mean(magnetizations)**2) / Temp / (L**2),#X
               (np.mean(energies**2)-np.mean(energies)**2) / (Temp**2) / (L**2),#c
               1-np.mean(magnetizations**4)/(3*np.mean(magnetizations**2)**2),#u
          ])
         
    return data

@numba.njit(nopython=True, nogil=True)
def multi_metropolis(Temp_range, L_range, sweep, n, init_up_rate):
     results = []
     print("READY!!")
     for L in L_range:
          for Temp in Temp_range:
               mean_data = np.zeros(shape=(5,n), dtype=np.float64)
               for j in range(n):
                    mean_data[:,j] = metropolis(L, init_up_rate, sweep, Temp)
               result = [L, Temp]
               for data in mean_data:
                    result.extend([np.mean(data), np.std(data)])
               results.append(result)
               #print(results[-1][0:2])
     return results

def calc_time(T_range, L_range, sweep, n):
     calc_time = 0
     for L in L_range:
          calc_time += round(8.7693834E-08 * len(T_range)*len(L_range)*L**2*sweep*n)
     print(f'약 {str(datetime.timedelta(seconds = calc_time))} 예상')
     print(f'시작: {datetime.datetime.now()}')
     print(f'종료: {str(datetime.datetime.now() + datetime.timedelta(seconds = calc_time))}')
     
def save(result, pth=''):
     print(f'종료: {datetime.datetime.now()}')
     df = pd.DataFrame(result, columns=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])
     if pth:
          df.to_csv(pth+"/result.csv", index=False, encoding='utf-8', header=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])
     else:
          df.to_csv("result.csv", index=False, encoding='utf-8', header=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])