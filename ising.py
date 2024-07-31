import matplotlib.pyplot as plt
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
def deltaE(lattice, x, y, L):
    E_i = 0
    E_i += lattice[x-1 if x>0 else L-1,y]
    E_i += lattice[x+1 if L-1>x else 0,y]
    E_i += lattice[x,y-1 if y>0 else L-1]
    E_i += lattice[x,y+1 if L-1>y else 0]

    dE = 2 * lattice[x, y] * E_i
    return dE

@numba.njit(nopython=True, nogil=True)
def metropolis(L, init_up_rate, sweep, T):
     lattice, magnetization, energy = set_init_state(L, init_up_rate)
     N = sweep*L**2
     
     energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
     for i in range(sweep):
          for j in range(L**2):
               x = np.random.randint(0,L)
               y = np.random.randint(0,L)
               
               s_i = lattice[x, y]
               dE = deltaE(lattice, x, y, L)
               
               if dE <= 0 or np.exp(-dE/T) > np.random.random():
                    lattice[x,y] = - s_i
                    energy += dE * 2
                    magnetization += -2 * s_i
               
               energies[i*L**2 + j] = energy
               magnetizations[i*L**2 + j] = magnetization
     
     return energies, magnetizations

@numba.njit(nopython=True, nogil=True)
def sigma(arr):
    out = np.where(np.abs(arr - np.mean(arr)) < np.std(arr))[0]
    if len(out) ==0:
        out = 0
    else:
        out = out[0]
    return arr[out:]

@numba.njit(nopython=True, nogil=True)
def physics(L, T, energies, magnetizations):
     M = np.mean(np.absolute(magnetizations))
     M2 = np.mean(magnetizations**2)
     M4 = np.mean(magnetizations**4)
     E = np.mean(energies)/2
     E2 = np.mean((energies/2)**2)
     
     data = np.array([
               #M,#M
               M / L**2,#m
               #E,#E
               E / L**2,#e
               (M2 - M**2) / T / (L**2),#X
               (E2 - E**2) / (T**2) / (L**2),#c
               1 - (M4/(3*M2**2)) if M2!=0 else 0,#u
          ])
     return data

@numba.njit(nopython=True, nogil=True)
def multi_metropolis(Temp_range, L_range, sweep, n, init_up_rate):
     results = []
     print("READY!!")
     for L in L_range:
          for T in Temp_range:
               mean_data = np.zeros(shape=(5,n), dtype=np.float64)
               for j in range(n):
                    energies, magnetizations = metropolis(L, init_up_rate, sweep, T)
                    energies, magnetizations = sigma(energies), sigma(magnetizations)
                    mean_data[:,j] = physics(L, T, energies, magnetizations)
               result = [L, T]
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
          

def show(show, error = True):
    #show = 'm' #@param ['m', 'e', 'c', 'x', 'u']
    #error = True #@param {type:"boolean"}

    dic = {'m': ['m','sm','Magnetization per spin', [-0.1,1.1]],
       'e': ['e','se','Energy per spin', [-2.1,0.1]],
       'c': ['c','sc', 'Specific heat per spin', [-0.1,2.1]],
       'x': ['x','sx', 'Magnetic susceptibility', [-0.1,4.1]],
       'u': ['u','su','Binder cumulant', [-0.1,0.77]]}

    df = pd.read_csv('result.csv')
    for i in sorted(set(df['L'])):
      if error:
        plt.errorbar(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], yerr=df[df['L']==i][dic[show][1]], label=f'L={i}')
      else:
        plt.plot(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], label=f'L={i}')

#Tc = 2/np.log(1+np.sqrt(2))
#df2 = pd.DataFrame(np.array(df[df['L']==i]['Temp']), columns=['Temp'])
#df2['m'] = df2['Temp'].apply(lambda x: (1-np.sinh(2/x)**-4)**(1/8) if x<Tc else 0)

#print(df2)
#if show in ['m']:
  #plt.plot(df2['Temp'], df2[show], label=f'L=inf')

    plt.legend()
    plt.xlabel('Temperature')
    plt.ylabel(dic[show][2])
    plt.ylim(dic[show][3][0],dic[show][3][1])
    plt.title(show)
    plt.show()
    
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