import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import numba
import pickle

#초기 격자 생성
@numba.njit(nopython=True, nogil=True)
def set_init_state(L, init_up_rate):
    lattice = np.where(np.random.random_sample((L,L))<init_up_rate,1,-1)
    magnetization = np.sum(lattice)
    energy = get_energy(lattice, L)
    return lattice, magnetization, energy

#총 에너지 계산
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

     return (-lattice * convolution).sum() / 2

#x, y 좌표의 스핀 뒤집을 때의 에너지 변화
@numba.njit(nopython=True, nogil=True)
def deltaE(lattice, x, y, L):
    E_i = 0
    E_i += lattice[x-1 if x>0 else L-1,y]
    E_i += lattice[x+1 if L-1>x else 0,y]
    E_i += lattice[x,y-1 if y>0 else L-1]
    E_i += lattice[x,y+1 if L-1>y else 0]

    dE = 2 * lattice[x, y] * E_i
    return dE

#메트로폴리스 알고리즘 실행
@numba.njit(nopython=True, nogil=True)
def metropolis(L, init_up_rate, N, T):
     lattice, magnetization, energy = set_init_state(L, init_up_rate)
     
     energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
     for i in range(N):
          x = np.random.randint(0,L)
          y = np.random.randint(0,L)
          
          s_i = lattice[x, y]
          dE = deltaE(lattice, x, y, L)
          
          if dE <= 0 or np.exp(-dE/T) > np.random.random():
               lattice[x,y] = - s_i
               energy += dE
               magnetization += -2 * s_i
          
          energies[i] = energy
          magnetizations[i] = magnetization
     
     return energies, magnetizations

#울프 알고리즘 실행
@numba.njit(nopython=True, nogil=True)
def wolff(L, init_up_rate, N, T):
     lattice, magnetization, energy = set_init_state(L, init_up_rate)
     
     energies, magnetizations = np.zeros(shape=(N,), dtype=np.float64), np.zeros(shape=(N,), dtype=np.float64)
     for i in range(N):
          x = np.random.randint(0,L)
          y = np.random.randint(0,L)
          
          s_i = lattice[x, y]
          flip = [[x,y]]
          p = 1. - np.exp(-2 / T)
          
          f_old = [[x,y]]
          while len(f_old)>0:
               f_new = []
               for x,y in f_old:
                    nears = [[x-1 if x>0 else L-1,y], [x+1 if L-1>x else 0,y], [x,y-1 if y>0 else L-1], [x,y+1 if L-1>y else 0]]
                    for near in nears:
                         if lattice[near[0], near[1]] == s_i and near not in flip:
                              if np.random.rand() < p:
                                   f_new.append(near)
                                   flip.append(near)
               f_old = f_new
          
          for x, y in flip:
               dE = deltaE(lattice, x, y, L)
               lattice[x,y] = -s_i
               energy += dE
               magnetization += -2 * s_i
          
          energies[i] = energy
          magnetizations[i] = magnetization
     
     return energies, magnetizations

#초기의 수렴 전 데이터 제거(burn-in) 1-sigma면 수렴 판단
@numba.njit(nopython=True, nogil=True)
def burn_in(arr, sigma=1.):
     out = np.where(np.abs(arr - np.mean(arr)) < sigma * np.std(arr))[0]
     if len(out) ==0:
          out = 0
     else:
          out = out[0]
     return arr[out:]

#magnetizations, energies로 물리적 의미 찾기
@numba.njit(nopython=True, nogil=True)
def physics(L, T, energies, magnetizations):
     M = np.mean(np.absolute(magnetizations))
     M2 = np.mean(magnetizations**2)
     M4 = np.mean(magnetizations**4)
     E = np.mean(energies)
     E2 = np.mean((energies)**2)
     
     data = np.array([
               #M,#M
               M / L**2,#m
               #E,#E
               E / L**2,#e
               (M2 - M**2) / T / (L**2),#x
               (E2 - E**2) / (T**2) / (L**2),#c
               1 - (M4/(3*M2**2)) if M2!=0 else 0,#u
          ])
     return data

#여러 번 얻은 결과 통계
#@numba.njit(nopython=True, nogil=True)
def statistics(Temp_range, L_range, N, n, sigma, init_up_rate = 0.5, f=wolff):
     results = []
     for L in L_range:
          rate = init_up_rate
          for T in Temp_range:
               mean_data = np.zeros(shape=(5,n), dtype=np.float64)
               for j in range(n):
                    energies, magnetizations = f(L, rate, N, T)
                    energies, magnetizations = burn_in(energies), burn_in(magnetizations)
                    mean_data[:,j] = physics(L, T, energies, magnetizations)
               rate = np.mean(mean_data[0])#m
               rate = (rate + 1) / 2
               result = [L, T]
               for data in mean_data:
                    result.extend([np.mean(data), sigma*np.std(data)])
               results.append(result)
               print(L, T)
     return results


# 오차 범위 내에 있는지 확인하고 새로운 값을 계산하는 함수
def calc_new_values(result):
     df = pd.DataFrame(result, columns=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])
     grouped = df.groupby(['L', 'Temp'])
     new_data = []

     for name, group in grouped:
          new_row = {'L': name[0], 'Temp': name[1]}
          for col in ['m', 'e', 'x', 'c', 'u']:
               q1, q2 = group[col].values
               sq1, sq2 = group['s' + col].values
               if abs(q1 - q2) <= max(sq1, sq2):
                    new_row[col] = np.mean([q1, q2])
                    new_row['s' + col] = max(sq1, sq2)
               else:
                    new_row[col] = np.nan
                    new_row['s' + col] = np.nan
          new_data.append(new_row)

     return pd.DataFrame(new_data)

#대략적 시간 계산
def calc_time(N, n):
     calc_time = round(8.7693834E-08 * N * n)
     print(f'약 {str(datetime.timedelta(seconds = calc_time))} 예상')
     print(f'시작: {datetime.datetime.now()}')
     print(f'종료: {str(datetime.datetime.now() + datetime.timedelta(seconds = calc_time))}')
     
#데이터 저장
def save(result, pth=''):
     print(f'종료: {datetime.datetime.now()}')
     df = pd.DataFrame(result, columns=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])
     if pth:
          df.to_csv(pth+"/result.csv", index=False, encoding='utf-8', header=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])
     else:
          df.to_csv("result.csv", index=False, encoding='utf-8', header=['L', 'Temp', 'm', 'sm', 'e', 'se', 'x', 'sx', 'c', 'sc', 'u', 'su'])

#피클 쓰기         
def write(data, filename):
     with open(filename,'wb') as fw:
          pickle.dump(data, fw)

#피클 읽기
def read(filename):
     with open(filename,'rb') as fr:
          data = pickle.load(fr)
     return data

#데이터 보이기
def show(show, error = True, fig = False):
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
               #plt.errorbar(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], yerr=df[df['L']==i][dic[show][1]], label=f'L={i}', capsize=4)
               plt.errorbar(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], yerr=df[df['L']==i][dic[show][1]], alpha=.75, fmt=':', capsize=3, capthick=1, label=f'L={i}')
               plt.fill_between(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]] - df[df['L']==i][dic[show][1]], df[df['L']==i][dic[show][0]] + df[df['L']==i][dic[show][1]], alpha=.25)
          else:
               plt.plot(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], label=f'L={i}')
     plt.legend()
     plt.xlabel('Temperature')
     plt.ylabel(dic[show][2])
     plt.ylim(dic[show][3][0],dic[show][3][1])
     plt.title(show)
     if fig:
          plt.savefig('{}.png'.format(dic[show][2]))
     plt.show()
    
#몬테카를로 시간에 따른 energies, magnetization 보이기
def plot_energy_magnetization(energies, magnetization):
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