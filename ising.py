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

#여러 번 얻은 결과 통계
#@numba.njit(nopython=True, nogil=True)
def statistics(Temp_range, L_range, N, n, init_up_rate = 0.5, f=wolff):
     dict_L = {}#
     for L in L_range:
          rate = init_up_rate
          dict_T = {}#
          for T in Temp_range:
               dict_n = {}#
               magnetizations_data = np.zeros(n, dtype=np.float64)
               for j in range(n):
                    energies, magnetizations = f(L, rate, N, T)
                    dict_E = hist_dict(energies, magnetizations)#
                    magnetizations_data[j] = np.mean(np.absolute(magnetizations))/L**2
                    dict_n[j]=dict_E
               rate = np.mean(magnetizations_data)
               rate = (rate + 1) / 2
               print(L, T)
               dict_T[T]=dict_n
          dict_L[L]=dict_T
     return dict_L

def hist_dict(energies, magnetizations):
     energies, magnetizations = burn_in(energies), burn_in(magnetizations)
     l = min(len(energies), len(magnetizations))
     energies, magnetizations = energies[-l:], magnetizations[-l:]
     
     dict_E = {}
     for i, e in enumerate(energies):
          if dict_E.get(e):
               dict_E[e] = update_dict(dict_E[e], magnetizations[i])
          else:
               dict_E[e] = update_dict([0,0,0,0], magnetizations[i])
     return dict_E
               
def update_dict(li, magnetization):
     M, M2, M4 = magnetization, magnetization**2, magnetization**4
     return [li[0]+1, avg(li[0],li[1],M), avg(li[0],li[2],M2), avg(li[0],li[3],M4)]

def avg(cnt, mean, n):
     return mean*(cnt/(cnt+1)) + n/(cnt+1)

#피클 쓰기         
def write(data, filename):
     with open(filename,'wb') as fw:
          pickle.dump(data, fw)

#피클 읽기
def read(filename):
     with open(filename,'rb') as fr:
          data = pickle.load(fr)
     return data