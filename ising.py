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
     """
     D#온도 범위
     E#에너지 범위
     energy[MAX_D][MAX_E]#에너지 준위 ex)MAX_D번째 온도에서 MAX_E번째 에너지 준위의 에너지 값
     hist_E[MAX_D][MAX_E]#에너지 개수 ex)MAX_D번째 온도에서 MAX_E번째 에너지 준위의 출현 수
     hist_M[MAX_D][MAX_E]#평균 M
     hist_M2[MAX_D][MAX_E]#평균 M^2
     hist_M4[MAX_D][MAX_E]#평균 M^4
     T1[MAX_D]#MAX_D번째 온도의 온도 값
     Z1[MAX_D]#온도에 따른 분배 함수
     """
     
     data = {}
     for L in L_range:
          rate = init_up_rate
          D = []
          for T in Temp_range:
               energies, magnetizations = f(L, rate, N, T)
               energy, hist_E, hist_M, hist_M2, hist_M4 = hist_dict(energies, magnetizations)
               rate = (np.mean(np.absolute(magnetizations))/L**2 + 1) / 2
               print(L, T)
               D.append([T, energy, hist_E, hist_M, hist_M2, hist_M4])
          data[L] = [(list(x)) for x in zip(*D)]
     return data

def hist_dict(energies, magnetizations):
     energies, magnetizations = burn_in(energies), burn_in(magnetizations)
     l = min(len(energies), len(magnetizations))
     energies, magnetizations = energies[-l:], abs(magnetizations[-l:])
     
     dict = {}
     for i, e in enumerate(energies):
          if dict.get(e):
               dict[e] = update_dict(dict[e], magnetizations[i])
          else:
               dict[e] = update_dict([0,0,0,0], magnetizations[i])
     
     energy = list(dict.keys())
     hist = list(dict.values())
     hist_E, hist_M, hist_M2, hist_M4 = [np.array(list(x)) for x in zip(*hist)]
     
     return energy, hist_E, hist_M, hist_M2, hist_M4
               
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

def get_Z(data, L):
     
     T1, energy, hist_E, hist_M, hist_M2, hist_M4 = data[L]
     size_d = len(T1)
     size_e = np.array([len(e) for e in energy])
     
     Z1 = np.ones(size_d)
     Z2 = np.zeros(size_d)
     for iter in range(50):
          for D in range(size_d):
               for i in range(size_d):
                    for k in range(size_e[i]):
                         sZ = 0.
                         for j in range(size_d):
                              sZ += np.exp((1./T1[D] - 1./T1[j]) * energy[i][k])/Z1[j]
                         Z2[D] += hist_E[i][k] / sZ
          tZ = sum(Z2)
          diff = 0.
          for D in range(size_d):
               Z2[D] = Z2[D]/tZ
               diff += abs(Z1[D] - Z1[D])
               Z1[D] = Z2[D]
          if diff<0.0001:
               break
     return Z1

def get_avg(data, L, T, Z1):
     
     T1, energy, hist_E, hist_M, hist_M2, hist_M4 = data[L]
     size_d = len(T1)
     size_e = np.array([len(e) for e in energy])
     Nsite = L**2

     s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e, tZ = [0 for _ in range(9)]
     for D in range(size_d):
          for i in range(size_e[D]):
               e1 = energy[D][i]/Nsite
               m1 = hist_M[D][i]
               m2 = hist_M2[D][i]
               m4 = hist_M4[D][i]
               Z = Z1[D] * hist_E[D][i]*np.exp(-energy[D][i]*(1./T - 1./T1[D]))
               
               s_e += e1*Z
               s_e2 += e1*e1*Z
               s_m += m1*Z
               s_m2 += m2*Z
               s_m4 += m4*Z
               s_me += m1*e1*Z
               s_m2e += m2*e1*Z
               s_m4e += m4*e1*Z
               tZ += Z
     s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = s_e/tZ, s_e2/tZ, s_m/tZ, s_m2/tZ, s_m4/tZ, s_me/tZ, s_m2e/tZ, s_m4e/tZ

     return s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e


def Cv(data, L, T, Z1):
    Nsite = L**2
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(data, L, T, Z1)
    return Nsite * (s_e2 - s_e * s_e) / (T * T)


def chi(data, L, T, Z1):
    Nsite = L**2
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(data, L, T, Z1)
    return Nsite * (s_m2 - s_m * s_m) / T


def dm_dbeta(data, L, T, Z1):
    Nsite = L**2
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(data, L, T, Z1)
    return Nsite * (s_m * s_e - s_me)


def binder(data, L, T, Z1):
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(data, L, T, Z1)
    return 1.0 - s_m4 / (s_m2 * s_m2 * 3.0)


def dbinder_dbeta(data, L, T, Z1):
    Nsite = L**2
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(data, L, T, Z1)
    return (Nsite * (s_m2 * (s_m4e + s_m4 * s_e) - 2.0 * s_m4 * s_m2e) /
            (3.0 * s_m2 * s_m2 * s_m2))