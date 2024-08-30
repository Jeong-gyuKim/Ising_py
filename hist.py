from ising import read, get_Z, get_avg, Cv, chi, dm_dbeta, binder, dbinder_dbeta
import numpy as np
import matplotlib.pyplot as plt

L = 12

data = read('data.pickle')
T1, energy, hist_E, hist_M, hist_M2, hist_M4 = data[L]
m = []
for i in range(len(T1)):
    m.append(sum(np.array(hist_E[i]) * np.array(hist_M[i])/L**2) / sum(hist_E[i]))
plt.plot(T1, m)
plt.show()
'''
Z1 = get_Z(data, L)

t = []
e = []
for T in np.arange(0.01,8, 0.01):
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(data, L, T, Z1)
    t.append(T)
    e.append(s_m)
plt.plot(t,e)
plt.show()'''