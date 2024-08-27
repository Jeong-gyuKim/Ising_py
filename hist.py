from ising import read
import numpy as np
import matplotlib.pyplot as plt

data = read('data.pickle')

L = 4
origin = 2.9
n=0

ori = data[L][origin][n]
ori_E = np.array(sorted(list(ori.keys())))
ori_cnt = np.array([ori[E][0] for E in ori_E])

T = []
E = []
for target in np.arange(0.01,8,0.01):
    Z = np.array([np.exp((1./origin - 1./target) * E) for E in ori_E])
    hist_cnt = ori_cnt * Z
    mean = sum(ori_E * hist_cnt) / sum(hist_cnt)

    T.append(target)
    E.append(mean/L**2)
    
plt.scatter(T,E, label='{}'.format(L))
plt.legend()
plt.show()