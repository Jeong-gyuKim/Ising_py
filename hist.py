from ising import read
import numpy as np
import matplotlib.pyplot as plt

data = read('data.pickle')
print('L:',np.array(list(data.keys())))

L = 4
print('T:',np.array(list(data[L].keys())))

origin = 2.3
target = 8

ori = data[L][origin][0]
#tar = data[L][target][0]

ori_E = np.array(sorted(list(ori.keys())))
#tar_E = np.array(sorted(list(tar.keys())))

ori_cnt = np.array([ori[E][0] for E in ori_E])
#tar_cnt = np.array([tar[E][0] for E in tar_E])

T = []
E = []
for target in np.arange(0.01,8,0.01):
    Z = np.array([np.exp((1./origin - 1./target) * E) for E in ori_E])
    hist_cnt = ori_cnt * Z
    mean = sum(ori_E * hist_cnt) / sum(hist_cnt)

    T.append(target)
    E.append(mean/L**2)
plt.scatter(T,E, label='{}'.format(L))

#plt.bar(ori_E, ori_cnt/max(ori_cnt), 2, label='ori')
#plt.bar(ori_E, hist_cnt/max(hist_cnt), 1.5, label='hist')
#plt.bar(tar_E, tar_cnt/max(tar_cnt), 1, label='tar')

#print('ori', sum(ori_E * ori_cnt) / sum(ori_cnt))
#print('hist', sum(ori_E * hist_cnt) / sum(hist_cnt))
#print('tar', sum(tar_E * tar_cnt) / sum(tar_cnt))

plt.legend()
plt.show()