from ising import read_from_file, get_Z, get_avg, E, M, Cv, chi, dm_dbeta, binder, dbinder_dbeta
import numpy as np
import matplotlib.pyplot as plt

L = 12

data = read_from_file('data.pickle')

T1 = np.arange(0.01, 5, 0.01)
for L in data.keys():
    Z1 = get_Z(data, L)

    Q = []
    for T in T1:
        Q.append(binder(data, L, T, Z1))
    plt.scatter(T1,Q,label='{}'.format(L))

plt.legend()
plt.show()