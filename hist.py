from ising import read_from_file, get_Z, LinearRegression
from ising import E, M, Cv, chi, dm_dbeta, binder, dbinder_dbeta
from ising import NU, GAMMA, ALPHA
import numpy as np
import matplotlib.pyplot as plt

data = read_from_file('data1-3.pickle')
T1 = np.arange(1, 5, 0.01)

L_range = list(data.keys())

for L in L_range:
    Z1 = get_Z(data, L)
    print(sum(Z1),Z1)
    Q = [M(data, L, T, Z1) for T in T1]
    plt.plot(T1, Q, label="{}".format(L))
plt.show()