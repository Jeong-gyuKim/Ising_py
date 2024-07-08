from Izing_2d import metropolis, plot_magnetization_energy, get_lattice, get_kernel, get_energy, multi_metropolis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
#ising model
L = 10
h = 0
init_up_rate = 1
Temp = 2.2
N=10**7
kernel = get_kernel()

lattice = get_lattice(L, init_up_rate)
energy = get_energy(lattice, kernel, h)
magnetization = np.sum(lattice)

#lattice, energies, magnetizations = metropolis(lattice, energy, magnetization, Temp, L, h, N)
#print(abs(np.mean(magnetizations)), np.mean(energies))

#plot_magnetization_energy(magnetizations, energies)
'''

result = multi_metropolis(np.arange(0.01,3,0.01), np.arange(10,60,10), 10**6, 10)

df = pd.DataFrame(result, columns=['L', 'Temp', 'mM', 'sM', 'mE', 'sE', 'M2', 'E2', 'M4', 'EM', 'EM2'])

df['C_v'] = (df['E2']-df['mE']**2)/(df['L']**2*df['Temp']**2)
df['X'] = (df['M2']-df['mM']**2)/(df['L']**2*df['Temp'])
df['u'] = 1-(df['M4']/(3*df['M2']**2))
df2 = df[['Temp', 'L', 'mM', 'sM', 'mE', 'sE', 'C_v', 'X', 'u']]
df2.to_csv("result.csv", index=False, encoding='utf-8', header=['Temp', 'L', 'mM', 'sM', 'mE', 'sE', 'C_v', 'X', 'u'])