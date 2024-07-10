from new_ising import multi_metropolis
import numpy as np
import pandas as pd

result = multi_metropolis(np.arange(0.01,8,0.01), np.arange(4,14,2), 10**7, 20)

df = pd.DataFrame(result, columns=['L', 'Temp', 'mM', 'sM', 'mE', 'sE', 'M2', 'E2', 'M4', 'EM', 'EM2'])

df['C_v'] = (df['E2']-df['mE']**2)/(df['L']**2*df['Temp']**2)
df['X'] = (df['M2']-df['mM']**2)/(df['L']**2*df['Temp'])
df['u'] = 1-(df['M4']/(3*df['M2']**2))
df2 = df[['Temp', 'L', 'mM', 'sM', 'mE', 'sE', 'C_v', 'X', 'u']]
df2.to_csv("result.csv", index=False, encoding='utf-8', header=['Temp', 'L', 'mM', 'sM', 'mE', 'sE', 'C_v', 'X', 'u'])
