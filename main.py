from new_ising import multi_metropolis, calc_time
import numpy as np
import pandas as pd

init_L = 4#@param{type:"integer"}
final_L = 14#@param{type:"integer"}
step_L = 2#@param{type:"integer"}

init_Temp = 0.01#@param{type:"number"}
final_Temp = 8#@param{type:"number"}
step_Temp = 0.01#@param{type:"number"}

N = 10**7#@param{type:"integer"}
n=20#@param{type:"integer"}

T_range = np.arange(init_Temp,final_Temp,step_Temp)
L_range = np.arange(init_L,final_L,step_L)

calc_time(len(T_range), len(L_range), N, n)

result = multi_metropolis(T_range, L_range, N, n)

df = pd.DataFrame(result, columns=['L', 'Temp', 'mM', 'sM', 'mE', 'sE', 'M2', 'E2', 'M4', 'EM', 'EM2'])

df['C_v'] = (df['E2']-df['mE']**2)/(df['L']**2*df['Temp']**2)
df['X'] = (df['M2']-df['mM']**2)/(df['L']**2*df['Temp'])
df['u'] = 1-(df['M4']/(3*df['M2']**2))
df2 = df[['Temp', 'L', 'mM', 'sM', 'mE', 'sE', 'C_v', 'X', 'u']]
df2.to_csv("result.csv", index=False, encoding='utf-8', header=['Temp', 'L', 'mM', 'sM', 'mE', 'sE', 'C_v', 'X', 'u'])
