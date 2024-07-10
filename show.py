import matplotlib.pyplot as plt
import pandas as pd

show = 'M' #@param ['M', 'E', 'C_v', 'X', 'u']
error = True #@param {type:"boolean"}

dic = {'M': ['mM','sM','Magnetization'],
       'E': ['mE','sE','Energy'],
       'C_v': ['C_v','sC_v', 'Specific heat'],
       'X': ['X', 'sX', 'Magnetic susceptibility'],
       'u': ['u','su','Binder cumulant']}

df = pd.read_csv('result.csv')
df['sC_v'] = df['C_v'] * 0
df['sX'] = df['X'] * 0
df['su'] = df['u'] * 0
for i in sorted(set(df['L'])):
  if error:
    plt.errorbar(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], yerr=df[df['L']==i][dic[show][1]], label=f'L={i}')
  else:
    plt.plot(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], label=f'L={i}')

plt.legend()
plt.xlabel('Temperature')
plt.ylabel(dic[show][2])
plt.show()