import matplotlib.pyplot as plt
import pandas as pd

show = 'M' #@param ['M', 'E', 'C_v', 'X', 'u']
error = False #@param {type:"boolean"}

dic = {'M': ['mM','sM','Magnetization'],
       'E': ['mE','sE','Energy'],
       'C_v': ['C_v','sC_v', 'Specific heat'],
       'X': ['X', 'sX', 'Magnetic susceptibility'],
       'u': ['u','su','Binder cumulant']}

df2 = pd.read_csv('result.csv')
for i in sorted(set(df2['L'])):
  if error:
    plt.errorbar(df2[df2['L']==i]['Temp'], df2[df2['L']==i][dic[show][0]], yerr=df2[df2['L']==i][dic[show][1]], label=f'L={i}')
  else:
    plt.plot(df2[df2['L']==i]['Temp'], df2[df2['L']==i][dic[show][0]], label=f'L={i}')

plt.legend()
plt.xlabel('Temperature')
plt.ylabel(dic[show][2])
plt.show()