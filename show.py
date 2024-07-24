import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

show = 'm' #@param ['m', 'e', 'c', 'x', 'u']
error = True #@param {type:"boolean"}

dic = {'m': ['m','sm','Magnetization per spin'],
       'e': ['e','se','Energy per spin'],
       'c': ['c','sc', 'Specific heat per spin'],
       'x': ['x','sx', 'Magnetic susceptibility'],
       'u': ['u','su','Binder cumulant']}

df = pd.read_csv('result.csv')
for i in sorted(set(df['L'])):
  if error:
    plt.errorbar(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], yerr=df[df['L']==i][dic[show][1]], label=f'L={i}')
  else:
    plt.plot(df[df['L']==i]['Temp'], df[df['L']==i][dic[show][0]], label=f'L={i}')

Tc = 2/np.log(1+np.sqrt(2))
df2 = pd.DataFrame(np.array(df[df['L']==i]['Temp']), columns=['Temp'])
df2['m'] = df2['Temp'].apply(lambda x: (1-np.sinh(2/x)**-4)**(1/8) if x<Tc else 0)

df2['e'] = df2['Temp'].apply(lambda x: (0) if x<Tc else 1)
df2['c'] = df2['Temp'].apply(lambda x: (0) if x<Tc else 1)
df2['x'] = df2['Temp'].apply(lambda x: (0) if x<Tc else 1)
df2['u'] = df2['Temp'].apply(lambda x: (0) if x<Tc else 1)

#print(df2)
if show in ['m']:
  plt.plot(df2['Temp'], df2[show], label=f'L=inf')

plt.legend()
plt.xlabel('Temperature')
plt.ylabel(dic[show][2])
plt.title(show)
plt.show()