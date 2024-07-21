import matplotlib.pyplot as plt
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

plt.legend()
plt.xlabel('Temperature')
plt.ylabel(dic[show][2])
plt.show()