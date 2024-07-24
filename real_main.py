from real_ising import get_avg_Q, get_energy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numba

@numba.njit(nopython=True, nogil=True)
def Q(lattice):
  return np.sum(lattice)#M

L_li = [1,2,3,4]  # Grid size
T_li = np.arange(0.1, 3, 0.1)  # Example temperature range

df = pd.DataFrame()
df['Temperature'] = T_li

for L in L_li:
  print(L)
  Q_li = get_avg_Q(L, T_li, Q)
  df[f'L={L}'] = Q_li

print(df.head())
df.to_csv("Q_li.csv", index=False, encoding='utf-8')

# Plot the results
for L in L_li:
    plt.plot(df['Temperature'], df[f'L={L}']/L**2, label=f'L={L}')
plt.xlabel('Temperature')
plt.ylabel('Average Q')
plt.title('Average Q vs Temperature for Various Lattice Sizes')
plt.legend()
plt.show()