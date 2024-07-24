import numpy as np
import numba

@numba.njit(nopython=True, nogil=True)
def get_energy(lattice):
  L = len(lattice)
  kernel = np.array([[False,  True, False], [ True, False,  True], [False,  True, False]])
  padded = np.zeros(shape=(L+2,L+2))
  padded[1:-1,1:-1] = lattice
  padded[:,0] = padded[:,-2]
  padded[:,-1] = padded[:,1]
  padded[0,:] = padded[-2,:]
  padded[-1,:] = padded[1,:]

  convolution = np.zeros(shape=(L,L))
  for y in range(0,L):
    for x in range(0,L):
      convolution[y,x] = np.sum(padded[y:y+3,x:x+3]*kernel)

  return -convolution.sum()

@numba.njit(nopython=True, nogil=True)
def product(arr, r):
  result = [] # Initialize a list to store the results
  if r == 1:
    for i in range(len(arr)):
      result.append([arr[i]]) # Append single-element lists
  else:
    for i in range(len(arr)):
      for sub in product(arr, r-1):
        result.append([arr[i]] + sub) # Append combined lists
  return result # Return the accumulated result list

@numba.njit(nopython=True, nogil=True)
def get_avg_Q(L, T_li, Q):
  Q_li = []
  lattices = np.array(product(np.array([1, -1]), L**2)).reshape(-1, L, L)
  for T in T_li:
    partition_function = 0
    Q_sum = 0
    
    for lattice in lattices:
        energy = get_energy(lattice)
        prob = np.exp(-energy / T)
        partition_function += prob
        Q_sum += Q(lattice) * prob
    
    average_Q = Q_sum / partition_function
    Q_li.append(average_Q)
  return Q_li