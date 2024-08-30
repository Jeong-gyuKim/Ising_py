import numpy as np
import numba
import pickle

# Initialize the lattice with spins based on initial up rate
@numba.njit(nopython=True, nogil=True)
def initialize_lattice(L: int, init_up_rate: float) -> tuple:
    lattice = np.where(np.random.random_sample((L, L)) < init_up_rate, 1, -1)
    magnetization = np.sum(lattice)
    energy = compute_total_energy(lattice, L)
    return lattice, magnetization, energy

# Compute the total energy of the lattice
@numba.njit(nopython=True, nogil=True)
def compute_total_energy(lattice: np.ndarray, L: int) -> float:
    kernel = np.array([[False,  True, False], 
                       [ True, False,  True], 
                       [False,  True, False]])

    padded = np.zeros((L + 2, L + 2), dtype=np.int32)
    padded[1:-1, 1:-1] = lattice

    # Manually wrap the borders to simulate periodic boundary conditions
    padded[:,0] = padded[:,-2]
    padded[:,-1] = padded[:,1]
    padded[0,:] = padded[-2,:]
    padded[-1,:] = padded[1,:]

    convolution = np.zeros((L, L), dtype=np.float64)

    for y in range(L):
        for x in range(L):
            convolution[y, x] = np.sum(padded[y:y + 3, x:x + 3] * kernel)

    return (-lattice * convolution).sum() / 2

# Calculate the energy change when flipping a spin
@numba.njit(nopython=True, nogil=True)
def calculate_energy_change(lattice: np.ndarray, x: int, y: int, L: int) -> float:
    E_i = (
        lattice[x - 1 if x > 0 else L - 1, y] +
        lattice[x + 1 if x < L - 1 else 0, y] +
        lattice[x, y - 1 if y > 0 else L - 1] +
        lattice[x, y + 1 if y < L - 1 else 0]
    )

    return 2 * lattice[x, y] * E_i

# Perform the Wolff algorithm to simulate the Ising model
@numba.njit(nopython=True, nogil=True)
def wolff_algorithm(L: int, init_up_rate: float, N: int, T: float) -> tuple:
    lattice, magnetization, energy = initialize_lattice(L, init_up_rate)
    energies = np.zeros(N, dtype=np.float64)
    magnetizations = np.zeros(N, dtype=np.float64)

    for i in range(N):
        x, y = np.random.randint(0, L, size=2)
        s_i = lattice[x, y]
        flip = [[x, y]]
        p = 1. - np.exp(-2 / T)
        f_old = [[x, y]]

        while f_old:
            f_new = []
            for x, y in f_old:
                neighbors = [
                    [x - 1 if x > 0 else L - 1, y], 
                    [x + 1 if x < L - 1 else 0, y], 
                    [x, y - 1 if y > 0 else L - 1], 
                    [x, y + 1 if y < L - 1 else 0]
                ]
                for near in neighbors:
                    if lattice[near[0], near[1]] == s_i and near not in flip:
                        if np.random.rand() < p:
                            f_new.append(near)
                            flip.append(near)
            f_old = f_new

        for x, y in flip:
            dE = calculate_energy_change(lattice, x, y, L)
            lattice[x, y] = -s_i
            energy += dE
            magnetization += -2 * s_i

        energies[i] = energy
        magnetizations[i] = magnetization

    return energies, magnetizations

# Remove initial non-equilibrium data (burn-in) based on 1-sigma convergence
@numba.njit(nopython=True, nogil=True)
def burn_in(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    mask = np.abs(data - np.mean(data)) < sigma * np.std(data)
    if np.any(mask):
        return data[np.argmax(mask):]
    return data

# Compute statistics across multiple simulations
def compute_statistics(temp_range: np.ndarray, L_range: np.ndarray, N: int, init_up_rate: float = 0.5, simulation_func=wolff_algorithm) -> dict:
    data = {}
    for L in L_range:
        D = []
        for T in temp_range:
            energies, magnetizations = simulation_func(L, init_up_rate, N, T)
            energy, hist_E, hist_M, hist_M2, hist_M4 = compute_histogram(energies, magnetizations)
            print(L, T)
            D.append([T, energy, hist_E, hist_M, hist_M2, hist_M4])
        
        # Use lists instead of arrays for data consistency
        data[L] = [list(x) for x in zip(*D)]
        
    return data

def compute_histogram(energies: np.ndarray, magnetizations: np.ndarray) -> tuple:
    energies, magnetizations = burn_in(energies), burn_in(magnetizations)
    l = min(len(energies), len(magnetizations))
    energies, magnetizations = energies[-l:], abs(magnetizations[-l:])

    histogram = {}
    for e, m in zip(energies, magnetizations):
        if e in histogram:
            histogram[e] = update_histogram(histogram[e], m)
        else:
            histogram[e] = update_histogram([0, 0, 0, 0], m)

    energy_levels = list(histogram.keys())
    histograms = list(histogram.values())
    hist_E, hist_M, hist_M2, hist_M4 = [np.array(x) for x in zip(*histograms)]

    return energy_levels, hist_E, hist_M, hist_M2, hist_M4

def update_histogram(current_hist: list, magnetization: float) -> list:
    M, M2, M4 = magnetization, magnetization**2, magnetization**4
    return [
        current_hist[0] + 1, 
        compute_average(current_hist[0], current_hist[1], M), 
        compute_average(current_hist[0], current_hist[2], M2), 
        compute_average(current_hist[0], current_hist[3], M4)
    ]


def compute_average(count: int, current_mean: float, new_value: float) -> float:
     return current_mean*(count/(count+1)) + new_value/(count+1)

# Serialize data to a file using pickle
def write_to_file(data: dict, filename: str) -> None:
    with open(filename, 'wb') as fw:
        pickle.dump(data, fw)

# Deserialize data from a file using pickle
def read_from_file(filename: str) -> dict:
    with open(filename, 'rb') as fr:
        return pickle.load(fr)

def get_Z(data, L):
    T1, energy, hist_E, _, _, _ = data[L]
    size_d = len(T1)
    size_e = np.array([len(e) for e in energy])
    
    Z1 = np.ones(size_d)
    Z2 = np.zeros(size_d)

    for _ in range(50):  # Maximum of 50 iterations for convergence
        for D in range(size_d):
            for i in range(size_d):
                for k in range(size_e[i]):
                    sZ = 0.
                    for j in range(size_d):
                        sZ += np.exp((1. / T1[D] - 1. / T1[j]) * energy[i][k]) / Z1[j]
                    Z2[D] += hist_E[i][k] / sZ

        tZ = np.sum(Z2)
        diff = 0.
        for D in range(size_d):
            Z2[D] /= tZ
            diff += abs(Z1[D] - Z2[D])
            Z1[D] = Z2[D]
        
        if diff < 0.0001:  # Convergence criterion
            break
            
    return Z1

def get_avg(data, L, T, Z1):
    T1, energy, hist_E, hist_M, hist_M2, hist_M4 = data[L]
    size_d = len(T1)
    size_e = np.array([len(e) for e in energy])
    Nsite = L**2

    # Initialize sums
    s_e = s_e2 = s_m = s_m2 = s_m4 = s_me = s_m2e = s_m4e = tZ = 0

    for D in range(size_d):
        for i in range(size_e[D]):
            e1 = energy[D][i] / Nsite
            m1 = hist_M[D][i] / Nsite
            m2 = hist_M2[D][i] / Nsite**2
            m4 = hist_M4[D][i] / Nsite**4
            Z = Z1[D] * hist_E[D][i] * np.exp(-energy[D][i] * (1. / T - 1. / T1[D]))
            
            s_e += e1 * Z
            s_e2 += e1 * e1 * Z
            s_m += m1 * Z
            s_m2 += m2 * Z
            s_m4 += m4 * Z
            s_me += m1 * e1 * Z
            s_m2e += m2 * e1 * Z
            s_m4e += m4 * e1 * Z
            tZ += Z

    # Normalize by partition function
    s_e /= tZ
    s_e2 /= tZ
    s_m /= tZ
    s_m2 /= tZ
    s_m4 /= tZ
    s_me /= tZ
    s_m2e /= tZ
    s_m4e /= tZ

    return s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e

def E(data, L, T, Z1):
    s_e, _, _, _, _, _, _, _ = get_avg(data, L, T, Z1)
    return s_e

def M(data, L, T, Z1):
    _, _, s_m, _, _, _, _, _ = get_avg(data, L, T, Z1)
    return s_m

def Cv(data, L, T, Z1):
    Nsite = L**2
    s_e, s_e2, _, _, _, _, _, _ = get_avg(data, L, T, Z1)
    return Nsite * (s_e2 - s_e * s_e) / (T * T)

def chi(data, L, T, Z1):
    Nsite = L**2
    _, _, s_m, s_m2, _, _, _, _ = get_avg(data, L, T, Z1)
    return Nsite * (s_m2 - s_m * s_m) / T

def dm_dbeta(data, L, T, Z1):
    Nsite = L**2
    s_e, _, s_m, _, _, s_me, _, _ = get_avg(data, L, T, Z1)
    return Nsite * (s_m * s_e - s_me)

def binder(data, L, T, Z1):
    _, _, _, s_m2, s_m4, _, _, _ = get_avg(data, L, T, Z1)
    return 1.0 - s_m4 / (s_m2 * s_m2 * 3.0)

def dbinder_dbeta(data, L, T, Z1):
    Nsite = L**2
    s_e, _, _, s_m2, s_m4, _, s_m2e, s_m4e = get_avg(data, L, T, Z1)
    return (Nsite * (s_m2 * (s_m4e + s_m4 * s_e) - 2.0 * s_m4 * s_m2e) /
            (3.0 * s_m2 * s_m2 * s_m2))
