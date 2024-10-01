import numpy as np
import numba

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
    D = 0
    for L in L_range:
        for T in temp_range:
            energies, magnetizations = simulation_func(L, init_up_rate, N, T)
            energy, hist_E, hist_M, hist_M2, hist_M4 = compute_histogram(energies, magnetizations)
            
            with open("data/hist{}.csv".format(D), "w") as out:
                out.writelines("T,{},Nsite,{}\nenergy,hist_E,hist_M,hist_M2,hist_M4\n".format(T, L**2))
                for i in range(len(energy)):
                    out.writelines("{},{:11.8e},{:11.8e},{:11.8e},{:11.8e}\n".format(energy[i], hist_E[i], hist_M[i], hist_M2[i], hist_M4[i]))
            D += 1
    return 0


def compute_histogram(energies: np.ndarray, magnetizations: np.ndarray) -> tuple:
    energies, magnetizations = burn_in(energies), burn_in(magnetizations)
    l = min(len(energies), len(magnetizations))
    energies, magnetizations = energies[-l:], abs(magnetizations[-l:])

    histogram = {}
    for e, m in zip(energies, magnetizations):
        if e in histogram:
            histogram[e] = histogram[e] + [m]
        else:
            histogram[e] = [m]
    histogram = dict(sorted(histogram.items()))

    energy_levels = list(histogram.keys())
    hist_E, hist_M, hist_M2, hist_M4 = np.zeros_like(energy_levels), np.zeros_like(energy_levels), np.zeros_like(energy_levels), np.zeros_like(energy_levels)
    for i in range(len(energy_levels)):
        Ms = np.array(histogram[energy_levels[i]])
        hist_E[i], hist_M[i], hist_M2[i], hist_M4[i] = len(Ms), np.mean(Ms), np.mean(Ms**2), np.mean(Ms**4)
    return energy_levels, hist_E, hist_M, hist_M2, hist_M4

def LinearRegression(x,y):
    x, y = np.array(x), np.array(y)
    a = sum((x-np.mean(x))*(y-np.mean(y)))/sum((x-np.mean(x))**2)
    b = np.mean(y) - a * np.mean(x)
    return a, b