from ising import read_from_file
import numpy as np

data = read_from_file('data.pickle')
L = np.array(list(data.keys()))

def write_input(data):
    L = np.array(list(data.keys()))
    D = 0
    for l in L:
        T, energy, hist_E, hist_M, hist_M2, hist_M4 = data[l]
        for i in range(len(T)):
            with open("data/hist{}.csv".format(D), "w") as out:
                out.writelines("T,{},Nsite,{}\nenergy,hist_E,hist_M,hist_M2,hist_M4\n".format(T[i], l**2))
                for j in range(len(energy[i])):
                    out.writelines("{},{},{},{},{}\n".format(energy[i][j], hist_E[i][j], hist_M[i][j], hist_M2[i][j], hist_M4[i][j]))
            D+=1
write_input(data)