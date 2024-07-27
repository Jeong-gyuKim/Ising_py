from ising import multi_metropolis, calc_time, save
import numpy as np

init_L = 4#@param{type:"integer"}
final_L = 14#@param{type:"integer"}
step_L = 2#@param{type:"integer"}

init_Temp = 1.5#@param{type:"number"}
final_Temp = 3#@param{type:"number"}
step_Temp = 0.1#@param{type:"number"}

N = 10**6#@param{type:"integer"}
n = 10#@param{type:"integer"}

T_range = np.arange(init_Temp,final_Temp,step_Temp)
L_range = np.arange(init_L,final_L,step_L)

calc_time(len(T_range), len(L_range), N, n)

result = multi_metropolis(T_range, L_range, N, n)
save(result)