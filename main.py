from ising import multi_metropolis, calc_time, save
import numpy as np

init_L = 4#@param{type:"integer"}
final_L = 14#@param{type:"integer"}
step_L = 2#@param{type:"integer"}

init_Temp = 0.1#@param{type:"number"}
final_Temp = 8#@param{type:"number"}
step_Temp = 0.1#@param{type:"number"}

sweep = 1000#@param{type:"integer"}
n = 10#@param{type:"integer"}

init_up_rate = 1

T_range = np.round(np.arange(init_Temp,final_Temp,step_Temp),3)
L_range = np.arange(init_L,final_L,step_L)

calc_time(T_range, L_range, sweep, n)

result = multi_metropolis(T_range, L_range, sweep, n, init_up_rate)
save(result)