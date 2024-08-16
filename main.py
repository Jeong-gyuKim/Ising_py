from ising import multi_metropolis, calc_time, save, calc_new_values
import numpy as np
#Tc = 2/np.log(1+np.sqrt(2))

####상태 입력####
init_L = 1#@param{type:"integer"}
final_L = 6#@param{type:"integer"}
step_L = 1#@param{type:"integer"}

init_Temp = 0.1#@param{type:"number"}
final_Temp = 8#@param{type:"number"}
step_Temp = 0.1#@param{type:"number"}

sweep = 1000#@param{type:"integer"}
n = 10#@param{type:"integer"}

init_up_rate = 0.5

sigma = 2.0#95% error bar

########
T_range = np.round(np.arange(init_Temp,final_Temp,step_Temp),3)
T_range = np.concatenate((np.flip(T_range),T_range))#T 고온에서 저온으로, 저온에서 고온으로

L_range = np.arange(init_L,final_L,step_L)
########

calc_time(T_range, L_range, sweep, n)#대충 시간

result = multi_metropolis(T_range, L_range, sweep, n, init_up_rate, sigma)#계산
result = calc_new_values(result)#오차범위 확인
save(result)#저장