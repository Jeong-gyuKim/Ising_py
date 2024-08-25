from ising import statistics, calc_time, save, read, write, calc_new_values
import numpy as np
#Tc = 2/np.log(1+np.sqrt(2))

####상태 입력####
init_L = 4#@param{type:"integer"}
final_L = 13#@param{type:"integer"}
step_L = 2#@param{type:"integer"}

init_Temp = 1#@param{type:"number"}
final_Temp = 3#@param{type:"number"}
step_Temp = 0.1#@param{type:"number"}

N = 1000#@param{type:"integer"}
n = 10#@param{type:"integer"}

sigma = 2.0#95% error bar

########
T_range = np.round(np.arange(init_Temp,final_Temp,step_Temp),3)
T_range = np.concatenate((np.flip(T_range),T_range))#T 고온에서 저온으로, 저온에서 고온으로

L_range = np.arange(init_L,final_L,step_L)
########

calc_time(N,n)#대충 시간
print("격자 크기:", init_L,final_L,step_L,
      "\n온도:", init_Temp,final_Temp,step_Temp,
      "\nN:",N,
      "\nn번 평균:", n,
      "\n오차:", sigma)

result = statistics(T_range, L_range, N, n, sigma)#계산
result = calc_new_values(result)#오차범위 확인
save(result)#저장