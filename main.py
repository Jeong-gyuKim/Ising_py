from ising import compute_statistics, write_to_file
import numpy as np
#Tc = 2/np.log(1+np.sqrt(2))

####상태 입력####
init_L = 10#@param{type:"integer"}
final_L = 20#@param{type:"integer"}
step_L = 1#@param{type:"integer"}

init_Temp = 1#@param{type:"number"}
final_Temp = 3#@param{type:"number"}
step_Temp = 0.1#@param{type:"number"}

N = 10000#@param{type:"integer"}

########
T_range = np.round(np.arange(init_Temp,final_Temp,step_Temp),3)
T_range = np.flip(T_range)
#T_range = np.concatenate((np.flip(T_range),T_range))#T 고온에서 저온으로, 저온에서 고온으로

L_range = np.arange(init_L,final_L,step_L)
########
print("격자 크기:", init_L,final_L,step_L,
      "\n온도:", init_Temp,final_Temp,step_Temp,
      "\nN:",N)

result = compute_statistics(T_range, L_range, N)#계산
write_to_file(result, 'data.pickle')