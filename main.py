from ising import compute_statistics
import numpy as np

#Tc = 2/np.log(1+np.sqrt(2))

####상태 입력####
init_L, final_L, step_L = 2,11,1
T0, DT_scan, dT_scan = 2.7, 1.0, 0.5
N = 10**3

########
T_range = np.round(np.arange(T0-DT_scan,T0+DT_scan+dT_scan,dT_scan),3)
L_range = np.arange(init_L,final_L,step_L)

########
print("격자 크기:", init_L,final_L,step_L,
      "\n",L_range,
      "\n온도:", np.round(T0-DT_scan,3),np.round(T0+DT_scan,3),np.round(dT_scan,3),
      "\n",T_range,
      "\nN:",N)

result = compute_statistics(T_range, L_range, N, init_up_rate = 0)