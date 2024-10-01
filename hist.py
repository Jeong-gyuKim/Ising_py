import numpy as np
import numba

DT_scan = 2#1.0
dT_scan = 0.001

DT = 2#0.05
tol_T = 0.0000001
tol_Z1 = 0.0001

MAX_D = 200#5
MAX_E = 100000

#/* histogram method */
energy = np.zeros((MAX_D, MAX_E),dtype=np.double)
hist_E = np.zeros((MAX_D, MAX_E),dtype=np.double)
hist_M = np.zeros((MAX_D, MAX_E),dtype=np.double)
hist_M2 = np.zeros((MAX_D, MAX_E),dtype=np.double)
hist_M4 = np.zeros((MAX_D, MAX_E),dtype=np.double)

SIZE_E = np.zeros(MAX_D,dtype=np.int64)
SIZE_D = None

T0 = None
T1 = np.zeros(MAX_D,dtype=np.double)    #/* Original Temperature */
Nsite = None
Z1 = np.zeros(MAX_D,dtype=np.double)

def read_hist():
    global SIZE_D, T0, Nsite
    T0 = 0.0
    D = 0
    for D0 in range(4):
        dum2 = "data/hist{}.csv".format(D0)
        try:
            with open(dum2,"r") as histfile:
                if D >= MAX_D:
                    print("Too small MAX_D = {}".format(MAX_D))
                    return -1
                print("#Reading file {}".format(dum2))
                
                dum, tmp1, dum2, tmp2 = histfile.readline().split(',')
                T1[D] = float(tmp1)
                Nsite = int(tmp2)
                T0 += T1[D]
                dum = histfile.readline()
                
                print("#T1 = {}  Nsite = {}".format(T1[D],Nsite))
                
                for i in range(MAX_E):
                    try:
                        tmp1, tmp2, tmp3, tmp4, tmp5 = histfile.readline().split(',')
                        energy[D][i] = float(tmp1)
                        hist_E[D][i] = float(tmp2)
                        hist_M[D][i] = float(tmp3)
                        hist_M2[D][i] = float(tmp4)
                        hist_M4[D][i] = float(tmp5)
                        
                    except:
                        break
                    i += 1
                SIZE_E[D] = i
                if SIZE_E[D] == MAX_E:
                    print("Too small MAX_E = {}".format(MAX_E))
                    return -1
        except:
            continue
        D += 1
    SIZE_D = D
    if SIZE_D == 0:
        print("Error reading file hist?.in")
        return -1
    T0 = T0 / SIZE_D
    
def get_Z1():
    Z2 = np.zeros(MAX_D,dtype=np.double)
    
    for D in range(SIZE_D):
        Z1[D] = 1.0
        
    if SIZE_D <= 1:
        return
    
    for iter in range(50):
        tot_Z = 0.
        for D in range(SIZE_D):
            Z2[D] = 0.
            for i in range(SIZE_D):
                for k in range(SIZE_E[i]):
                    s_Z = 0.
                    for j in range(SIZE_D):
                        s_Z += np.exp((1./T1[D] - 1./T1[j])*energy[i][k])/Z1[j]
                    Z2[D] += hist_E[i][k] / s_Z
            tot_Z += Z2[D]
        diff_Z = 0.
        for D in range(SIZE_D):
            Z2[D] = Z2[D]/tot_Z
            diff_Z += abs(Z2[D] - Z1[D])
            Z1[D] = Z2[D]
        if diff_Z < tol_Z1:
            break
    print("After {} iteration, Z_i is obtained.".format(iter))
    for D in range(SIZE_D):
        print("{:.8e}".format(Z1[D]),end=" ")
    print()

def get_avg(T):
    s_e = s_e2 = s_m = s_m2 = s_m4 = s_me = s_m2e = s_m4e = Z = 0.
    for D in range(SIZE_D):
        for i in range(SIZE_E[D]):
            Z_1   = np.log(Z1[D] * hist_E[D][i]) + (-energy[D][i]*(1./T-1./T1[D]))
            Z += np.log(1 + np.exp(Z_1 - Z))
        #Z += np.log(1 - np.exp(-Z))
    for D in range(SIZE_D):
        for i in range(SIZE_E[D]):
            e_1   = energy[D][i]/np.double(Nsite)
            m_1   = hist_M[D][i]
            m_2   = hist_M2[D][i]
            m_4   = hist_M4[D][i]
            Z_1   = Z1[D] * hist_E[D][i] * np.exp(-energy[D][i]*(1./T-1./T1[D]) - Z)
            
            s_e  += e_1 * Z_1 
            s_e2 += (e_1*e_1) * Z_1 
            s_m  += m_1 * Z_1 
            s_m2 += m_2 * Z_1 
            s_m4 += m_4 * Z_1 
            s_me += (m_1*e_1) * Z_1 
            s_m2e += (m_2*e_1) * Z_1 
            s_m4e += (m_4*e_1) * Z_1 
    """
    s_e = s_e2 = s_m = s_m2 = s_m4 = s_me = s_m2e = s_m4e = Z = 0.
    for D in range(SIZE_D):
        for i in range(SIZE_E[D]):
            e_1   = energy[D][i]/np.double(Nsite)
            m_1   = hist_M[D][i]
            m_2   = hist_M2[D][i]
            m_4   = hist_M4[D][i]
            Z_1   = Z1[D] * hist_E[D][i]*np.exp(-energy[D][i]*(1./T-1./T1[D]))
            
            s_e  += e_1 * Z_1 
            s_e2 += (e_1*e_1) * Z_1 
            s_m  += m_1 * Z_1 
            s_m2 += m_2 * Z_1 
            s_m4 += m_4 * Z_1 
            s_me += (m_1*e_1) * Z_1 
            s_m2e += (m_2*e_1) * Z_1 
            s_m4e += (m_4*e_1) * Z_1 

            Z    += Z_1 
    
    s_e  = s_e/Z
    s_e2 = s_e2/Z
    s_m  = s_m/Z
    s_m2 = s_m2/Z
    s_m4 = s_m4/Z
    s_me  = s_me/Z
    s_m2e = s_m2e/Z
    s_m4e = s_m4e/Z
    """
    return s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e

def Cv(T):
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(T)
    return np.double(Nsite)*(s_e2-s_e*s_e)/(T*T)

def chi(T):
    s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(T)
    return np.double(Nsite)*(s_m2-s_m*s_m)/(T)
    
def dm_dbeta(T):
   s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(T)

   return np.double(Nsite)*(s_m*s_e - s_me)

def binder(T):
   s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(T)

   return 1.0 - s_m4/(s_m2*s_m2*3.0)

def dbinder_dbeta(T):
   s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(T)

   return np.double(Nsite)*(s_m2*(s_m4e+s_m4*s_e) - 2.0*s_m4*s_m2e)/(3.0*s_m2*s_m2*s_m2)

def scan_T():
    with open("hist.csv", "w") as out:
        out.writelines("T,m,chi,E,Cv,dm/dbeta,binder,dbinder/dbeta\n")
        
        for T in np.arange(T0-DT_scan,T0+DT_scan,dT_scan):
            s_e, s_e2, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e = get_avg(T)
            out.writelines("{:.3f},{:11.8e},{:11.8e},{:11.8e},{:11.8e},{:11.8e},{:11.8e},{:11.8e}\n".format(T, s_m/Nsite, np.double(Nsite)*(s_m2-s_m*s_m)/(T), s_e, np.double(Nsite)*(s_e2-s_e*s_e)/(T*T), np.double(Nsite)*(s_m*s_e - s_me), 1.0 - s_m4/(s_m2*s_m2*3.0), np.double(Nsite)*(s_m2*(s_m4e+s_m4*s_e) - 2.0*s_m4*s_m2e)/(3.0*s_m2*s_m2*s_m2) ))
    
def golden(ax, bx, cx, f, tol):
    R = 0.61803399
    C = (1.0-R)

    x0, x3 = ax, cx
    if abs(cx-bx) < abs(bx-ax):
        x1, x2 = bx, bx + C*(cx-bx)
    else:
        x2, x1 = bx, bx - C*(bx-ax)
    
    f1, f2 = f(x1), f(x2)
    while abs(x3 - x0) > tol * (abs(x1) + abs(x2)):
        if f2>f1:
            x0, x1, x2 = x1, x2, R*x1+C*x3
            f1, f2 = f2, f(x2)
        else:
            x3, x2, x1 = x2, x1, R*x2+C*x0
            f2, f1 = f1, f(x1)
    
    if f1 > f2:
        return x1, f1
    else:
        return x2, f2

#/* Main Function */
def main():
    if read_hist() == -1:
        return 0
    get_Z1()
    scan_T()

    T_chi_max, chi_max                      = golden(T0-DT,T0,T0+DT,chi          ,tol_T)
    T_Cv_max, Cv_max                        = golden(T0-DT,T0,T0+DT,Cv           ,tol_T)
    T_dm_dbeta_max, dm_dbeta_max            = golden(T0-DT,T0,T0+DT,dm_dbeta     ,tol_T)
    T_dbinder_dbeta_max, dbinder_dbeta_max  = golden(T0-DT,T0,T0+DT,dbinder_dbeta,tol_T)

    print("T_chi_max\tT_Cv_max\tT_dm_dbeta_max\tT_dbinder_dbeta_max\tchi_max\t\tCv_max\t\tdm_dbeta_max\tdbinder_dbeta_max ")
    print("{:11.8e}\t{:11.8e}\t{:11.8e}\t{:11.8e}\t\t{:11.8e}\t{:11.8e}\t{:11.8e}\t{:11.8e}".format(T_chi_max, T_Cv_max, T_dm_dbeta_max, T_dbinder_dbeta_max,chi_max,Cv_max,dm_dbeta_max,dbinder_dbeta_max))

main()