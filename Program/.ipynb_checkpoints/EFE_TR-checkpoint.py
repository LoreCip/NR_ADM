import numpy as np
from numba import njit

@njit
def ev_A(f0, i, r, dR, N, OPL):
    al = f0[6*N:7*N]
    A  = f0[0  :  N]
    KA = f0[4*N:5*N]
    
    return - 2 * al[i] * A[i] * KA[i]

@njit
def ev_B(f0, i, r, dR, N, OPL):
    al = f0[6*N:7*N]
    B  = f0[  N:2*N]
    KB = f0[5*N:6*N]
    return - 2 * al[i] * B[i] * KB[i]

@njit
def ev_DA(f0, i, r, dR, N, OPL):
    al = f0[6*N:7*N]
    KA = f0[4*N:5*N]
    
    p1 = KA[i] * der_r(np.log(al), i, dR)
    p2 = der_r(KA, i, dR)
    return - 2 * al[i] * (p1 + p2)

@njit
def ev_DB(f0, i, r, dR, N, OPL):
    al = f0[6*N:7*N]
    KB = f0[5*N:6*N]
    
    p1 = KB[i] * der_r(np.log(al), i, dR)
    p2 = der_r(KB, i, dR)
    return - 2 * al[i] * (p1 + p2)

@njit
def ev_KA(f0, i, r, dR, N, OPL):
    A  = f0[0  :  N]
    B  = f0[  N:2*N]
    DA = f0[2*N:3*N]
    DB = f0[3*N:4*N]
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    
    psi = 1 + 1 / 4 / r
    r = r[i]
    Dal = np.array([der_r(np.log(al), j, dR) for j in range(N)])
    
    
    p1 = der_r(Dal + DB, i, dR)
    p2 = Dal[i]**2 + 0.5 * (- Dal[i] * DA[i] + DB[i]**2 - DA[i] * DB[i])
    p3 = - psi[i]**4 * A[i] * KA[i] * (KA[i] + 2*KB[i])
    p4 = - (DA[i] - 2 * DB[i]) / r
    p5 = 4 * sec_der_r(np.log(psi), i, dR) + der_r(np.log(psi), i, dR) * (2 * DB[i] - 2 * DA[i] - 2 * Dal[i]  + 4 / r)
	
    return - al[i] * (p1 + p2 + p3 + p4 + p5) / (A[i] * psi[i]**4)

@njit
def ev_KB(f0, i, r, dR, N, OPL):
    A  = f0[0  :  N]
    B  = f0[  N:2*N]
    DA = f0[2*N:3*N]
    DB = f0[3*N:4*N]
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    
    psi = 1 + 1 / 4 / r
    r = r[i]
    Dal = np.array([der_r(np.log(al), j, dR) for j in range(N)])
	
    p1 = der_r(DB, i, dR)
    p2 = Dal[i] * DB[i] + DB[i]**2 - 0.5 * DA[i] * DB[i]
    p3 = - (DA[i] - 2 * Dal[i] - 4 * DB[i]) / r
    p4 = - 2 * (A[i] - B[i]) / (B[i] * r**2)
    
    p5 = 4 * sec_der_r(np.log(psi), i, dR) + der_r(np.log(psi), i, dR) * (8*der_r(np.log(psi), i, dR) + 4*Dal[i] + 6*DB[i] -2*DA[i] + 12 / r)
	
    p6 = al[i] * KB[i] * (KA[i] + 2 * KB[i])
    return - 0.5 * al[i] * (p1 + p2 + p3 + p4 + p5) / (A[i] * psi[i]**4) + p6

@njit
def ev_al(f0, i, r, dR, N, OPL):
    if OPL:
        KA = f0[4*N:5*N]
        KB = f0[5*N:6*N]
        al = f0[6*N:7*N]
        
        return -2 * al[i] * (KA[i] + KB[i]*2)
    else:
        return 0

@njit
def der_r(f, i, dR):
    if i == 0:
        # Forward
        return 0.5 * (- 3 * f[0] + 4 * f[1] - f[2]) / dR
    elif i == len(f)-1:
        # Backward
        return 0.5 * (3 * f[-1] - 4 * f[-2] + f[-3]) / dR
    else:
        # Central
        return 0.5 * (f[i+1] - f[i-1]) / dR
    
@njit
def sec_der_r(f, i, dR):
    if i == 0:
        # Forward
        return (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / dR**3
    elif i == len(f)-1:
        # Backward
        return (2 * f[-1] -5 * f[-2] + 4 * f[-3] - f[-4]) / dR**3
    else:
        # Central
        return (f[i+1] - 2 * f[i] + f[i-1]) / dR**2