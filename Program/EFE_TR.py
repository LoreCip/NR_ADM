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
    
    p1 = KA[i] * d_r(np.log(al), i, dR) 
    p2 = ghost_derivative(KA, i, dR)
    
    return - 2 * al[i] * (p1 + p2)

@njit
def ev_DB(f0, i, r, dR, N, OPL):
    al = f0[6*N:7*N]
    KB = f0[5*N:6*N]
    
    p1 = KB[i] * d_r(np.log(al), i, dR)
    p2 = ghost_derivative(KB, i, dR)
    return - 2 * al[i] * (p1 + p2)


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
    Dal = np.array([d_r(np.log(al), j, dR) for j in range(N)])
    
    p1 = ghost_derivative(Dal + DB, i, dR)
    p2 = Dal[i]**2 + 0.5 * (- Dal[i] * DA[i] + DB[i]**2 - DA[i] * DB[i])
    p3 = - psi[i]**4 * A[i] * KA[i] * (KA[i] + 2*KB[i])
    p4 = - (DA[i] - 2 * DB[i]) / r
    p5 = 4 * d_r(np.log(psi), i, dR) + d_r(np.log(psi), i, dR) * (2 * DB[i] - 2 * DA[i] - 2 * Dal[i]  + 4 / r)
    
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
    Dal = np.array([d_r(np.log(al), j, dR) for j in range(N)])
	
    p1 = ghost_derivative(DB, i, dR)
    p2 = Dal[i] * DB[i] + DB[i]**2 - 0.5 * DA[i] * DB[i]
    p3 = - (DA[i] - 2 * Dal[i] - 4 * DB[i]) / r
    p4 = - 2 * (A[i] - B[i]) / (B[i] * r**2)
    
    p5 = 4 * d_r(np.log(psi), i, dR) + d_r(np.log(psi), i, dR) * (8*d_r(np.log(psi), i, dR) + 4*Dal[i] + 6*DB[i] -2*DA[i] + 12 / r)
	
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
def ghost_derivative(f, i, dR):
    return ((f[i-2] - f[i+2]) / 12 - 2 * (f[i-1] + f[i+1]) / 3 ) / dR
    
@njit
def d_r(f, i, dR):
    if i == 2 or i == 3:
        return (- 25 * f[i] / 12 + 4 * f[i+1] - 3 * f[i+2] + 4 * f[i+3] / 3 - f[i+4] / 4) / dR
    elif i == len(f) - 1 or i == len(f) - 2:
        return (25 * f[i] / 12 - 4 * f[i-1] + 3 * f[i-2] - 4 * f[i-3] / 3 + f[i-4] / 4) / dR
    else:
        return ((f[i-2] - f[i+2]) / 12 - 2 * (f[i-1] + f[i+1]) / 3 ) / dR
    
@njit
def d_r(f, i, dR):
    if i == 2 or i == 3:
        return (35*f[i]-104*f[i+1]+114*f[i+2]-56*f[i+3]+11*f[i+4])/(12*dR**2)
    elif i == len(f) - 1 or i == len(f) - 2:
        return (11*f[i-4]-56*f[i-3]+114*f[i-2]-104*f[i-1]+35*f[i])/(12*dR**2)
    else:
        return (-f[i-2]+16*f[i-1]-30*f[i+0]+16*f[i+1]-f[i+2])/(12*dR**2)