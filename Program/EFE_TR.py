import numpy as np
from numba import njit

@njit
def ev_A(f0, r, dR, N, OPL):
    al = f0[6*N:7*N]
    A  = f0[0  :  N]
    KA = f0[4*N:5*N]
    
    return - 2 * al[2:] * A[2:] * KA[2:]

@njit
def ev_B(f0, r, dR, N, OPL):
    al = f0[6*N:7*N]
    B  = f0[  N:2*N]
    KB = f0[5*N:6*N]
    return - 2 * al[2:] * B[2:] * KB[2:]

@njit
def ev_DA(f0, r, dR, N, OPL):
    al = f0[6*N:7*N]
    KA = f0[4*N:5*N]
    
    p1 = KA[2:] * d_r(np.log(al), dR)[2:]
    p2 = ghost_derivative(KA, dR)[2:]
    
    return - 2 * al[2:] * (p1 + p2)

@njit
def ev_DB(f0, r, dR, N, OPL):
    al = f0[6*N:7*N]
    KB = f0[5*N:6*N]
    
    p1 = KB[2:] * d_r(np.log(al), dR)[2:]
    p2 = ghost_derivative(KB, dR)[2:]
    return - 2 * al[2:] * (p1 + p2)


def ev_KA(f0, r, dR, N, OPL):
    A  = f0[0  :  N]
    B  = f0[  N:2*N]
    DA = f0[2*N:3*N]
    DB = f0[3*N:4*N]
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    
    psi = 1 + 1 / 4 / r
    Dal = d_r(np.log(al), dR)
    
    p1 = ghost_derivative(Dal + DB, dR)[2:]
    p2 = Dal[2:]**2 + 0.5 * (- Dal[2:] * DA[2:] + DB[2:]**2 - DA[2:] * DB[2:])
    p3 = - psi[2:]**4 * A[2:] * KA[2:] * (KA[2:] + 2*KB[2:])
    p4 = - (DA[2:] - 2 * DB[2:]) / r[2:]
    p5 = 4 * d_r(np.log(psi), dR)[2:] + d_r(np.log(psi), dR)[2:] * (2 * DB[2:] - 2 * DA[2:] - 2 * Dal[2:]  + 4 / r[2:])
    
    return - al[2:] * (p1 + p2 + p3 + p4 + p5) / (A[2:] * psi[2:]**4)

@njit
def ev_KB(f0, r, dR, N, OPL):
    A  = f0[0  :  N]
    B  = f0[  N:2*N]
    DA = f0[2*N:3*N]
    DB = f0[3*N:4*N]
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    
    psi = 1 + 1 / 4 / r
    Dal = d_r(np.log(al), dR)
	
    p1 = ghost_derivative(DB, dR)[2:]
    p2 = Dal[2:] * DB[2:] + DB[2:]**2 - 0.5 * DA[2:] * DB[2:]
    p3 = - (DA[2:] - 2 * Dal[2:] - 4 * DB[2:]) / r[2:]
    p4 = - 2 * (A[2:] - B[2:]) / (B[2:] * r[2:]**2)
    
    p5 = 4 * d_r(np.log(psi), dR)[2:] + d_r(np.log(psi), dR)[2:] * (8*d_r(np.log(psi), dR)[2:] + 4*Dal[2:] + 6*DB[2:] -2*DA[2:] + 12 / r[2:])
	
    p6 = al[2:] * KB[2:] * (KA[2:] + 2 * KB[2:])
    return - 0.5 * al[2:] * (p1 + p2 + p3 + p4 + p5) / (A[2:] * psi[2:]**4) + p6

@njit
def ev_al(f0, r, dR, N, OPL):
    if OPL:
        KA = f0[4*N:5*N]
        KB = f0[5*N:6*N]
        al = f0[6*N:7*N]
        
        return -2 * al[2:] * (KA[2:] + KB[2:]*2)
    else:
        return np.zeros(N)[2:]
   
@njit
def ghost_derivative(f, dR):
    out = np.zeros_like(f)
    out[2:-3] = ((f[0:-5] - f[4:-1]) / 12 - 2 * (f[1:-4] + f[3:-2]) / 3 )  /  dR
    out[-3:-1] = (25 * f[-3:-1] / 12 - 4 * f[-4:-2] + 3 * f[-5:-3] - 4 * f[-6:-4] / 3 + f[-7:-5] / 4) / dR
    return out
    
@njit
def d_r(f, dR):
    out = np.zeros_like(f)
    out[2:-3]  = ((f[0:-5] - f[4:-1]) / 12 - 2 * (f[1:-4] + f[3:-2]) / 3 )  /  dR
    out[0:2]   = (- 25 * f[0:2] / 12 + 4 * f[1:3] - 3 * f[2:4] + 4 * f[3:5] / 3 - f[4:6] / 4) / dR
    out[-3:-1] = (25 * f[-3:-1] / 12 - 4 * f[-4:-2] + 3 * f[-5:-3] - 4 * f[-6:-4] / 3 + f[-7:-5] / 4) / dR
    return out
    
@njit
def d2_r(f, i, dR):
    out = np.zeros_like(f)
    out[2:-3]  = (-f[0:-5]+16*f[1:-4]-30*f[2:-3]+16*f[3:-2]-f[4:-1])/(12*dR**2)
    out[0:2]   = (35*f[0:2]-104*f[1:3]+114*f[2:4]-56*f[3:5]+11*f[4:6])/(12*dR**2)
    out[-3:-1] = (11*f[-7:-5]-56*f[-6:-4]+114*f[-5:-3]-104*f[-4:-2]+35*f[-3:-1])/(12*dR**2)
    return out
    