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
    Dal =f0[7*N:8*N]
    
    p1 = KA[2:] * Dal[2:]
    p2 = d_r(KA, dR, True)[2:]
    return - 2 * al[2:] * (p1 + p2)

@njit
def ev_DB(f0, r, dR, N, OPL):
    al = f0[6*N:7*N]
    KB = f0[5*N:6*N]
    Dal =f0[7*N:8*N]
    
    p1 = KB[2:] * Dal[2:]
    p2 = d_r(KB, dR, True)[2:]
    return - 2 * al[2:] * (p1 + p2)


def ev_KA(f0, r, dR, N, OPL):
    A  = f0[0  :  N]
    B  = f0[  N:2*N]
    DA = f0[2*N:3*N]
    DB = f0[3*N:4*N]
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    Dal =f0[7*N:8*N]
    
    psi = 1 + 1 / 4 / r
    
    p1 = d_r(Dal + DB, dR, True)[2:]
    p2 = Dal[2:]**2 + 0.5 * (- Dal[2:] * DA[2:] + DB[2:]**2 - DA[2:] * DB[2:])
    p3 = - psi[2:]**4 * A[2:] * KA[2:] * (KA[2:] + 2*KB[2:])
    p4 = - (DA[2:] - 2 * DB[2:]) / r[2:]
    p5 = 4 * d2_r(np.log(psi[2:]), dR) + d_r(np.log(psi[2:]), dR) * (2 * DB[2:] - 2 * DA[2:] - 2 * Dal[2:] + 4 / r[2:])
    
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
    Dal =f0[7*N:8*N]
    
    psi = 1 + 1 / 4 / r
	
    p1 = d_r(DB, dR, True)[2:]
    p2 = Dal[2:] * DB[2:] + DB[2:]**2 - 0.5 * DA[2:] * DB[2:]
    p3 = - (DA[2:] - 2 * Dal[2:] - 4 * DB[2:]) / r[2:]
    p4 = - 2 * (A[2:] - B[2:]) / (B[2:] * r[2:]**2)
    
    p5 = 4 * d2_r(np.log(psi[2:]), dR) + d_r(np.log(psi[2:]), dR) * (8*d_r(np.log(psi[2:]), dR) + 4*Dal[2:] + 6*DB[2:] -2*DA[2:] + 12 / r[2:])
	
    p6 = al[2:] * KB[2:] * (KA[2:] + 2 * KB[2:])
    return - 0.5 * al[2:] * (p1 + p2 + p3 + p4 + p5) / (A[2:] * psi[2:]**4) + p6

@njit
def ev_al(f0, r, dR, N, OPL):
    if OPL:
        KA = f0[4*N:5*N]
        KB = f0[5*N:6*N]
        al = f0[6*N:7*N]
        
        return -2 * al[2:] * (KA[2:] + 2 * KB[2:])
    else:
        return np.zeros(N)[2:]


@njit
def ev_Dal(f0, r, dR, N, OPL):
    if OPL:
        A  = f0[0  :  N]
        B  = f0[  N:2*N]
        KA = f0[4*N:5*N]
        KB = f0[5*N:6*N]
        
        psi = 1 + 1 / 4 / r
        
        #return - 2 * d_r(KA[2:] / (A[2:] * psi[2:]**4) + 2 * KB[2:] / (r[2:]**2 * B[2:] * psi[2:]**4), dR)
        
        return - 2 * d_r(KA[2:] + 2 * KB[2:], dR)
    else:
        return np.zeros(N)[2:]


@njit
def d_r(f, dR, ghost = False):
    out = np.zeros_like(f)
    if not ghost:
        out[0:2] = (-25*f[0:2]/12+4*f[1:3]-3*f[2:4]+4*f[3:5]/3-f[4:6]/4)/dR
    out[2:-2] = (f[0:-4]-8*f[1:-3]+8*f[3:-1]-f[4:])/(12*dR)
    out[-2:] = (25*f[-2:]/12-4*f[-3:-1]+3*f[-4:-2]-4*f[-5:-3]/3+f[-6:-4]/4)/dR
    return out

    
@njit
def d2_r(f, dR):
    out = np.zeros_like(f)
    out[0:2]  = (35*f[0:2]-104*f[1:3]+114*f[2:4]-56*f[3:5]+11*f[4:6])/(12*dR**2)
    out[2:-2] = (-f[0:-4]+16*f[1:-3]-30*f[2:-2]+16*f[3:-1]-f[4:])/(12*dR**2)
    out[-2:]  = (11*f[-6:-4]-56*f[-5:-3]+114*f[-4:-2]-104*f[-3:-1]+35*f[-2:])/(12*dR**2)
    return out
    