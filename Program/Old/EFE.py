import numpy as np

def ev_A(f0, i, r, dR, N):
    """
    field :: class object containing all the fields (self in Fields class)
    i,j,k   :: position in the grid
    """
    al = f0[6*N:7*N]
    A  = f0[0  :  N]
    KA = f0[4*N:5*N]
    
    return - 2 * al[i] * A[i] * KA[i]


def ev_B(f0, i, r, dR, N):
    """
    field :: class object containing all the fields (self in Fields class)
    i,j,k   :: position in the grid
    """
    al = f0[6*N:7*N]
    B  = f0[  N:2*N]
    KB = f0[5*N:6*N]
    return - 2 * al[i] * B[i] * KB[i]


def ev_DA(f0, i, r, dR, N):
    """
    field :: class object containing all the fields (self in Fields class)
    i,j,k   :: position in the grid
    """
    al = f0[6*N:7*N]
    KA = f0[4*N:5*N]
    
    p1 = KA[i] * der_r(np.log(al), i, dR)
    p2 = der_r(KA, i, dR)
    return - 2 * al[i] * (p1 + p2)


def ev_DB(f0, i, r, dR, N):
    """
    field :: class object containing all the fields (self in Fields class)
    i,j,k   :: position in the grid
    """
    al = f0[6*N:7*N]
    KB = f0[5*N:6*N]
    
    p1 = KB[i] * der_r(np.log(al), i, dR)
    p2 = der_r(KB, i, dR)
    return - 2 * al[i] * (p1 + p2)


def ev_KA(f0, i, r, dR, N):
    """
    field :: class object containing all the fields (self in Fields class)
    i,j,k   :: position in the grid
    """
    
    A  = f0[0  :  N]
    B  = f0[  N:2*N]
    DA = f0[2*N:3*N]
    DB = f0[3*N:4*N]
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    
    r = r[i]
    Dal = [der_r(np.log(al), j, dR) for j in range(N)]
    
    p1 = der_r(Dal + DB, i, dR)
    p2 = Dal[i]**2 + 0.5 * (- Dal[i] * DA[i] + DB[i]**2 - DA[i] * DB[i])
    p3 = - A[i] * KA[i] * (KA[i] + 2*KB[i])
    p4 = - (DA[i] - 2 * DB[i]) / r
    return - al[i] * (p1 + p2 + p3 + p4) / A[i]


def ev_KB(f0, i, r, dR, N):
    """
    field :: class object containing all the fields (self in Fields class)
    i,j,k   :: position in the grid
    """
    A  = f0[0  :  N]
    B  = f0[  N:2*N]
    DA = f0[2*N:3*N]
    DB = f0[3*N:4*N]
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    
    r = r[i]
    
    p1 = der_r(DB, i, dR)
    p2 = der_r(np.log(al), i, dR) * DB[i] + DB[i]**2 - 0.5 * DA[i] * DB[i]
    p3 = - (DA[i] - 2 * der_r(np.log(al), i, dR) - 4 * DB[i]) / r
    p4 = - 2 * (A[i] - B[i]) / (B[i] * r**2)
    p6 = al[i] * KB[i] * (KA[i] + 2 * KB[i])
    return - 0.5 * al[i] * (p1 + p2 + p3 + p4) / A[i] + p6


def ev_al(f0, i, r, dR, N):
    KA = f0[4*N:5*N]
    KB = f0[5*N:6*N]
    al = f0[6*N:7*N]
    
    return -2 * al[i] * (KA[i] + KB[i]*2)   ## Prvare KA - 2*KB


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