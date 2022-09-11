import numpy as np
from numba import njit

from EFE_TR import d2_r, d_r

@njit
def RicciScalar(A, B, r, dR):
    p1 = - 2 * d2_r(B, dR) / (A * B) + 0.5 * d_r(B, dR)**2 / (A * B*B)
    p2 = d_r(A, dR) * d_r(B, dR) / (A*A * B) - 6 * d_r(B, dR) / (r * A * B)
    p3 = 2 * d_r(A, dR) / (r * A*A) + 2 / (r*r * B) - 2 / (r*r * A)
    return p1 + p2 + p3

@njit
def l2norm(f):
    return np.sqrt(np.sum(np.power(f,2)))

def comp_Hconstraint(fields):
    
    r = fields.r[2:]
    A = fields.A()[2:] * fields.psi[2:]**4
    B = fields.B()[2:] * fields.psi[2:]**4
    KA = fields.KA()[2:]
    KB = fields.KB()[2:]
    
    R = RicciScalar(A, B, r, fields.dR)
    K = KA / A + 2 * KB / (r*r * B)
    return l2norm(R + K**2 - (KA / A)**2 - 2 * (KB / (r*r * B))**2)