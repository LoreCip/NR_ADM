import numpy as np
from numba import njit

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from EFE_TR import d_r

@njit
def func(A, B, KB, r, N, dR):
    dB = d_r(B, dR)
    return (2 / r + dB / B) / np.sqrt(A) - 2 * KB

@njit
def comp_Surface(rh, B):
	return 4*np.pi*B*rh**2

@njit
def find_sgn_change(function, N):
    out = np.where(function[:-1] * function[1:] < 0 )[0] +1
    return out[-1]

def comp_appHorizon(fields):
    
    r = fields.r[2:]
    A = fields.A()[2:] * fields.psi[2:]**4
    B = fields.B()[2:] * fields.psi[2:]**4
    KB = fields.KB()[2:]
    
    function = func(A, B, KB, r, fields.N, fields.dR)
    i = find_sgn_change(function, fields.N)

    pos = [r       [j] for j in [i-2, i-1, i, i+1, i+2]]
    val = [function[j] for j in [i-2, i-1, i, i+1, i+2]]
    Bs  = [B       [j] for j in [i-2, i-1, i, i+1, i+2]]
        
    interp  = interp1d(pos, val, kind = 'cubic')
    interpB = interp1d(pos, Bs, kind = 'cubic')
    root = fsolve(interp, r[i])
    surf = comp_Surface(root, interpB(root))
    
    return root, surf
	
	
