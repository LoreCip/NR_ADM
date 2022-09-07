import numpy as np
from numba import njit

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from RK4 import der_r

@njit
def func(A, B, KB, r, N, dR):
    dB = np.zeros(N)
    for i in range(N):
        dB[i] = der_r(B, i, dR)
    
    return (2 / r + dB / B) / np.sqrt(A) - 2 * KB

@njit
def comp_Surface(rh, B):
	return 4*np.pi*B*rh**2

@njit
def find_sgn_change(function, N):
    for i in range(N-1):
        if np.sign(function[i]) != np.sign(function[i+1]):
            break
    return i

def comp_appHorizon(fields):
    
    A = fields.A() * fields.psi**4
    B = fields.B() * fields.psi**4
    
    function = func(A, B, fields.KB(), fields.r, fields.N, fields.dR)
    
    i = find_sgn_change(function, fields.N)
    
    pos = [fields.r[j] for j in [i-1, i, i+1, i+2]]
    val = [function[j] for j in [i-1, i, i+1, i+2]]
    Bs  = [B       [j] for j in [i-1, i, i+1, i+2]]
    
    interp = interp1d(pos, val, kind = 'cubic')
    interpB = interp1d(pos, Bs, kind = 'cubic')
    
    root = fsolve(interp, fields.r[i])
    surf = comp_Surface(root, interpB(root))
    return root, surf
	
	
