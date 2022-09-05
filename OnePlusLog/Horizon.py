import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from RK4 import der_r

def func(fields):
    A = fields.A() * fields.psi**4
    B = fields.B() * fields.psi**4
    
    dB = np.zeros(fields.N)
    for i in range(fields.N):
        dB[i] = der_r(B, i, fields.dR)
    
    return (2 / fields.r + dB / B) / np.sqrt(A) - 2 * fields.KB(), B

def comp_Surface(rh, B):
	return 4*np.pi*B*rh

def comp_appHorizon(fields):
	
    function, B = func(fields)
    for i in range(fields.N-1):
        if np.sign(function[i]) != np.sign(function[i+1]):
            break
    pos = [fields.r[j] for j in [i-1, i, i+1, i+2]]
    val = [function[j] for j in [i-1, i, i+1, i+2]]
    Bs  = [B       [j] for j in [i-1, i, i+1, i+2]]
    
    interp = interp1d(pos, val, kind = 'cubic')
    interpB = interp1d(pos, Bs, kind = 'cubic')
    
    root = fsolve(interp, fields.r[i])
    surf = comp_Surface(root, interpB(root))
    return root, surf
	
	
