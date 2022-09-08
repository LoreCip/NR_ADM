import numpy as np
from scipy.interpolate import interp1d

from EFE_TR import *

def eulerStep(fields, dt, k, fac):
    fields_dict = {
            0: fields.A,  1: fields.B,
            2: fields.DA, 3: fields.DB,
            4: fields.KA, 5: fields.KB,
            6: fields.al
                    }
    
    results = np.zeros(7 * fields.N)
    for f in range(7):
        init = f*fields.N
        fin  = (f+1)*fields.N
        
		# Central part
        results[init+2:fin-1] = fields_dict[f]()[2:-1] + k[init+2:fin-1] * fac * dt
        # (A)symmetry condition (ghosts) TRUE????
        if f == 2 or f == 3:
            results[init+1] = - results[init+2]
            results[init]   = - results[init+3]
        else:
            results[init+1] =   results[init+2]
            results[init]   =   results[init+3]
        
        # Boundary extrapolation
        ite = interp1d([fields.r[-3], fields.r[-2]], [results[fin-3], results[fin-2]], fill_value='extrapolate')
        results[fin-1] = ite(fields.r[-1])
    return results

def RHS(f0, r, dr, N, OPL):
    func_dict = {
            0: ev_A,  1: ev_B,
            2: ev_DA, 3: ev_DB,
            4: ev_KA, 5: ev_KB,
            6: ev_al
            }
    rhs = np.zeros(7 * N)
    for f in range(7):
        rhs[N * f + 2 : N * f + N] = func_dict[f](f0, r, dr, N, OPL)
    return rhs

def rk4(fields, dt):
    k1 = RHS(fields.fields, fields.r, fields.dR, fields.N, fields.OPL)
    k2 = RHS(eulerStep(fields, dt, k1, 0.5), fields.r, fields.dR, fields.N, fields.OPL)
    k3 = RHS(eulerStep(fields, dt, k2, 0.5), fields.r, fields.dR, fields.N, fields.OPL)
    k4 = RHS(eulerStep(fields, dt, k3, 1  ), fields.r, fields.dR, fields.N, fields.OPL)
    out = fields.fields + (k1/6 + k2/3 + k3/3 + k4/6) * dt / 2
    
    for f in range(7):
         init = f*fields.N
         fin  = (f+1)*fields.N
        # (A)symmetry condition
         if f == 2 or f == 3:
            out[init+1] = - out[init+2]
            out[init]   = - out[init+3]
         else:
            out[init+1] =   out[init+2]
            out[init]   =   out[init+3]
        # Boundary extrapolation
         ite = interp1d([fields.r[-3], fields.r[-2]], [out[fin-3], out[fin-2]], fill_value='extrapolate')
         out[fin-1] = ite(fields.r[-1])
    return out