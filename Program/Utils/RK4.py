import numpy as np
from scipy.interpolate import interp1d

from Utils.EFE_TR import *

def imposeBC(arr, f, init, fin, r):
	"""
	Imposes the boundary conditions on each field.
	At infinity the BC is found with linear interpolation, at zero with symmetry considerations:
			- {A, B, KA, KB}[1 - i] = {A, B, KA, KB}[i]
			- {DA, DB}      [1 - i] = {DA, DB}      [i]
	for i = 1, 2. Note that cells 0 and 1 are ghost cells included only to compute the derivatives
	using central differences approximations.
	
	Inputs:
			arr   : array containing all the fields
			f     : number identifying which field is considered
			init  : starting index of field f
			fin   : last index of field f
			r     : spacial grid
			
	Outputs:
			arr   : array containing all the fields with the correct boundary conditions
	"""
	# Internal boundary (A)symmetry condition (on ghost cells)
	if f == 2 or f == 3:
		arr[init+1] = - arr[init+2]
		arr[init]   = - arr[init+3]
	else:
		arr[init+1] =   arr[init+2]
		arr[init]   =   arr[init+3]
	# External boundary extrapolation
	ite = interp1d([r[-4], r[-3]], [arr[fin-4], arr[fin-3]], fill_value='extrapolate')
	arr[fin-1] = ite(r[-1])
	arr[fin-2] = ite(r[-2])
	
	return arr



def eulerStep(fields, dt, k, fac):
	"""
	Performs a simple Euler step to compute intermediate values in the RK4 algorithm.
	
	Inputs:
			fields  : Fields class containing all the fields informations
			dt      : time step
			k       : array containing the RHS of the evolution equations
			fac     : parameter define in the RK4 algorithm
			
	Outputs:
			results : an array of length fields.nfield * fields.N containing all intermediate fields
	"""
	fields_dict = {
			0: fields.A,  1: fields.B,
			2: fields.DA, 3: fields.DB,
			4: fields.KA, 5: fields.KB,
			6: fields.al, 7: fields.Dal
					}
	
	results = np.zeros(fields.nfields * fields.N)
	for f in range(fields.nfields):
		init = f*fields.N
		fin  = (f+1)*fields.N
		
		# Central part
		results[init+2:fin-2] = fields_dict[f]()[2:-2] + k[init+2:fin-2] * fac * dt
		
		results = imposeBC(results, f, init, fin, fields.r)
		
	return results


def RHS(f0, r, dr, M, N, nfields, OPL):
	"""
	Computes the right hand side of the evoultion equations.
	
	Inputs:
			fo     : array of length fields.nfield * fields.N containg all the old fields
			r      : array containing the spacial grid
			dr     : discretization step of the spacial grid
			M      : mass of the black hole
			N      : number of points on the spacial grid
			nfields: number of fields to simulate
			OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
			rhs    : an array of length fields.nfield * fields.N containing all the RHS of the evolution equations
	"""
	func_dict = {
			0: ev_A,  1: ev_B,
			2: ev_DA, 3: ev_DB,
			4: ev_KA, 5: ev_KB,
			6: ev_al, 7: ev_Dal
			}
	rhs = np.zeros(nfields * N)
	for f in range(nfields):
		rhs[N * f + 2 : (f+1)*N-2] = func_dict[f](f0, r, dr, M, N, OPL)
	return rhs



def rk4(fields, dt):
	"""
	Performs one step of Runge-Kutta 4th order evolution in time.
	
	Inputs:
			fields: Fields class containing all the fields informations
			dt    : time step
			
	Outputs:
			out:  : an array of length fields.nfield * fields.N containing all the new fields
	"""
	
	# Runge-Kutta 4 algorithm
	params = (fields.r, fields.dR,  fields.M, fields.N, fields.nfields, fields.OPL)
	k1 = RHS(fields.fields, *params)
	k2 = RHS(eulerStep(fields, dt, k1, 0.5), *params)
	k3 = RHS(eulerStep(fields, dt, k2, 0.5), *params)
	k4 = RHS(eulerStep(fields, dt, k3, 1  ), *params)
	out = fields.fields + (k1/6 + k2/3 + k3/3 + k4/6) * dt
	
	# Apply boundary conditions to each field
	for f in range(fields.nfields):
		init = f*fields.N
		fin  = (f+1)*fields.N
		out = imposeBC(out, f, init, fin, fields.r)
		
	return out