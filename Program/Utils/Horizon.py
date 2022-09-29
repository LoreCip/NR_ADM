import numpy as np
from numba import njit

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from Utils.EFE_TR import d_r

@njit
def h_def(A, B, KB, r, N, dR):
	"""
	Computes in each point of the grid the definition of the apparent horizon.
	Inputs:
			A  : field A
			B  : field B
			KB : field KB
			r  : spacial grid
			N  : number of points on the spacial grid
			dR : discretization step of the spacial grid
		
	Outputs:
			_  : array containing the values of definition of the apparent horizon in each grid point
	"""
	dB = d_r(B, dR)
	return (2 / r + dB / B) / np.sqrt(A) - 2 * KB


@njit
def find_sgn_change(func):
	"""
	Find where the function changes sign.
	
	Inputs:
			func  : array containing the function whose zeros are to be found
			N  : number of points on the spacial grid
			
	Outputs:
			_  : last index of func in which the sign changes
	"""
	out = np.where(func[:-2] * func[2:] < 0 )[0] + 1
	return out[-1]

def comp_appHorizon(fields):
	"""
	Finds the apparent horizon location at each time step. An estimate is found looking
	for zeros of the function h_def(); greater accuracy then grid spacing is obtained
	using cubic interpolation.
	
	Inputs:
			fields  : Fields class containing all the fields informations
	
	Outputs:
			root    : value of the location of the apparent horizon
			surf    : surface of the apparent horzion, surf = 4 * pi * B(root) * root^2
	"""
	r = fields.r[2:-2]
	A = fields.A()[2:-2] * fields.psi[2:-2]**4
	B = fields.B()[2:-2] * fields.psi[2:-2]**4
	KB = fields.KB()[2:-2]
	
	# Evalute the definition of the apparent horizon in each point
	expansion = h_def(A, B, KB, r, fields.N, fields.dR)
	# Find in which interval the function has its outmost zero
	i = find_sgn_change(expansion)

	pos = [r        [j] for j in [i-2, i-1, i, i+1, i+2]]
	val = [expansion[j] for j in [i-2, i-1, i, i+1, i+2]]
	Bs  = [B        [j] for j in [i-2, i-1, i, i+1, i+2]]

	# Define cubic interpolations for the function and the B field
	interp  = interp1d(pos, val, kind = 'cubic')
	interpB = interp1d(pos, Bs, kind = 'cubic')
	# Compute the root and the surface
	root = fsolve(interp, r[i])
	surf = 4 * np.pi * interpB(root) * root**2
	
	return root, surf