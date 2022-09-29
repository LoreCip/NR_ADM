import numpy as np
from numba import njit

@njit
def ev_A(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for A.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	al = f0[6*N:7*N]
	A  = f0[0  :  N]
	KA = f0[4*N:5*N]
	return - 2 * al[2:-2] * A[2:-2] * KA[2:-2]

@njit
def ev_B(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for B.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	al = f0[6*N:7*N]
	B  = f0[  N:2*N]
	KB = f0[5*N:6*N]
	return - 2 * al[2:-2] * B[2:-2] * KB[2:-2]

@njit
def ev_DA(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for DA.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	al = f0[6*N:7*N]
	KA = f0[4*N:5*N]
	Dal =f0[7*N:8*N]
	
	p1 = KA[2:-2] * Dal[2:-2]
	p2 = d_r(KA, dR, True)[2:-2]
	return - 2 * al[2:-2] * (p1 + p2)

@njit
def ev_DB(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for DB.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	al = f0[6*N:7*N]
	KB = f0[5*N:6*N]
	Dal =f0[7*N:8*N]
	
	p1 = KB[2:-2] * Dal[2:-2]
	p2 = d_r(KB, dR, True)[2:-2]
	return - 2 * al[2:-2] * (p1 + p2)

@njit
def ev_KA(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for KA.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	A  = f0[0  :  N]
	B  = f0[  N:2*N]
	DA = f0[2*N:3*N]
	DB = f0[3*N:4*N]
	KA = f0[4*N:5*N]
	KB = f0[5*N:6*N]
	al = f0[6*N:7*N]
	Dal =f0[7*N:8*N]
	
	psi = 1 + M / (2 * r)
	
	p1 = d_r(Dal + DB, dR, True)[2:-2]
	p2 = Dal[2:-2]**2 + 0.5 * (- Dal[2:-2] * DA[2:-2] + DB[2:-2]**2 - DA[2:-2] * DB[2:-2])
	p3 = - psi[2:-2]**4 * A[2:-2] * KA[2:-2] * (KA[2:-2] + 2*KB[2:-2])
	p4 = - (DA[2:-2] - 2 * DB[2:-2]) / r[2:-2]
	p5 = 4 * d2_r(np.log(psi[2:-2]), dR) + d_r(np.log(psi[2:-2]), dR) * (2 * DB[2:-2] - 2 * DA[2:-2] - 2 * Dal[2:-2] + 4 / r[2:-2])
	
	return - al[2:-2] * (p1 + p2 + p3 + p4 + p5) / (A[2:-2] * psi[2:-2]**4)

@njit
def ev_KB(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for KB.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	A  = f0[0  :  N]
	B  = f0[  N:2*N]
	DA = f0[2*N:3*N]
	DB = f0[3*N:4*N]
	KA = f0[4*N:5*N]
	KB = f0[5*N:6*N]
	al = f0[6*N:7*N]
	Dal =f0[7*N:8*N]
	
	psi = 1 + M / (2 * r)
	
	p1 = d_r(DB, dR, True)[2:-2]
	p2 = Dal[2:-2] * DB[2:-2] + DB[2:-2]**2 - 0.5 * DA[2:-2] * DB[2:-2]
	p3 = - (DA[2:-2] - 2 * Dal[2:-2] - 4 * DB[2:-2]) / r[2:-2]
	p4 = - 2 * (A[2:-2] - B[2:-2]) / (B[2:-2] * r[2:-2]**2)
	
	p5 = 4 * d2_r(np.log(psi[2:-2]), dR) + d_r(np.log(psi[2:-2]), dR) * (8*d_r(np.log(psi[2:-2]), dR) + 4*Dal[2:-2] + 6*DB[2:-2] -2*DA[2:-2] + 12 / r[2:-2])

	p6 = al[2:-2] * KB[2:-2] * (KA[2:-2] + 2 * KB[2:-2])
	return - 0.5 * al[2:-2] * (p1 + p2 + p3 + p4 + p5) / (A[2:-2] * psi[2:-2]**4) + p6

@njit
def ev_al(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for the lapse.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	if OPL:
		KA = f0[4*N:5*N]
		KB = f0[5*N:6*N]
		al = f0[6*N:7*N]
		
		return -2 * al[2:-2] * (KA[2:-2] + 2 * KB[2:-2])
	else:
		return np.zeros(N)[2:-2]


@njit
def ev_Dal(f0, r, dR, M, N, OPL):
	"""
	Evoultion equation for D_alpha.
	
	Input:
		fo     : array of length fields.nfield * fields.N containg all the old fields
		r      : array containing the spacial grid
		dR     : discretization step of the spacial grid
		M      : mass of the black hole
		N      : number of points on the spacial grid
		OPL    : logical variable, check which evolution equation to use for the lapse
			
	Outputs:
		_      : an array of length fields.N - 2 containing the RHS of the evolution equations
	"""
	if OPL:
		KA = f0[4*N:5*N]
		KB = f0[5*N:6*N]
		
		return - 2 * d_r(KA[2:-2] + 2 * KB[2:-2], dR)
	else:
		return np.zeros(N)[2:-2]


@njit
def d_r(f, dR, ghost = False):
	"""
	Computes the first derivative with respect to the radial coordinate at fourth order using
	finite differences.
	
	Inputs:
		f    : array to differenciate
		dR   : discretization step of the spacial grid
		ghost: if ghost is False the first two ghost cell are ignored in the computation of the derivate
			   If ghost is True the derivative of the ghost cell is computed using asymmetric finite differences
	
	Outputs:
		out  : array containing the derivative of f
	"""
	out = np.zeros_like(f)
	if not ghost:
		out[0:2] = (-25*f[0:2]/12+4*f[1:3]-3*f[2:4]+4*f[3:5]/3-f[4:6]/4)/dR
		out[-2:] = (25*f[-2:]/12-4*f[-3:-1]+3*f[-4:-2]-4*f[-5:-3]/3+f[-6:-4]/4)/dR
	out[2:-2] = (f[0:-4]-8*f[1:-3]+8*f[3:-1]-f[4:])/(12*dR)
	return out

    
@njit
def d2_r(f, dR):
	"""
	Computes the second derivative with respect to the radial coordinate at fourth order using
	finite differences.
	
	Inputs:
		f    : array to differenciate
		dR   : discretization step of the spacial grid
	
	Outputs:
		out  : array containing the derivative of f
	"""
	out = np.zeros_like(f)
	out[0:2]  = (35*f[0:2]-104*f[1:3]+114*f[2:4]-56*f[3:5]+11*f[4:6])/(12*dR**2)
	out[2:-2] = (-f[0:-4]+16*f[1:-3]-30*f[2:-2]+16*f[3:-1]-f[4:])/(12*dR**2)
	out[-2:]  = (11*f[-6:-4]-56*f[-5:-3]+114*f[-4:-2]-104*f[-3:-1]+35*f[-2:])/(12*dR**2)
	return out