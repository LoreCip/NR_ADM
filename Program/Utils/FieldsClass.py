import numpy as np

class Fields():
	"""
	Class that contains all the informations on the eight fields that are being evolved in time.
	"""
	def __init__(self, R = 2.5, N = 100, nfields = 8):
		
		self.M = 1
		
		self.OPL = False
		self.nfields = nfields
		
		# Domain is a sphere of radius R
		self.Rmax = R
		self.N    = N + 2  + 2# 2 are the ghost cells at the left and right of each field
		self.dR  = R / N
		
		# Staggered grid to avoid r = 0
		self.r = np.array([(j - 0.5)*self.dR for j in range(1,N+3)], dtype = np.float64)
		
		# Allocate space for the fields
		self.fields = np.zeros(nfields * self.N)
		# Compute psi
		self.psi = 1 + self.M / (2 * self.r) 
		
		# Add to r and psi two None ghost cells at the beginning
		self.r = np.insert(self.r, 0, [None, None])
		self.psi = np.insert(self.psi, 0, [None, None])


	# Methods to retrieve each field from the self.fields array
	def A(self):
		return self.fields[0:self.N]
	def B(self):
		return self.fields[self.N:2*self.N]
	def DA(self):
		return self.fields[2*self.N:3*self.N]
	def DB(self):
		return self.fields[3*self.N:4*self.N]
	def KA(self):
		return self.fields[4*self.N:5*self.N]
	def KB(self):
		return self.fields[5*self.N:6*self.N]
	def al(self):
		return self.fields[6*self.N:7*self.N]
	def Dal(self):
		return self.fields[7*self.N:8*self.N]
    
    
	def IC_1plusLogSlicing(self):
		"""
		1+Log Slicing gauge condition.
				\tilde{A} = \tilde{B} = 1
				\tilde{DA} = \tilde{DB} = 0
				KA = KB = 0
				alpha   = 1
		"""
		self.OPL = True
		
		self.fields[0:self.N] = np.ones(self.N)
		self.fields[self.N:2*self.N]  = np.ones(self.N)
		self.fields[6*self.N:7*self.N] = np.ones(self.N)


	def IC_GeodesicSlicing(self):
		"""
		Geodesic Slicing gauge condition.
				\tilde{A} = \tilde{B} = 1
				\tilde{DA} = \tilde{DB} = 0
				KA = KB = 0
				alpha = 1 = cost.
		"""
		
		self.fields[0:self.N] = np.ones(self.N)
		self.fields[self.N:2*self.N]  = np.ones(self.N)
		self.fields[6*self.N:7*self.N] = np.ones(self.N)