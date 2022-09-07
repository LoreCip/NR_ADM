import numpy as np

from RK4 import der_r

class Fields():
    
    def __init__(self, R = 10, N = 110):
        # Domain is a sphere of radius R
        self.Rmax = R
        self.N    = N
        
        self.dR  = R / N
    
        self.r = np.array([(j - 0.5)*self.dR for j in range(1,N+1)], dtype = np.float64)
        
        self.fields = np.zeros(7 * N)
        
        self.psi = 1 + 1 / 4 / self.r  #r_s = 1
        self.OPL = False
        
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
    
    def IC_1plusLogSlicing(self):
        """
        1+Log Slicing gauge condition.
                \tilde{A} = \tilde{B} = 1
                \tilde{DA} = \tilde{DB} = 0
                KA = KB = 0
                alpha   = 1
        """
        self.OPL = True
        
        self.fields[0:self.N]  = np.ones(self.N)
        self.fields[self.N:2*self.N]  = np.ones(self.N)
        self.fields[6*self.N:7*self.N] = np.ones(self.N)
        
        
    def IC_GeodesicSlicing(self):
        """
        Geodesic Slicing gauge condition.
                \tilde{A} = \tilde{B} = 1
                \tilde{DA} = \tilde{DB} = 0
                KA = KB = 0
                alpha   = 1 = cost.
        """
        self.fields[0:self.N]  = np.ones(self.N)
        self.fields[self.N:2*self.N]  = np.ones(self.N)
        self.fields[6*self.N:7*self.N] = np.ones(self.N)