import numpy as np
#import mpi4py.MPI as mpi

from datetime import datetime

import h5py
import os

from FieldsClass import Fields
from RK4 import rk4, der_r
from Horizon import comp_appHorizon

def main(N = 200, R = 2.5, dt = 0.0001, itmax = 100, SAVE = 1):
	
	if SAVE:
		name = f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_1pL_res.h5'
		h5f = h5py.File(name, 'w')

	fields = Fields(R = R, N = N)
	fields.IC_1plusLogSlicing()
	
	
	horizons = np.zeros((N, 4))
	j = 0
	t = 0
	while j < itmax:
		
		fields.fields = np.copy(rk4(fields, dt))
		t += dt
		
		horizons[j,0] = t
		horizons[j,1], horizons[j,3] = comp_appHorizon(fields)
		horizons[j,2] = horizons[j,1] * (1 + 1/4/horizons[j,1])**2
		
		if SAVE:
			reshaped_output = np.reshape(fields.fields, (7, fields.N))
			h5f.create_dataset(f'{j}', data=reshaped_output, compression = 9)
		j += 1
	
	if SAVE:
		h5f.create_dataset(f'Horizon', data=horizons, compression = 9)
		h5f.close()
		os.system(f'python3 Plots_1plusLog.py {name} {N} {itmax} {fields.dR}')


if __name__ == "__main__":
	for N in [100]:
		main(N)
