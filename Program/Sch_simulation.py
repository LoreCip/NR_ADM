import numpy as np

from time import time
from datetime import timedelta

import h5py
import argparse
from pathlib import Path


from Utils.FieldsClass import Fields
from Utils.RK4 import rk4
from Utils.Horizon import comp_appHorizon

np.seterr(invalid='raise')

def progress_bar(current, total, t_med, bar_length=20):
	"""
	Display a progress bar during the simulation. The estimated time left is computed from the mean iteration time.
	Inputs:
			current   :  current iteration
			total     :  total number of iterations to perform
			t_med     :  mean time per iteration
			bar_length:  length in terminal units of the bar
	"""
	fraction = current / total

	arrow = int(fraction * bar_length - 1) * '-' + '>'
	padding = int(bar_length - len(arrow)) * ' '

	ending = '\n' if current == total else '\r'

	print(f'Progress: [{arrow}{padding}] {int(fraction*100)}% - Est. time left {timedelta(seconds=(total - current)*t_med)}', end=ending)


def main(N = 100, R = 2.5, dt = 0.0001, itmax = 100, OPL = 1, SAVE = 1, SILENT = 0, outdir = 'Outputs/'):
	
	if not SILENT:
		print(f'Initializing the simulation.')

	if SAVE:
		name = f'N{N}_R{R}_t{itmax}_dt{dt}_OPL{OPL}.h5'
		h5f = h5py.File(outdir + name, 'w')
		
	# Fields initialization
	fields = Fields(R = R, N = N)
	# Space allocation for horizon data
	horizons = np.zeros((itmax, 4))
	
	# Courant factor warning
	if dt > fields.dR:
		print(f'Warning: {dt = } > dR = {fields.dR}')
	
	if not SILENT:
		print('Setting fields initial values.')
	
	# Initial conditions
	if OPL:
		fields.IC_1plusLogSlicing()
	else:
		fields.IC_GeodesicSlicing()
	
	j = 0
	t = 0
	t_it = 0
	if not SILENT:
		print('Performing MoL evolution:')
	
	while j < itmax:
		t1 = time()
		# Try to perform a step in time
		try:
			fields.fields = np.copy(rk4(fields, dt))
			
			# Save time in first column
			horizons[j,0] = t
			# Save r_h and S_h in second and fourth column
			horizons[j,1], horizons[j,3] = comp_appHorizon(fields)
			# Save r_h in Schwarzschild standard coordinatates in the third column
			horizons[j,2] = horizons[j,1] * (1 + 1/4/horizons[j,1])**2
		
		# Graceful exti in case of code crash (infinity or NaNs in evolution)
		except:
			print(f'Max iteration is {j}, t = {t}')
			print()
			break
		
		t += dt
		
		# Save a snapshot of the fields every sixth of the total number of steps
		if SAVE and np.any(j == np.linspace(1, itmax-1, 6, dtype = np.int64)) :
			reshaped_output = np.reshape(fields.fields, (fields.nfields, fields.N))
			reshaped_output = np.delete(reshaped_output, [0,1], 1)
			reshaped_output = np.delete(reshaped_output, [-1,-2], 1)
			h5f.create_dataset(f'{j}', data=reshaped_output, compression = 9)
			
		t_it += time() - t1 
		j += 1
		# Update progress bar
		if not SILENT:
			progress_bar(j, itmax, t_it/j)
		
	if not SILENT:    
		print(f'Simulation complete. Total elapsed time: {np.round(t_it,4)} s')
	
	# Save th horizon data and close the H5 file
	if SAVE:    
		if not SILENT:
			print(f'Saving to {name}.')	
		h5f.create_dataset(f'Horizon', data=horizons, compression = 9)
		h5f.close()



if __name__ == '__main__':

	# Default parameters
	params = {
		'points'   : 100,
		'radius'   : 2.5,
		'time_step': 0.0001,
		'iter'     : 100,
		'oplslic'  : 1,
		'savevar'  : 1,
		'silentvar': 0
	}
	
	# Argparser to change default paramters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('-N', '--points', help = 'Number of points', type = int)
	parser.add_argument('-R','--radius', help = 'Length of simulation domain', type = float)
	parser.add_argument('-dt','--time-step', help = 'Time step to employ', type = float)
	parser.add_argument('-it','--iter', help = 'Number of time iterations', type = int)
	parser.add_argument('-OPL','--oplslic', help = '1 to apply 1+Log Slicing conditions, 0 to apply Geodesic Slicing conditions', type = int)
	parser.add_argument('-SAVE','--savevar', help = '1 to save the output in an .h5 file, 0 otherwise', type = int)
	parser.add_argument('-SILENT','--silentvar', help = '1 to suppress text output, 0 otherwise', type = int)

	args = parser.parse_args()
	for arg in vars(args):
		if getattr(args, arg) is not None:
			params[arg] = getattr(args, arg)
	
	# Make Outputs/ directory if it does not exists in this path
	Path("./Outputs/").mkdir(parents=True, exist_ok=True)
	
	# Call to main
	main(*list(params.values()))

