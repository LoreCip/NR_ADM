import numpy as np

from time import time
from datetime import timedelta

import h5py
import argparse

from FieldsClass import Fields
from RK4 import rk4
from EFE_TR import ghost_derivative
from Horizon import comp_appHorizon
from DashPlots import rundash

def progress_bar(current, total, t_med, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}% - Est. time left {timedelta(seconds=(total - current)*t_med)}', end=ending)

def main(N = 100, R = 2.5, dt = 0.00001, itmax = 100, OPL = 1, SAVE = 1, SILENT = 0):
	
    if SAVE:
	    name = f'N{N}_R{R}_t{itmax}.h5'
	    h5f = h5py.File(name, 'w')
    fields = Fields(R = R, N = N)
    if not SILENT:
        print('Setting fields initial values.')
    
    if OPL:
        fields.IC_1plusLogSlicing()
    else:
        fields.IC_GeodesicSlicing()
    
    horizons = np.zeros((itmax+1, 4))
    j = 0
    t = 0
    t_it = 0
    if not SILENT:
        print('Performing MoL evolution:')

    while j <= itmax:
        t1 = time()
        fields.fields = np.copy(rk4(fields, dt))
        t += dt
        
        horizons[j,0] = t
        horizons[j,1], horizons[j,3] = comp_appHorizon(fields)
        horizons[j,2] = horizons[j,1] * (1 + 1/4/horizons[j,1])**2
        
        if SAVE:
            reshaped_output = np.reshape(fields.fields, (7, fields.N))
            reshaped_output = np.delete(reshaped_output, [0,1], 1)
            h5f.create_dataset(f'{j}', data=reshaped_output, compression = 9)
            
        t_it += time() - t1 
        j += 1
        if not SILENT:
            progress_bar(j, itmax, t_it/j)
        
    if not SILENT:    
        print(f'Simulation complete. Total elapsed time: {np.round(t_it,4)} s')
    
    if SAVE:    
        if not SILENT:
            print(f'Saving to {name}.')
        
        h5f.create_dataset(f'Horizon', data=horizons, compression = 9)
        h5f.close()
        
        rundash(name, fields.N, itmax, fields.r[2:], fields.dR)

    if not SILENT:
        print('')
    
    
    
    
if __name__ == "__main__":
    params = {
        'points'   : 100,
        'radius'   : 2.5,
        'time_step': 0.00001,
        'iter'     : 100,
        'oplslic'  : 1,
        'savevar'  : 1,
        'silentvar': 0
    }
    
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

    ## This first run is needed to compile Numba
    if not params['silentvar']:
        print(f'Initializing the simulation.')
    main(N = 10, itmax = 3, SAVE = 0, SILENT = 1)
    
    main(*list(params.values()))
