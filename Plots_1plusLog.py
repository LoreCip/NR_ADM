import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

import h5py
import sys

name = sys.argv[1]

N = int(sys.argv[2])
tmax = int(sys.argv[3])
dR = float(sys.argv[4])

dataH5 = h5py.File(name,'r')
data = np.zeros((len(dataH5), 7, N))
for i in range(tmax):
    data[i,:,:] = dataH5[f'{i}']
    
    
r = np.array([(j - 0.5)*dR for j in range(1,N+1)], dtype = np.float64)

# create the figure and axes objects
fig, axs = plt.subplots(3,2, constrained_layout=True)


def animate(i):
	
	for ax, ff, idx in zip(axs.flat, ['A', 'B', 'KA', 'KB', 'alpha'], [0,1,4,5,6]):
		ax.clear()
		ax.set_title(f'Field = {ff}, time index = {i}')
		if ff == 'A' or ff == 'B':
			ax.plot(r, (1 + 1/4/r)**4, 'r', label = 'Exact solution')
		ax.plot(r, data[i,idx,:], 'k', label = 'Numerical solution') # A!
	
		ax.set_xlabel('r')
		ax.set_ylim(np.min(data[i,idx,:])*0.8,np.max(data[i,idx,:])*1.4)
		ax.set_xlim(0,2.5)
		#plt.legend()
# run the animation
ani = FuncAnimation(fig, animate, frames=tmax, interval=100, repeat=True)
an_name = name[:-3]+'.gif'
ani.save(an_name)
plt.show()