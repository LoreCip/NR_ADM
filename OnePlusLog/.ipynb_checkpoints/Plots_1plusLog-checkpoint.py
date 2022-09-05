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
    
horizonData = dataH5['Horizon']

r = np.array([(j - 0.5)*dR for j in range(1,N+1)], dtype = np.float64)

# create the figure and axes objects
fig, axs = plt.subplots(3,2, constrained_layout=True)

def animate(i):
	
	for ax, ff, idx in zip(axs.flat, [r'\tilde{A}', r'\tilde{B}', 'KA', 'KB', 'alpha'], [0,1,4,5,6]):
		ax.clear()
		ax.set_title(f'{ff}')
		if ff == r'\tilde{A}' or ff == r'\tilde{B}':
			#ax.plot(r, (1 + 1/4/r)**4, 'r', label = 'Exact solution')
			ax.hlines(1, r[0], r[-1])
		ax.plot(r, data[i,idx,:], 'k', label = 'Numerical solution') # A!
		
		ax.set_xlabel('r')
		ax.set_ylim(np.min(data[i,idx,:])*0.8,np.max(data[i,idx,:])*1.4)
		ax.set_xlim(0,2.5)
		if ff != r'\tilde{A}' and ff != r'\tilde{B}':
			ax.set_ylim(min(data[i,idx,:]), max(data[i,idx,:]))

		fig.suptitle(f"Time index = {i}")
# run the animation
ani = FuncAnimation(fig, animate, frames=tmax, interval=100, repeat=True)
an_name = name[:-3]+'.gif'
ani.save(an_name)
plt.show()


fig, axs = plt.subplots(1,3, constrained_layout=True)
axs = axs.flat
axs[0].set_title(f'Horizon - Isotropic')
axs[0].plot(horizonData[:,0], horizonData[:,1])
axs[0].set_xlabel('time')

axs[1].set_title(f'Horizon - Schwarzschild')
axs[1].plot(horizonData[:,0], horizonData[:,2])
axs[1].set_xlabel('time')

axs[2].set_title(f'Surface')
axs[2].plot(horizonData[:,0], horizonData[:,3])
axs[2].set_xlabel('time')

plt.savefig(f'{name}_horizon.png')