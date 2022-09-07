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
data = np.zeros((len(dataH5), 8, N))
for i in range(tmax):
    data[i,:,:] = dataH5[f'{i}']
    
    
r = np.array([(j - 0.5)*dR for j in range(1,N+1)], dtype = np.float64)

# create the figure and axes objects
fig, ax = plt.subplots()


def animate(i):
	ax.clear()
	ax.set_title(f'time index = {i}')
	ax.hlines(1, r[0], r[-1], 'r', label = 'Exact solution')
	ax.plot(r, data[i,0,:], 'kx', label = 'Numerical solution') # A!
	
	ax.set_xlabel('r')
	ax.set_ylabel(r'$\tilde{A} = A / \psi^4$') 
	ax.set_ylim(0.99,1.05)
	plt.legend()
# run the animation
ani = FuncAnimation(fig, animate, frames=tmax, interval=100, repeat=True)
an_name = name[:-3]+'.gif'
ani.save(an_name)
plt.show()