
import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
 
import numpy as np
import matplotlib.pyplot as plt
import time 
import matplotlib.colors as colors
import matplotlib as mpl
 
import QNV4py as qnv
 
L = 12
min_r = 0.9
max_r = 1.1
seed = 1
B_field_dir = 'z'
detuning = 1.0
rabi_freq = 0.5
noise = 0.05
 
c13_spins = qnv.NV_system(B_field_dir,L,min_r,max_r,seed)
c13_spins = qnv.NV_system.default(L)
 
 
def func(x,w):
   return np.sin(x*w)
       
AC_function = [func,2*np.pi/(0.5+1.0+0.4)]
 
n1=4
n2=2
kick_building_blocks = [ [[('x',0.5),('dd',1.0),('y',0.4)],n1], [[('z',0.5)],n2] ]
 
c13_dynamics = qnv.NV_dynamics(c13_spins,rabi_freq,kick_building_blocks,detuning,AC_function,noise)
 
observables = c13_dynamics.SP_observable(['x','y','z'])

basis = c13_dynamics.basis
initial_state = np.zeros(basis.Ns)
initial_state[basis.index('1'*L)]=1
 
steps = 10000
observables, times = c13_dynamics.evolve_periodic(initial_state,steps,observables,'example_file',save_every=50,folder='example_data')
 
plt.plot(times,observables[0])
plt.xscale('log')
plt.show()