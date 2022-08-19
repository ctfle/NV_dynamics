import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import matplotlib.pyplot as plt
import h5py
#import time 
#import matplotlib.colors as colors
#import matplotlib as mpl

#from quspin.operators import hamiltonian
#from quspin.basis import spin_basis_1d
#from quspin.tools.evolution import expm_multiply_parallel



import QNV4py as qnv
 


L = 12

min_r = 0.9
max_r = 1.1
seed = 1
B_field_dir = 'z'


detuning = None #0.2
rabi_freq = np.pi/2
noise = 0.05



#c13_spins = qnv.NV_system(B_field_dir,L,min_r,max_r,seed,scaling_factor=0.1)

c13_spins = qnv.NV_system.default(L)

 
def func(x,w):
   return np.sin(x*w)
       
AC_function =  [func,2*np.pi/(0.5+0.2+0.1)]
 
n1=1
n2=1
kick_building_blocks = [ [[('dd',0.2),('x',0.5)],50], [[('z',1.0)],1]   ]

# NV_dynamics object
c13_dynamics = qnv.NV_dynamics(c13_spins,rabi_freq,kick_building_blocks,detuning,AC_function,noise)

# define some single particle observables to be measured
observables = c13_spins.SP_observable(['x'])

# define an initial state
psi_i = c13_spins.initial_state('x')



'''
basis = c13_dynamics.basis
initial_state = np.zeros(basis.Ns).astype(np.complex128)
initial_state[basis.index('1'*L)]=1
work_array=np.zeros((2*len(initial_state),), dtype=initial_state.dtype)
Oy=hamiltonian([['y',[[1.0,j] for j in range(L)] ],],[],basis=basis,dtype=np.complex128,check_symm=False,check_herm=False)   
rotation = expm_multiply_parallel(Oy.tocsr(),a=-1j*np.pi*0.25)
rotation.dot(initial_state,work_array=work_array,overwrite_v=True)
'''
description = {'description':"We store x,y and z magentization,the sequences is given by\
                              [[('x',0.5),('dd',1.0),('y',0.4)],4], [[('z',0.5)],2]",
               'comment':'this is the first data set of the series'}
                

steps = 20
'''
observables, times = c13_dynamics.evolve_random(psi_i,
                                                   steps,
                                                   observables,
                                                   'example_file',
                                                   save_every=10,
                                                   group_name='random_drive',
                                                   extra_save_parameters=description)


observables, times = c13_dynamics.evolve_periodic(psi_i,
                                                   steps,
                                                   observables,
                                                   'example_file',
                                                   save_every=10,
                                                   group_name='example_data',
                                                   extra_save_parameters=description)



load_dir = './data/'
filename = 'example_file.hdf5'
data_set_name='random_drive'
#
with h5py.File(load_dir + filename,'r') as h5f:
   times = h5f[data_set_name +'/'+ 'times'][:]
   observables= h5f[data_set_name +'/'+ 'observables'][:]
#  

print(observables)

'''



def fibonacci(n):
   if n > 1:
      return fibonacci(n-1)+fibonacci(n-2)
   elif n==1:
      return [1]
   elif n==0:
      return [0]


sequence = fibonacci(10) # 987 elements
steps = len(sequence)
observables, times = c13_dynamics.evolve_sequential(psi_i,
                                                   steps,
                                                   observables,
                                                   sequence,
                                                   'example_file',
                                                   save_every=10,
                                                   group_name='fibonacci',
                                                   extra_save_parameters=description)


load_dir = './data/'
filename = 'example_file.hdf5'
data_set_name='fibonacci'
#
with h5py.File(load_dir + filename,'r') as h5f:
   times = h5f[data_set_name +'/'+ 'times'][:]
   observables= h5f[data_set_name +'/'+ 'observables'][:]
#  

print(observables)


plt.plot(times, observables[0],'-')
plt.xscale('log')
plt.show()




