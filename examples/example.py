##  
# @page examples_page Basic example for QNV4py
# 
# @subsection example1 The following code serves as a protptype example
# @subsubsection importing Import necessary packages 
# and set relevant environment variables
#
# ~~~~~~~~~~~~~{.py}
# import os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# 
# import QNV4py as qnv
# ~~~~~~~~~~~~~
#
# @subsubsection system_param Create a NV_system object
#
# ~~~~~~~~~~~~~{.py}
# L = 12
# min_r = 0.9
# max_r = 1.1
# seed = 1
# B_field_dir = 'z'
# 
# c13_spins = qnv.NV_system(B_field_dir,L,min_r,max_r,seed)
#
# ~~~~~~~~~~~~~
#
# alternaatively you can use default settings and just specify the system size
# 
# ~~~~~~~~~~~~~{.py}
# c13_spins = qnv.NV_system.default(L)
# ~~~~~~~~~~~~~
#
# @subsubsection create Create a NV_dynamics object
# 
# To setup a NV_dynamcis object we first specfiy the parameters 
# 
# ~~~~~~~~~~~~~{.py}
# detuning = 1.0
# rabi_freq = 0.5
# noise = 0.05
# ~~~~~~~~~~~~~
#
# as well as a AC function 
# 
# ~~~~~~~~~~~~~{.py}
# def func(x,w):
# 	return np.sin(x*w)
#		
# AC_function = [func,2*np.pi/(0.5+1.0+0.4)]
# ~~~~~~~~~~~~~
#
# Next we specif the specific driving seqeunce we want to simulate
# 
# ~~~~~~~~~~~~~{.py}
# n1=4
# n2=2
# kick_building_blocks = [ [[('x',0.5),('dd',1.0),('y',0.4)],n1], [[('z',0.5)],n2] ]
# ~~~~~~~~~~~~~
#
# kick_building_blocks conatins all information on the sequence: a 'x' kick with time 0.5 (in units of the energy scale J)
# is followed by an evolution with the dipolar hamiltonian (signaled by 'dd') with time 1.0 and a 'y' kick with time 0.4. 
# The whole sequence of 'x' 'dd' 'y' is repeated n1 times. After that a 'z' kick is applied n2 times. The whole sequence defines a single Floquet cycle.
# 
# ~~~~~~~~~~~~~{.py}
# c13_dynamics = qnv.NV_dynamics(c13_spins,rabi_freq,kick_building_blocks,detuning,AC_function,noise)
# ~~~~~~~~~~~~~
# 
# Next we define observables 
# 
# ~~~~~~~~~~~~~{.py}
# observables = c13_dynamics.SP_observable(['x','y','z'])
# ~~~~~~~~~~~~~
#
# One can also write them directly using QuSPin hamiltonian
# 
# Finally, we define the initial state and the number of Floquet steps, the 
# 
# ~~~~~~~~~~~~~{.py}
# basis = c13_dynamics.basis
# initial_state = np.zeros(basis.Ns)
# initial_state[basis.index('1'*L)]=1
# 
# steps = 10000
# observables, times = c13_dynamics.evolve_periodic(initial_state,steps,observables,'example_file',save_every=50,folder='example_data')
# ~~~~~~~~~~~~~
# 
# @subsection comp-file Complete code
# 
# ~~~~~~~~~~~~~{.py}
# import os, sys
# os.environ['OMP_NUM_THREADS'] = '4'
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# import QNV4py as qnv
#
# 
# L = 12
# 
# detuning = 1.0
# rabi_freq = 0.5
# noise = 0.05
# 
# def func(x,w):
#    return np.sin(x*w)
#     
# AC_function = [func,2*np.pi/(0.5+1.0+0.4)]
# 
# n1=4
# n2=2
# kick_building_blocks = [ [[('x',0.5),('dd',1.0),('y',0.4)],n1], [[('z',0.5)],n2] ]
# 
# 
# c13_spins = qnv.NV_system.default(L)
# c13_dynamics = qnv.NV_dynamics(c13_spins,rabi_freq,kick_building_blocks,detuning,AC_function,noise)
# 
# 
# observables = c13_spins.SP_observable(['x','y','z'])
#
# psi_i = c13_spins.initial_state('x')
# 
# steps = 1000
# 
#
# description = {'description':"We store x,y and z magentization,the sequences is given by\
#                   [[('x',0.5),('dd',1.0),('y',0.4)],4], [[('z',0.5)],2]",
#                'comment':'this is the first data set of the series',
#                'some more info':'This is very important info'}
#               
# observables, times = c13_dynamics.evolve_periodic(psi_i,
#                                                     steps,
#                                                     observables,
#                                                     'example_file',
#                                                     save_every=50,
#                                                     folder='example_data',
#                                                     extra_save_parameters=description)
#
# plt.plot(times,observables[0])
# plt.xscale('log')
# plt.show()
# ~~~~~~~~~~~~~


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