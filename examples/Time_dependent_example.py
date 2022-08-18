
import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import matplotlib.pyplot as plt
import QNV4py as qnv

import QNV4py as qnv
 


L = 12

min_r = 0.9
max_r = 1.1
seed = 1
B_field_dir = 'z'


detuning = None #0.2
rabi_freq = np.pi/2
noise = 0.05
AC_function = None 

steps = 1000


# building blocks of the drive
kick_building_blocks = [ [[('dd',0.2)],1],  [[('x',0.9)],1]  ]

# define a function that describes the time dependence of the kick time
# here we chose a linear ramp
def function(init_val, n, steps):
   return init_val + 2*n / steps


# NV_sysgem objcet
c13_spins = qnv.NV_system.default(L)

# NV_dynamics object
c13_dynamics = qnv.NV_dynamics(c13_spins,rabi_freq,kick_building_blocks,noise=noise)

# define some single particle observables to be measured
observables = c13_spins.SP_observable(['x'])

# define an initial state
psi_i = c13_spins.initial_state('x')

# some extra metadata that is stored with the data
description = {'description':"We store x,y and z magentization,the sequences is given by\
                              [[('x',0.5),('dd',1.0),('y',0.4)],4], [[('z',0.5)],2]",
               'comment':'this is the first data set of the series'}
                


# time evolution
observables, times = c13_dynamics.evolve_time_dependent(psi_i,
                                                   steps,
                                                   observables,
                                                   [None,[function,steps]],
                                                   'example_file',
                                                   save_every=50,
                                                   folder='example_data_time_dep',
                                                   extra_save_parameters=description)



# plot the results
plt.plot(times,observables[0],'o-')
plt.xscale('log')
plt.show()






