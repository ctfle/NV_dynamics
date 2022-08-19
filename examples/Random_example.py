##  
# @page random_example Example code for a random drive
# 
# @subsection example3 Prototype example code for a random drive
#
# ~~~~~~~~~~~~~{.py}
# import os, sys
# os.environ['OMP_NUM_THREADS'] = '4'
# import numpy as np
# import matplotlib.pyplot as plt
# import QNV4py as qnv
#
# # system parameters
# L = 12
# AC_function= None
# detuning = None
# rabi_freq = np.pi/2
# noise = 0.05
#
# c13_spins = qnv.NV_system.default(L)
#
# # kick sequence: eventually we pick either of the two blocks randomly for each step in the time evolution 
# kick_building_blocks = [ [[('dd',0.2),('x',0.5)],1], [[('dd',0.2),('y',0.5)],1 ]  ]
#
# # NV_dynmaics object 
# c13_dynamics = qnv.NV_dynamics(c13_spins,rabi_freq,kick_building_blocks,detuning=detuning,AC_function=AC_function,noise=noise)
#
# # define some single particle observables to be measured
# observables = c13_spins.SP_observable(['x'])
#
# # define an initial state
# psi_i = c13_spins.initial_state('x')
# 
# description = {'description':"We store x,y and z magentization,the sequences is given by\
#                               [[('x',0.5),('dd',1.0),('y',0.4)],4], [[('z',0.5)],2]",
#                'comment':'this is the first data set of the series'}
#
#
# steps = 200
# # Evolve random: for each step one of the two giben blocks is applied randomly. 
# # The same logic applies when n blocks are given. 
# # To trace the random sequence set the variable seed_random_seq.
# observables, times = c13_dynamics.evolve_random(psi_i,
#                                                    steps,
#                                                    observables,
#                                                    'example_file',
#                                                    save_every=50,
#                                                    folder='example_data_random',
#                                                    extra_save_parameters=description
#                                                    seed_random_seq=2)
#                                                                                                   
# # plot the results                                              
# plt.plot(observables[0],'-')
# plt.xscale('log')
# plt.show()
# ~~~~~~~~~~~~~



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



#c13_spins = qnv.NV_system(B_field_dir,L,min_r,max_r,seed,scaling_factor=0.1)

c13_spins = qnv.NV_system.default(L)

 
def func(x,w):
   return np.sin(x*w)
       
AC_function =  [func,2*np.pi/(0.5+0.2+0.1)]
 
kick_building_blocks = [ [[('dd',0.2),('x',0.5)],1], [[('dd',0.2),('x',0.5)],1 ]  ]

# NV_dynamics object
c13_dynamics = qnv.NV_dynamics(c13_spins,rabi_freq,kick_building_blocks,detuning,AC_function,noise)

# define some single particle observables to be measured
observables = c13_spins.SP_observable(['x'])

# define an initial state
psi_i = c13_spins.initial_state('x')

description = {'description':"We store x,y and z magentization,the sequences is given by\
                              [[('x',0.5),('dd',1.0),('y',0.4)],4], [[('z',0.5)],2]",
               'comment':'this is the first data set of the series'}
                
steps = 200
observables, times = c13_dynamics.evolve_random(psi_i,
                                                   steps,
                                                   observables,
                                                   'example_file',
                                                   save_every=50,
                                                   folder='example_data_random',
                                                   extra_save_parameters=description)




# plot the results
plt.plot(times,observables[0],'-')
plt.xscale('log')
plt.show()






