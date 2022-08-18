##  
# @page sequence_example Example code for a user defined drive
# 
# @subsection example4 Prototype example code for a user define sequence
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
# #define a function to generarte the Fibonacci sequence
# def fibonacci(n):
#    if n > 1:
#       return fibonacci(n-1)+fibonacci(n-2)
#    elif n==1:
#       return [1]
#    elif n==0:
#       return [0]
# 
# 
# sequence = fibonacci(10) 
# steps = len(sequence)
# 
# # whenever the sequence returns 0 at a given step the 0th block is of kick_building_blocks is applied
# # whereas whenever sequence returns 1 the first block is applied.
# # The same logic applies with n blocks. 
# observables, times = c13_dynamics.evolve_sequential(psi_i,
#                                                    steps,
#                                                    observables,
#                                                    sequence,
#                                                    'example_file',
#                                                    save_every=50,
#                                                    folder='example_data_seq',
#                                                    extra_save_parameters=description)
#
# # plot the results
# plt.plot(times, observables[0],'-')
# plt.xscale('log')
# plt.show()s
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
                                                   save_every=50,
                                                   folder='example_data_seq',
                                                   extra_save_parameters=description)


 
plt.plot(times, observables[0],'-')
plt.xscale('log')
plt.show()




