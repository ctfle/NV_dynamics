##  
# @page example_load Load HDF5 files
# 
# @subsection example_load Save and load HDF5 files
# For a detailed documentation see https://docs.h5py.org/en/stable/quick.html
# A very basic example how to load files generated with QNV4py is given below
# 
# Lets generate some data
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
# # kick sequence with 50 x kicks of pi/2 followed by a single z kick with pi
# kick_building_blocks = [ [[('dd',0.2),('x',0.5)],50], [[('z',1.0)],1]   ]
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
# # evolve periodically
# observables, times = c13_dynamics.evolve_periodic(psi_i,
#                                                    steps,
#                                                    observables,
#                                                    'example_file',
#                                                    save_every=50,
#                                                    folder='example_data',
#                                                    extra_save_parameters=description)
#                                
# ~~~~~~~~~~~~~
#
# Load and plot it
# ~~~~~~~~~~~~~{.py}
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# 
# load_dir = './data/'
# filename = 'example_file.hdf5'
# data_set_name='random_drive'
#
# with h5py.File(load_dir + filename,'r') as h5f:
# 	times = h5f[data_set_name +'/'+ 'times'][:]
# 	observables= h5f[data_set_name +'/'+ 'observables'][:]
#  
# plt.plot(times,observables[0])
# plt.show()
# ~~~~~~~~~~~~~
#
#
