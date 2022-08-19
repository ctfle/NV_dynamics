import sys,os
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel
from scipy.sparse.linalg import eigsh
import numpy as np 
import time
import scipy.integrate as integrate
import contextlib
import h5py

#from helper_funcs import *


from QNV4py import Helper_funcs

hlp = Helper_funcs()



## @mainpage Quantum Dynamics in Dipolar Coupled Nuclear Spins
#  
#  @brief Python package for (periodically) driven dipolar coupled \f$ C^{13} \f$ nuclear spins. 
#  The package includes the class NV_system to create and design a system of dipolar coupled \f$ C^{13} \f$ spins
#  centered around a central NV center, and the class NV_dynamics for the time evolution of pure initial states of NV_system objects.
#  
# @subsection notes_main To be done
# - function to construct different initial states?
# - function to construct observables?
# - make it work on windows
# 
# @subsection intsall Installation via pip
# 
# Download the master branch from https://github.com/ctfle/NV_dynamics and un-zip. Open a terminal window and navigate into the folder QNV4py. Type
#
# Open a terminal window and navigate into the folder QNV4py. Type
#  ~~~~~~~~~~~~~{.py} 
#  pip install . 
#  ~~~~~~~~~~~~~
#
# @subsection notes_packages Required packages
# - numpy, scipy 
# - h5py, python package for .hdf5 data files (https://www.hdfgroup.org/solutions/hdf5/)
# - Quspin, for the construction of observables and Hamiltonians (https://weinbe58.github.io/QuSpin/)
# 
# 
# @subsection notes_examples How to use the code:
# 
# 
# Check out the example files (see sidebar)!
# 
# The code consists of two classes: NV_system, to construct a random graph of \f$ C^{13} \f$ atoms, 
# and NV_dynamics to evolve in time with user defined sequence. To setup a working code, first you have to 
# construct a NV_system object. Then, use it to build a NV_dynamics object. NV_system can be used with default parameter settings
# (for example  <code> C13_object = NV_system.default(L) </code>), which builds the random graph with
# default settings. An NV_dynamics object requires the following input: 
# 	- <code> nv_instance </code>, a NV_system object
# 	- <code> rabi_freq </code>, the amplitude of the kicks
# 	- <code> kick_building_blocks </code>, the elementary building blocks of the drive, for instance
# 	- <code> detuning=None </code>, Detuning (left over single particle field in the rotating frame, Default None)
# 	- <code> AC_function=None </code>, a (continous) AC field given as an arbitrary function, Default None
# 	- <code> noise=None </code>, some noise to increase ergodicity, Default None
# 
# <code> kick_building_blocks </code> as well as AC_function have to be provided in a special list format: <code>  [block1, block2, ...] </code>, 
# where each block is a list itself. For instance  <code> block1 = [[('dd',0.2),('x',0.1)],50] </code>. 
# 	 <code> block2 = [[('z',1.0)],1] </code>. The first elememt of this list is a list of tuples defining the sequence. 
# 	 In the example, <code> block1 </code> describes a sequence where the dipolar Hamiltonian (indicated by <code> 'dd' </code> ) 
# 	 is applied for a time of 0.2 (in units of the internally estimated inverse energy scale of the system) followed by a kick <code> 'x' </code> amplitude
# 	 <code> rabi_freq </code> and time 0.1. This protocol is repeated 50 times.
# 	 
# <code> AC_function </code> has to be provided as a list containing a defined function that returns a single value and some additional input parameters.
# For isntance, to feed in the function <code> func(x,param1,param2) </code>, we set <code> AC_function = [func,param1,param2] </code>.
# 
# 
# After initializing the a NV_dynamics object, you can evolve a given initial state in time using either of (so far)
# implemented methods
# - <code> evolve_periodic(initial_state,n_steps,observable,file_name,save_every=1000,save_dir='./data/',folder='new_data_set',extra_save_parameters=None,seed=1) </code> 
# 			eveolves a given initial state periodically using all given blocks as they appear  <code> kick_building_blocks </code>, 
# 			i.e.  <code> kick_building_blocks </code> defines a single Floquet period. (see Example code for a DTC)
# 
# - <code> evolve_random(initial_state,n_steps,observable,file_name,save_every=1000,save_dir='./data/',folder='new_data_set',extra_save_parameters=None,seed=1,seed_random_seq=2) </code> 
# 			evolves the initial state by chosing randomly one block out of <code> kick_building_blocks </code> for a given step.
# 			<code> seed_random_seq </code> can be specified to trace random numbers. (see Example code for a random drive)
# 
# - <code> evolve_sequential(initial_state,n_steps,observable,sequence,file_name,save_every=1000,save_dir='./data/',folder='new_data_set',extra_save_parameters=None,seed=1) </code> 
# 			Can be used to drive the initial state with s specific <code> sequence </code>. At each step the sequence element specifies which block out of <code> kick_building_blocks </code> is to be applied.
# 			(see Example code for sequential drive)
# - <code> evolve_time_dependent(initial_state,n_steps,observable,discrete_functions,file_name,save_every=1000,save_dir='./data/',folder='new_data_set',extra_save_parameters=None,seed=1,seed_random_seq=2) </code> 
# 			Evolves the system with time dependent <code> kick_building_blocks </code>. <code> discrete_functions </code> is used to specify the time dependence of each block in <code> kick_building_blocks </code>.
# 			<code> discrete_functions </code> is provided in list for (similar to <code> AC_function </code>), for instance <code> discrete_functions =[None,[func,some_param]]</code>. 
# 			The latter input attributes no time dependence to the first block in <code> kick_building_blocks </code> but adds some time dependence given by a the function <code> func(n,some_param) </code> 
# 			where the function input n specifies the evolution time step. (see Example code for a time dependent drive)
#
# Any of the above functions evaluates the given observables whenever only the dipolar Hamiltonian is applied.
# The results (measurement times and observable values) are stored in HDF5 data format in a file <code> save_dir + file_name </code>. 
# HDF5 stand fo hirachical data format and allows internal directory structures. 
# Therefore, several datasets can be stored in the same file by specifying <code> folder </code> which creates folder within the HDF5 file.
# Note that the applied save function automatically takes care of duplicate folder names and avoids overwriting. 
# 
# Another nice feature of HDF5 files is that one can attach metadata to datasets, hence there is no need to specify all parameters in the file-/folder-name 
#
#
# @file nv_system.py Contains the class NV_system
# 




class NV_system():
	
	"""! Sets up a random graph of L spins where each spin has a min_dist to all other spins
		and is at least connected to one other spin at no further than max_dist """
	
	def __init__(self,B_field_dir,L,min_dist,max_dist,seed,scaling_factor=0.1):

        ## Basic constructor. 
        #
        # @param seed seed value for the generation of random numbers during the build of the random graph of \f$ C^{13} \f$ spins
        # @param B_field_dir string, direction of the external B-field 
        # @param min_dist minimum distance allowed between two \f$ C^{13} \f$ spins
        # @param max_dist maximum nearest neighbor distance between \f$ C^{13} \f$ spins
        # @param scaling_factor parameter to scale the importance of single particle terms due to the field generated from the NV center. Default is 0.1
        # @param L system size
        # @param spin_positions Positions of spins on the random graph
        # @param basis QuSpin basis object 
        # @param energy_scale energy scale J of random graph of \$ C^{13} \f$ spins (without single particle terms! Those are normalized with J and scaled with scaling_factor).
        # 		Computed from the free induction decay of an initially \f$ \hat{x} \f$-polarized (pure) state.
        # @param H_dd dipolar Hamiltonian corresponding to the random graph (including single particle terms)


		interactions_x_y, interactions_z, spin_positions, couplings = hlp.sampling_points(B_field_dir,min_dist,max_dist,L,seed)

		self.__name = '{} nuclear spins randomly placed around a NV center'.format(L)


		#graph parameters
		##seed value for the generation of random numbers during the build of the random graph of \f$ C^{13} \f$ spins
		self.seed =seed
		
		## B_field_dir string, direction of the external B-field 
		self.B_field_dir = B_field_dir

		## min_dist minimum distance allowed between two \f$ C^{13} \f$ spins
		self.min_dist = min_dist
		
		## max_dist maximum nearest neighbor distance between \f$ C^{13} \f$ spins
		self.max_dist = max_dist
		
		## scaling_factor parameter to scale the importance of single particle terms due to the field generated from the NV center, default is 1.0
		self.scaling_factor = scaling_factor

		self.__interactions_x_y = interactions_x_y
		self.__interactions_z = interactions_z
		self.__couplings = couplings

		## system size 
		self.L = L

		## spin_positions Positions of spins on the random graph
		self.spin_positions = spin_positions
		

		# dipolar Hamiltonian and energy scale
		
		## QuSpin basis object 
		self.basis = spin_basis_1d(L=self.L,pauli=True)
		interactions = [['xx',self.__interactions_x_y],['yy',self.__interactions_x_y],['zz',self.__interactions_z]]
		H_dd = hlp.construct_Hamiltonian(self.basis,interactions)
		
		#estimate relevant energy scales (without disordered single particle fields)
		
		##energy scale J of random graph of \f$C^{13}\f$ spins (without single particle terms! Those are normalized with J and scaled with scaling_factor).
        # Computed from the free induction decay of an initially \f$ \hat{x} \f$-polarized (pure) state.
		self.energy_scale = hlp.estimate_scales(self.basis,self.L,H_dd,delta_t=0.0005,time_steps=1000)

		#rescale interactions in units of the energy_scale


		#add disordered single particle fields normalized by the energy_scale
		self.__z_field = hlp.compute_single_particle_fields(self.spin_positions,self.energy_scale,B_field_dir,scaling_factor=self.scaling_factor)
		
		#build the dipolar Hamiltonian and rescale in units of the energy_scale

		## dipolar Hamiltonian corresponding to the random graph (including single particle terms)
		self.H_dd = hlp.construct_Hamiltonian(self.basis,interactions + [['z',self.__z_field]]).tocsr()/self.energy_scale
	

	@classmethod	
	def default(cls,L):
		
		"""! Classmethod to initialize the system with default parameters: 
				B_field_dir='z'
				min_dist=0.9
				max_dist=1.1
				seed=1
				scaling_factor=1.0
		"""

		B_field_dir='z'
		min_dist=0.9
		max_dist=1.1
		seed=1
		scaling_factor=0.1
		print('\nInitializing NV_system object with default parameters:\n')
		print("B_field_dir='z'\nseed={0:d}\nmin_dist={1:0.1f}\nmax_dist={2:0.1f}\nscaling_factor=1.0\n".format(1,0.9,1.1))
		return cls(B_field_dir,L,min_dist,max_dist,seed,scaling_factor=scaling_factor)



	def __str__(self):
		"""! Print function """
		print('\nSampled spin postions are:\n\n')
		for spin_pos in self.spin_positions:
			print(spin_pos)
		print('\n')
		print('seed of the graph: {0:d} \n'.format(self.seed))
		print('dynamically extracted energy scale J={0:0.3f} \n'.format(self.energy_scale))

		if hlp.yes_no('Print out interaction couplings? '):
			
			print('xx couplings:\n')
			print(self.__interactions_x_y,'\n')

			print('yy couplings:\n')
			print(self.__interactions_x_y,'\n')

			print('zz couplings:\n')
			print(self.__interactions_z,'\n')

			print('onsite z couplings:\n')
			print(self.__z_field,'\n')

		return self.__name


	def initial_state(self,direction):
		"""! Construct an initial state polarized along direction"""

		initial_state = np.zeros(self.basis.Ns).astype(np.complex128)
		initial_state[self.basis.index('1'*self.L)]=1

		if direction=='z':
			return initial_state

		
		elif direction=='y':
			# rotate around x by -pi/2			
			work_array=np.zeros((2*len(initial_state),), dtype=initial_state.dtype)
			Ox=hamiltonian([['x',[[1.0,j] for j in range(self.L)] ],],[],basis=self.basis,dtype=np.complex128,check_symm=False,check_herm=False)	
			rotation = expm_multiply_parallel(Ox.tocsr(),a=1j*np.pi*0.25)
			rotation.dot(initial_state,work_array=work_array,overwrite_v=True)
			return initial_state

		elif direction=='x':
			#rotate around y by pi/2
			work_array=np.zeros((2*len(initial_state),), dtype=initial_state .dtype)
			Oy=hamiltonian([['y',[[1.0,j] for j in range(self.L)] ],],[],basis=self.basis,dtype=np.complex128,check_symm=False,check_herm=False)	
			rotation = expm_multiply_parallel(Oy.tocsr(),a=-1j*np.pi*0.25)
			rotation.dot(initial_state,work_array=work_array,overwrite_v=True)
			return initial_state

		else:
			raise AssertionError ("input not understood: direction should be 'x', 'y' or 'z'")



	def SP_observable(self,directions):
		"""!Construct single partcile observables according to directions"""
		##
		# @param directions list of char. Char must be 'x', 'y', 'z'
		#
		
		observables = []
		for char in directions:
			assert char in ['x','y','z'], 'input not understood' 

			O=hamiltonian([[char,[[1.0,j] for j in range(self.L)] ],],[],
							basis=self.basis,dtype=np.complex128,check_symm=False,check_herm=False)	
			
			observables += [O]	

		return observables


	def spectrum(self):
		"""! Computes the spectrum of H_dd"""
		## @return eigenvalues and eigenvectors of H_dd
		e,v =np.linalg.eigh(self.H_dd.toarray())
		return e, v



