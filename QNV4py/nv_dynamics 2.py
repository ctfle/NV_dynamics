import sys,os
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel
from scipy.sparse.linalg import eigsh
import numpy as np 
import pickle
import scipy.integrate as integrate
import contextlib
from scipy.linalg import logm, expm
import h5py
import copy
#from helper_funcs import *
#from nv_system import *


from QNV4py import Helper_funcs
from QNV4py import NV_system

hlp = Helper_funcs()


##
# @file nv_system.py Contains the class NV_dynamics
# 



class NV_dynamics(NV_system):
	"""! Computes (Floquet) dynamics generated from the input sequence of kicks """
	

	def __init__(self,nv_instance,rabi_freq,kick_building_blocks,detuning=None,AC_function=None,noise=None):
		#self,kick_seq,RK=False,*system_params):
		#parameters = {param: getattr(nv_instance, param) for param in dir(nv_instance) if not param.startswith("__")} 
		super().__init__(nv_instance.B_field_dir,nv_instance.L, nv_instance.min_dist,nv_instance.max_dist,nv_instance.seed,scaling_factor=nv_instance.scaling_factor)

		## Detuning. Default is None
		self.detuning = detuning
		
		## Rabi frequency
		self.rabi_freq = rabi_freq
		
		## noise attributed to the drive. E.g. 0.05 corresponds to max 5% variataion every time the dipolar Hamiltonian is applied during time evolution
		self.noise = noise

		## AC function. Must be list e.g. [function, parameter1, parameter2, parameter3]. 
		# Then, during time evolution f(x,parameter1,parameter2,parameter3) will be used (and integrated over x).
		# Applied to each block of the sequence (see building_blocks) separately.
		self.AC_function =AC_function
		
		#check if kick_building_block is of right form
		if type(kick_building_blocks)!=list:
			raise AssertionError ('kick_squence must be  of type list')
		for b,block_element in enumerate(kick_building_blocks):
			# block_element -> [[(),(),()...],n_times]
			assert type(block_element) == list , 'block_element must be of the form [[(),(),...],n_times]'

			for e,element in enumerate(block_element[0]):
				if type(element)!=tuple:
					raise AssertionError ('element in block_element[0] must be given as list of tuples')
				elif len(element)!=2:
					raise AssertionError ('element in block_element[0] has more than 2 entries')
				elif element[0] not in ['x','y','z','dd']:
					raise AssertionError ('input in kick_sequence element number {0:d} not understood'.format(e))
			
			assert type(block_element[1]==int), 'building blocks not unterstood'
		
		
		# initialize the kick sequence according to the specific case under consideration
		# building blocks [[building_block1],[building_block2]] with building_block1 = [[(),(),()...],n_times],
		# set up list of exponentials according to the building blocks
		
		## building blocks of the seqeunces to be applied
		self.building_blocks = hlp.setup_expH(self.L,self.basis,self.H_dd,
			kick_building_blocks,rabi_freq,detuning,self.AC_function,self.noise)


		#print(self.building_blocks)
		

	def sequence_elements(self):
		nr_of_seq=[]
		for block in self.building_blocks:
			nr_of_seq += [len(block[0])*block[1]]
		return nr_of_seq
	

	def data_points(self):
		nr_of_data=[]
		
		for block in self.building_blocks:
			points_per_block =0
			for element in block[0]:
				if element[0]=='dd':
					points_per_block += 1
			points_per_block *= block[1]
			nr_of_data += [points_per_block]
		return nr_of_data



	def save_data(self,data,file_name,save_dir,folder,data_name,
					overwrite=False,extra_save_parameters=None):

		"""! Method to save data in hdf5 format. Called during evolve_periodic, evolve_random and evolve_sequential.
		 All relevant parameters of the system are stored automaticlly within the data set and do not need to be specified explicitly in e.g. the file name."""

		##
		# @param data input data to be stored in hdf5 format
		# @param file_name filename (without ending) of the resulting file
		# @param save_dir directory where to store the data. Generated if not existent
		# @param folder Folder within the hdf5 file where to store the data
		# @param data_name name of the data set that is stored in the hdf5 file (hdf stands for hirachical data format).
		# One can make use of the hirachical structre by adding '/' to the data_name. Then, interanlly a folder structre will be generated in the .hdf5 file.
		# If data_name is not specfified, the default value 'new_data_set' wil be used
		# @param overwrite bool, if False, the method checks for existing files (and data_sets within the files) and avoids overwriting by creating adding a new data set in the correspoinding file 
		# (instead of overwriting existing ones). If True, no checks are applied. Default is False.
		# @param extra_save_parameters add extra parameters/description of the data stored in the corresponding data set of the .hdf5 file. Must be dict or None. For instance {'description':'some description of the data'}
		#
		# @return data_group



		assert isinstance(data,np.ndarray), 'data must be given as np.array'
		
		if os.path.exists(save_dir + file_name + ".hdf5") and overwrite==False:
			#check if data_set_name exists already

			#check for data
			file = h5py.File(save_dir + file_name + ".hdf5", 'r')
			folder_old = folder		

			j=0

			while file.get(folder + '/' + data_name)!=None: #data_name in file:
				print('data set exist already')
				#change data set name
				if j==0:
					folder += str(j)	
				else:
					folder = folder_old + str(j)
				print('try with group name:', folder)
				j +=1
			
			file.close()

			if folder != folder_old:
				print("\ndata group_name '" + folder_old + "' exists already in " +save_dir + file_name + ".hdf5\n" " group  has been changed to " + folder + "\n")

			else:
				print('group_name is ok. No changes applied')

			with h5py.File(save_dir + file_name + ".hdf5", 'a') as file:
				dset = file.create_dataset(folder + '/' + data_name,data=data)

				#store all relevant parameters as attributes
				dset.attrs['system_size']=self.L
				dset.attrs['seed_NV_system']=self.seed
				dset.attrs['B_field_dir_NV_system']=self.B_field_dir
				dset.attrs['rmin_NV_system']=self.min_dist
				dset.attrs['rmax_NV_system']=self.max_dist
				dset.attrs['scaling_factor_NV_system']=self.scaling_factor
				if self.detuning!=None:
					dset.attrs['detuning'] = self.detuning
				else:
					dset.attrs['detuning'] = 'None'
				
				dset.attrs['rabi_freq'] = self.rabi_freq
				if self.noise!=None:
					dset.attrs['noise'] = self.noise
				else:
					dset.attrs['noise'] = 'None'
				
				#extra parameters
				if extra_save_parameters!=None:
					keys=list(extra_save_parameters.keys())
					for key in keys:
						dset.attrs[key]=extra_save_parameters[key]
			print(' === data saved === ')

			
		elif os.path.exists(save_dir + file_name + ".hdf5") and overwrite==True:
			
			print('\noverwriting ' + folder + '/' + data_name + ' in file '+ save_dir + file_name + ".hdf5\n")
			
			with h5py.File(save_dir + file_name + ".hdf5", "r+") as file:
				if folder + '/'+ data_name in file:
					del file[folder + '/'+ data_name] 

				dset = file.create_dataset(folder + '/'+ data_name,data=data)
				
				#store all relevant parameters as attributes
				#graph parameters
				dset.attrs['system_size']=self.L
				dset.attrs['seed_NV_system']=self.seed
				dset.attrs['B_field_dir_NV_system']=self.B_field_dir
				dset.attrs['rmin_NV_system']=self.min_dist
				dset.attrs['rmax_NV_system']=self.max_dist
				dset.attrs['scaling_factor_NV_system']=self.scaling_factor

				#dynamic parameters
				if self.detuning!=None:
					dset.attrs['detuning'] = self.detuning
				else:
					dset.attrs['detuning'] = 'None'
				
				dset.attrs['rabi_freq'] = self.rabi_freq
				if self.noise!=None:
					dset.attrs['noise'] = self.noise
				else:
					dset.attrs['noise'] = 'None'
			
				#extra save parameters
				if extra_save_parameters!=None:
					keys=list(extra_save_parameters.keys())
					for key in keys:
						dset.attrs[key]=extra_save_parameters[key]
			print(' === data saved === ')
		
		elif not os.path.exists(save_dir + file_name + ".hdf5"):
			
			print("\ncreating file "+ save_dir + file_name + ".hdf5\n")
			
			with h5py.File(save_dir + file_name + ".hdf5", "w") as file:

				dset = file.create_dataset(folder + '/'+ data_name,data=data)

				#store all relevant parameters as attributes
				#graph parameters
				dset.attrs['system_size']=self.L
				dset.attrs['seed_NV_system']=self.seed
				dset.attrs['B_field_dir_NV_system']=self.B_field_dir
				dset.attrs['rmin_NV_system']=self.min_dist
				dset.attrs['rmax_NV_system']=self.max_dist
				dset.attrs['scaling_factor_NV_system']=self.scaling_factor

				#dynamic parameters
				if self.detuning!=None:
					dset.attrs['detuning'] = self.detuning
				else:
					dset.attrs['detuning'] = 'None'
				
				dset.attrs['rabi_freq'] = self.rabi_freq
				if self.noise!=None:
					dset.attrs['noise'] = self.noise
				else:
					dset.attrs['noise'] = 'None'
			
				#extra save parameters
				if extra_save_parameters!=None:
					keys=list(extra_save_parameters.keys())
					for key in keys:
						dset.attrs[key]=extra_save_parameters[key]
			print(' === data saved === ')

		return folder


	def save_data_tuple(self,data_tuple,file_name,save_dir,folder,sub_directories,
					overwrite=False,extra_save_parameters=None):
		for d, data in enumerate(data_tuple):
			folder =self.save_data(data,file_name,save_dir,folder,sub_directories[d],
					overwrite=overwrite,extra_save_parameters=extra_save_parameters)
		return folder


	def evolve_periodic(self,initial_state,n_steps,observable,
						file_name,save_every=1000,save_dir='./data/',
						folder='new_data_set',extra_save_parameters=None,seed=1):

		"""! Method for Floquet evolution of a given sequence """

		##
		# @param initial_state initial state of the system. Input using QuSpin e.g.
		# ~~~~~~~~~~~~~{.py} 
		# initial_state = np.zeros(basis.Ns)
		# initial_state[basis.index('1'*L)]=1
		# ~~~~~~~~~~~~~
		# correspoding to a pure \f$\hat{z}\f$-polarized initial state.
		# 
		# @param n_steps number of Floquet periods to evolve the initial state.
		# @param observable observables of interest. Must be list of QuSpin Hamiltonian objects.
		# @param save_every data is automatically save after save_every many Floquet periods. Default is 1000.
		# @param file_name filename (without ending) to save the data.
		# @param save_dir directory to save the data in. Default is './data/'.
		# @param folder folder name of the data set within the file file_name to save the data (check .hdf5 format). Default is 'new_data_set'.
		# @param extra_save_parameters dict of additional parameters to be save. For example {'description':'This is a description of the data'}
		# @param seed seed used to generate noisy sequence in case noise is not None. Default is 1.
		#
		# @return data
		
		# loop thorugh building blocks preidocially
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		assert isinstance(observable,list), 'observables should be input as list of quspin.operators.hamiltonian objects'

		for obs in observable:
			assert isinstance(obs,hamiltonian) , 'observable input not understood: must be of type quspin.operators.hamiltonian'

		# preallocate memory to store the data

		# compute the number of values to be stored
		nr_of_data_points = sum(self.data_points())*n_steps
		data = np.zeros((len(observable),nr_of_data_points+1))
		times = np.zeros(nr_of_data_points+1)
		current_time = 0.0
		time_point =0

		# preallocate memory 
		psi = initial_state.copy().astype(np.complex128)
		work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued

		#save data and check if the file already exists
		folder = self.save_data_tuple((data,times),file_name,save_dir,folder,
										('observables','times'),
										overwrite=False,
										extra_save_parameters=extra_save_parameters)
		

		#compute initial expectation values of observables
		current_measurment_point=0
		for j in range(len(observable)):
			data[j,current_measurment_point] = observable[j].expt_value(psi).real/self.L 

		#pick random numbers for the noise
		with temp_seed(seed):
			random_num = np.random.uniform(-1,1,size=sum(self.data_points())*n_steps)


		#loop through the individual blocks
		rand_n_count = 0
		for step in range(n_steps):
			
			for block in self.building_blocks:
				#[[(),(),()],nr_of_reps]
				nr_of_reps = block[1]

				for n in range(nr_of_reps):
					
					for element in block[0]:
						#element has three entries ('dd',time,expH or None)

						#update time
						time = element[1]
						current_time += time

						#check if some expm_multiply_parallel objects need to be build in the loop (due to noise)
						if element[2]==None:

							exp_H= hlp.build_noisy_expH(self.L,self.basis,
														self.H_dd,self.rabi_freq,
														self.detuning,
														self.AC_function,
														current_time,
														time,
														self.noise,
														random_num[rand_n_count])
							rand_n_count += 1
						else:
							exp_H = element[2]

						exp_H.dot(psi,work_array=work_array,overwrite_v=True) 

						# measure the observable whenever only the dipolar part ('dd') is applied 
						if element[0]=='dd':
							current_measurment_point +=1
							for j in range(len(observable)):
								data[j,current_measurment_point] = observable[j].expt_value(psi).real/self.L 
							
							#update time
							time_point +=1
							times[time_point]= current_time

			print('finished Floquet cycle {0:d}'.format(step+1))
			
			#save data in hdf5 format
			if step % save_every == 0 and n_steps != 0:
				# existing data is overwritten/updated here
				folder = self.save_data_tuple((data,times),file_name,save_dir,folder,
												('observables','times'),
												overwrite=True,
												extra_save_parameters=extra_save_parameters)
		
		folder = self.save_data_tuple((data,times),file_name,save_dir,folder,
										('observables','times'),
										overwrite=True,
										extra_save_parameters=extra_save_parameters)
		
		
		return data, times



	def evolve_random(self,initial_state,n_steps,observable,
						file_name,save_every=1000,save_dir='./data/',
						folder='new_data_set',extra_save_parameters=None,
						seed=1,seed_random_seq=2):

		"""! Method for random evolution based on blocks"""

		##
		# @param initial_state initial state of the system. Input using QuSpin e.g.
		# ~~~~~~~~~~~~~{.py} 
		# initial_state = np.zeros(basis.Ns)
		# initial_state[basis.index('1'*L)]=1
		# ~~~~~~~~~~~~~
		# correspoding to a pure \f$\hat{z}\f$-polarized initial state.
		# 
		# @param n_steps number of Floquet periods to evolve the initial state.
		# @param observable observables of interest. Must be list of QuSpin Hamiltonian objects.
		# @param save_every data is automatically save after save_every many Floquet periods. Default is 1000.
		# @param file_name filename (without ending) to save the data.
		# @param save_dir directory to save the data in. Default is './data/'.
		# @param folder folder name of the data set within the file file_name to save the data (check .hdf5 format). Default is 'new_data_set'.
		# @param extra_save_parameters dict of additional parameters to be save. For example {'description':'This is a description of the data'}
		# @param seed seed used to generate noisy sequence in case noise is not None. Default is 1.
		# @param seed_random_seq seed used to generate the random sequence of blocks. Default is 2
		# 
		# @return data


		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		assert isinstance(observable,list), 'observables should be input as list of quspin.operators.hamiltonian objects'

		for obs in observable:
			assert isinstance(obs,hamiltonian) , 'observable input not understood: must be of type quspin.operators.hamiltonian'

		# preallocate memory to store the data

		# compute the number of values to be stored
		data = []
		times = []
		current_time = 0.0

		psi = initial_state.copy().astype(np.complex128)
		work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued

		#save data and check if the file already exists
		folder = self.save_data_tuple((np.array(data),np.array(times)),file_name,save_dir,folder,
										('observables','times'),
										overwrite=False,
										extra_save_parameters=extra_save_parameters)
				
		#compute initial expectation values of observables
		obs_vals = []
		for j in range(len(observable)):
			obs_vals +=  [observable[j].expt_value(psi).real/self.L]
		data += [obs_vals]
		times += [0.0]
		#pick random numbers from 0 to len(self.building_blocks)-1
		with temp_seed(seed_random_seq):
			ind_list = np.random.randint(low=0,high=len(self.building_blocks),size=n_steps)

		#pick random numbers for the noise
		with temp_seed(seed):
			random_num = np.random.uniform(-1,1,size=sum(self.data_points())*n_steps)

		#loop through the individual blocks
		rand_n_count = 0
		for step in range(n_steps):

			ind = ind_list[step]
			block = self.building_blocks[ind] # has the following form[[(),(),()],nr_of_reps]
		
			nr_of_reps = block[1]
			for n in range(nr_of_reps):
					
				for element in block[0]:
					
					#update time
					time = element[1]
					current_time += time

					#check if some expm_multiply_parallel objects need to be build in the loop (due to noise)
					if element[2]==None:
						exp_H= hlp.build_noisy_expH(self.L,self.basis,
													self.H_dd,
													self.rabi_freq,
													self.detuning,
													self.AC_function,
													current_time,
													time,
													self.noise,
													random_num[rand_n_count])
						rand_n_count += 1
					else:
						exp_H = element[2]

					exp_H.dot(psi,work_array=work_array,overwrite_v=True) 
					#print(element)

					#measure the observable whenever only the dipolar part ('dd') is applied 
					if element[0]=='dd':
						obs_vals = []
						for j in range(len(observable)):
							obs_vals +=  [observable[j].expt_value(psi).real/self.L]
						data += [obs_vals]
						#update time
						times += [current_time]
			
			print('finished cycle {0:d}'.format(step+1))
			
			#save data in hdf5 format
			if step % save_every == 0 and n_steps != 0:

				# existing data is overwritten/updated here
				folder = self.save_data_tuple((np.array(data).T,np.array(times)),file_name,save_dir,folder,
										('observables','times'),
										overwrite=True,
										extra_save_parameters=extra_save_parameters)
		
		
		folder = self.save_data_tuple((np.array(data).T,np.array(times)),file_name,save_dir,folder,
										('observables','times'),
										overwrite=True,
										extra_save_parameters=extra_save_parameters)
		
		data = np.array(data).T
		times = np.array(times)
		
		return data, times


	def evolve_sequential(self,initial_state,n_steps,observable,sequence,
							file_name,save_every=1000,save_dir='./data/',
							folder='new_data_set',extra_save_parameters=None,seed=1):

		"""! Method for sequential evolution based on blocks"""

		##
		# @param initial_state initial state of the system. Input using QuSpin e.g.
		# ~~~~~~~~~~~~~{.py} 
		# initial_state = np.zeros(basis.Ns)
		# initial_state[basis.index('1'*L)]=1
		# ~~~~~~~~~~~~~
		# correspoding to a pure \f$\hat{z}\f$-polarized initial state.
		# 
		# @param n_steps number of Floquet periods to evolve the initial state.
		# @param observable observables of interest. Must be list of QuSpin Hamiltonian objects.
		# @param sequence sequence n_steps integers (ranging from 0 to len(building_blocks)) specifying the drive sequence
		# @param file_name filename (without ending) to save the data.
		# @param save_every data is automatically save after save_every many Floquet periods. Default is 1000.
		# @param save_dir directory to save the data in. Default is './data/'.
		# @param folder folder name of the data set within the file file_name to save the data (check .hdf5 format). Default is 'new_data_set'.
		# @param extra_save_parameters dict of additional parameters to be save. For example {'description':'This is a description of the data'}
		# @param seed seed used to generate noisy sequence in case noise is not None. Default is 1.
		# 
		# @return data


		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		assert isinstance(observable,list), 'observables should be input as list of quspin.operators.hamiltonian objects'

		if np.any(np.array(sequence)>len(self.building_blocks)-1) or np.any(np.array(sequence)<0):
			raise AssertionError ('invalid sequence of intergers')
		
		for obs in observable:
			assert isinstance(obs,hamiltonian) , 'observable input not understood: must be of type quspin.operators.hamiltonian'

		# preallocate memory to store the data

		# compute the number of values to be stored
		data = []
		times = []
		current_time = 0.0


		psi = initial_state.copy().astype(np.complex128)
		work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued

		#save data and check if the file already exists
		folder = self.save_data_tuple((np.array(data),np.array(times)),file_name,save_dir,folder,
										('observables','times'),
										overwrite=False,
										extra_save_parameters=extra_save_parameters)
				

		#compute initial expectation values of observables
		obs_vals = []
		for j in range(len(observable)):
			obs_vals +=  [observable[j].expt_value(psi).real/self.L]
		data += [obs_vals]
		times += [0.0]
		#pick random numbers for the noise
		with temp_seed(seed):
			random_num = np.random.uniform(-1,1,size=sum(self.data_points())*n_steps)

		#loop through the individual blocks
		rand_n_count = 0
		for step in range(n_steps):

			#pick a the number from 0 to len(self.building_blocks)-1
			
			ind = sequence[step]
			block = self.building_blocks[ind] # has the following form[[(),(),()],nr_of_reps]
		
			nr_of_reps = block[1]
			for n in range(nr_of_reps):
					
				for element in block[0]:
					
					#update time
					time = element[1]
					current_time += time

					#check if some expm_multiply_parallel objects need to be build in the loop (due to noise)
					if element[2]==None:

						exp_H= hlp.build_noisy_expH(self.L,self.basis,
													self.H_dd,
													self.rabi_freq,
													self.detuning,
													self.AC_function,
													current_time,
													time,
													self.noise,
													random_num[rand_n_count])
						rand_n_count += 1
					else:
						exp_H = element[2]

					exp_H.dot(psi,work_array=work_array,overwrite_v=True) 

					#measure the observable whenever only the dipolar part ('dd') is applied 
					if element[0]=='dd':
						obs_vals = []
						for j in range(len(observable)):
							obs_vals +=  [observable[j].expt_value(psi).real/self.L]
						data += [obs_vals]
						#update time
						times += [current_time]
			
			print('finished cycle {0:d}'.format(step+1))
			
			#save data in hdf5 format
			if step % save_every == 0 and n_steps != 0:
				# existing data is overwritten/updated here
				folder = self.save_data_tuple((np.array(data).T,np.array(times)),file_name,save_dir,folder,
										('observables','times'),
										overwrite=True,
										extra_save_parameters=extra_save_parameters)
		
		
		folder = self.save_data_tuple((np.array(data).T,np.array(times)),file_name,save_dir,folder,
										('observables','times'),
										overwrite=True,
										extra_save_parameters=extra_save_parameters)
				
		data = np.array(data).T
		times = np.array(times)
		
		return data, times


	def evolve_time_dependent(self,initial_state,n_steps,observable,
						discrete_functions,
						file_name,save_every=1000,save_dir='./data/',
						folder='new_data_set',extra_save_parameters=None,
						seed=1,seed_random_seq=2):
		
		"""! Method for time dependent evolution based on blocks"""

		##
		# @param initial_state initial state of the system. Input using QuSpin e.g.
		# ~~~~~~~~~~~~~{.py} 
		# initial_state = np.zeros(basis.Ns)
		# initial_state[basis.index('1'*L)]=1
		# ~~~~~~~~~~~~~
		# correspoding to a pure \f$\hat{z}\f$-polarized initial state.
		# 
		# @param n_steps number of Floquet periods to evolve the initial state.
		# @param observable observables of interest. Must be list of QuSpin Hamiltonian objects.
		# @param sequence sequence n_steps integers (ranging from 0 to len(building_blocks)) specifying the drive sequence
		# @param discrete_functions list of functions with the length of building_blocks.
		#  Each function describes the modification of the corresponding block over time. If None, no function will be applied.
		# @param file_name filename (without ending) to save the data.
		# @param save_every data is automatically save after save_every many Floquet periods. Default is 1000.
		# @param save_dir directory to save the data in. Default is './data/'.
		# @param folder folder name of the data set within the file file_name to save the data (check .hdf5 format). Default is 'new_data_set'.
		# @param extra_save_parameters dict of additional parameters to be save. For example {'description':'This is a description of the data'}
		# @param seed seed used to generate noisy sequence in case noise is not None. Default is 1.
		# 
		# @return data


		assert len(discrete_functions)==len(self.building_blocks)


		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		assert isinstance(observable,list), 'observables should be input as list of quspin.operators.hamiltonian objects'

		for obs in observable:
			assert isinstance(obs,hamiltonian) , 'observable input not understood: must be of type quspin.operators.hamiltonian'

		# preallocate memory to store the data

		# compute the number of values to be stored
		nr_of_data_points = sum(self.data_points())*n_steps
		data = np.zeros((len(observable),nr_of_data_points+1))
		times = np.zeros(nr_of_data_points+1)
		current_time = 0.0
		time_point =0

		# preallocate memory 
		psi = initial_state.copy().astype(np.complex128)
		work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued

		#save data and check if the file already exists
		folder = self.save_data_tuple((data,times),file_name,save_dir,folder,
										('observables','times'),
										overwrite=False,
										extra_save_parameters=extra_save_parameters)
		

		#compute initial expectation values of observables
		current_measurment_point=0
		for j in range(len(observable)):
			data[j,current_measurment_point] = observable[j].expt_value(psi).real/self.L 

		#pick random numbers for the noise
		with temp_seed(seed):
			random_num = np.random.uniform(-1,1,size=sum(self.data_points())*n_steps)


		#loop through the individual blocks
		rand_n_count = 0

		for step in range(n_steps):
			
			# update the building blocks
			blocks = []
			# modify blocks 
			for b in range(len(self.building_blocks)):
				original_block = self.building_blocks[b]
				#[[(),(),..],n]		
				sequence = []
				function_input = discrete_functions[b]
				if function_input != None:
					if len(function_input)>1:
						function = function_input[0]
						params = tuple(function_input[1:])
						for e, element in enumerate(self.building_blocks[b][0]):
							time = function(original_block[0][e][1],step,*params)
							sequence += [(element[0],time,element[2])]

					else:
						function = function_input[0]
						for e,element in enumerate(self.building_blocks[b][0]):
							time = function(original_block[0][e][1],step)
							sequence += [(element[0],time,element[2])]

					current_block = [sequence,self.building_blocks[b][1]]

					# compute updates
					current_block = hlp.update_building_blocks(current_block,self.L,
																	self.basis,self.H_dd,
																	self.rabi_freq,self.detuning,
																	self.AC_function,self.noise)	
					#print(building_blocks)
					#print('update block with index {0:d}'.format(b))
					blocks += [current_block]
				else:
					current_block = self.building_blocks[b]
					blocks += [current_block]
					continue
				

			for block in blocks:
				#[[(),(),()],nr_of_reps]
				
				nr_of_reps = block[1]

				for n in range(nr_of_reps):
					
					for element in block[0]:
						#element has three entries ('dd',time,expH or None)

						#update time
						time = element[1]
						current_time += time

						#check if some expm_multiply_parallel objects need to be build in the loop (due to noise)
						if element[2]==None:

							exp_H= hlp.build_noisy_expH(self.L,self.basis,
														self.H_dd,self.rabi_freq,
														self.detuning,
														self.AC_function,
														current_time,
														time,
														self.noise,
														random_num[rand_n_count])
							rand_n_count += 1
						else:
							exp_H = element[2]

						exp_H.dot(psi,work_array=work_array,overwrite_v=True) 

						# measure the observable whenever only the dipolar part ('dd') is applied 
						if element[0]=='dd':
							current_measurment_point +=1
							for j in range(len(observable)):
								data[j,current_measurment_point] = observable[j].expt_value(psi).real/self.L 
							
							#update time
							time_point +=1
							times[time_point]= current_time

			print('finished Floquet cycle {0:d}'.format(step+1))
			
			#save data in hdf5 format
			if step % save_every == 0 and n_steps != 0:
				# existing data is overwritten/updated here
				folder = self.save_data_tuple((data,times),file_name,save_dir,folder,
												('observables','times'),
												overwrite=True,
												extra_save_parameters=extra_save_parameters)
			

		
		folder = self.save_data_tuple((data,times),file_name,save_dir,folder,
										('observables','times'),
										overwrite=True,
										extra_save_parameters=extra_save_parameters)
		
		
		return data, times



		# evolve

		# repeat

		return 0


@contextlib.contextmanager
def temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield	
	finally:
		np.random.set_state(state)
