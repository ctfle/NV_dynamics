import sys,os
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel
from scipy.sparse.linalg import eigsh
import numpy as np 
import pickle
import time
import statistics as stat
import scipy.integrate as integrate
import contextlib
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm


@contextlib.contextmanager
def temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield	
	finally:
		np.random.set_state(state)

class Helper_funcs():

	def __init__(self):
		self.description = 'Contains all functionalities used in NV_system and NV_dynamics'


	def sampling_points(self,B_field_dir,min_dist,max_dist,L,seed):
		np.random.seed(seed)
		box_size=10 #will be updated if necessary
		
		if B_field_dir=='x':
			B = np.array([1,0,0])
		elif B_field_dir =='y':
			B = np.array([0,1,0])
		elif B_field_dir=='z':
			B = np.array([0,0,1])
		else:
			raise AssertionError ("B_field_dir must be 'x', 'y' or 'z'")

		interactions_z = []
		interactions_x_y = []
		couplings = []
		spin_positions = np.ones((L,3))*min_dist*(-1)
		
		ell = 0
		counter = 0
		while ell <= L-1:
			new_position = np.random.uniform(low=-box_size,high =box_size,size = 3)
			
			#only accept points no further away than box_size from the origin where the NV center is located
			if np.linalg.norm(new_position)>box_size:
				continue

			#compute distance to all other existing points on the graph
			distance = np.linalg.norm(spin_positions-new_position,axis=1)

			# increase the boxsize if necessary
			if counter > L**5:
				print('spin_positions are:', spin_positions)
				raise Exception('box to small: cannot fit all {0:d} points in the given box size with min distance {1:0.3f}. Only {2:d} spin could be placed'.format(*(L,min_dist,ell+1)))
				box_size +=1
				counter =0

			if ell == 0:
				spin_positions[ell] = new_position
				ell += 1

			else:
				#check if new_position sattisfies all conditions 
				min_cond = (distance < min_dist)
				max_cond = (np.min(distance[:ell]) > max_dist)
				cond = min_cond + max_cond

				if True in cond:
					counter +=1 
					continue

				# get all interactions
				else:
					spin_positions[ell] = new_position
					k = np.copy(ell) -1

					while k >= 0:
						dist = np.linalg.norm(new_position- spin_positions[k])
						
						angle = np.dot((new_position- spin_positions[k])/dist,B/np.linalg.norm(B))
						coupling_constant = (3*angle**2-1)/(dist)**3.0


						couplings.append(coupling_constant)
						interactions_z.append([2*coupling_constant,k,ell])
						interactions_x_y.append([-1*coupling_constant,k,ell])
						k -=1

					ell += 1

		return interactions_x_y, interactions_z, spin_positions, couplings





	def construct_Hamiltonian(self,basis,coupling_terms):

		H = hamiltonian(coupling_terms,[],basis=basis,dtype=np.complex128,check_symm=False, check_herm=False)

		return H



	def compute_observables(self,j,psi,L,obs,O):
		# updates variables in-place
		for k in range(len(obs)):
			obs[k,j+1] = O[k].expt_value(psi).real/L 



	def estimate_scales(self,basis,L,H,delta_t=0.0005,time_steps=1000):	
		"""! estimates relevant time scales via decay of spins using the fully polarized state as init state """
		
		psi_i = np.zeros(basis.Ns)
		psi_i[basis.index('1'*L)]=1

		Ox=hamiltonian([['x',[[1.0,j] for j in range(L)] ],],[],basis=basis,dtype=np.complex128,check_symm=False)	
		
		psi=psi_i.copy().astype(np.complex128)
		work_array=np.zeros((2*len(psi),), dtype=psi.dtype)
		
		Oy=hamiltonian([['y',[[1.0,j] for j in range(L)] ],],[],basis=basis,dtype=np.complex128,check_symm=False)	
		init_rotation = expm_multiply_parallel(Oy.tocsr(),a=-1j*np.pi*0.25)
		init_rotation.dot(psi,work_array=work_array,overwrite_v=True)

		observable=np.zeros((1,time_steps+1),dtype=np.float64)

		self.compute_observables(-1,psi,L,observable,[Ox])
			
		expH = expm_multiply_parallel(H.tocsr(),a=-1j*delta_t)
		for j in range(time_steps):
			expH.dot(psi,work_array=work_array,overwrite_v=True)
					
			self.compute_observables(j,psi,L,observable,[Ox])

		intersect=np.abs(np.abs(observable[0]) - np.exp(-1)).argmin()
		median_coupling = 1/(intersect*delta_t)

		return median_coupling 



	def compute_single_particle_fields(self,spin_positions,energy_scale,B_field_dir,scaling_factor=1):
		"""!Computes single particle terms originating from the NV center at the origin """

		if B_field_dir=='x':
			B = np.array([1,0,0])
		elif B_field_dir =='y':
			B = np.array([0,1,0])
		elif B_field_dir=='z':
			B = np.array([0,0,1])
		
		NV_distance = np.zeros(len(spin_positions))

		for s,spin_pos in enumerate(spin_positions):
			dist = np.linalg.norm(spin_pos)
			angle = np.dot(( spin_pos )/dist,B/np.linalg.norm(B))
			coupling_constant = (3*angle**2-1)/(dist)**3.0
			NV_distance[s] =coupling_constant

		NV_distance /= max(NV_distance)
		NV_distance *= energy_scale*scaling_factor
		z_field = [[NV_distance[z],z] for z in range(len(NV_distance))]
		
		return z_field



	def yes_no(self,message):
		# raw_input returns the empty string for "enter"
		yes = {'yes','y', 'ye', ''}
		no = {'no','n'}
		test = True
		while test:
			print(message,': [y/n]')
			choice = input().lower()
			if choice in yes:
				test=False
				return True
			elif choice in no:
				test=False
				return False
			else:
				sys.stdout.write("Please respond with 'yes' or 'no'\n")


	def setup_expH(self,L,basis,H_dd,kick_building_blocks,rabi_freq,detuning,AC_function,noise):
		"""! Constructs all the matrix exponentials from the building block inputs of the sequence"""

		# initialize the kick sequence according to the specific case under consideration
		# building blocks [[building_block1],[building_block2]] with building_block1 = [[('x',0.4),('dd',0.2),()...],n_times],
		# set up list of exponentials according to the building blocks
		
		#if noise=None, setup all block elements. else setup only those block elements that are always the same

		blocks = []
		if noise==None:
			for element in kick_building_blocks:
				current_time = 0.0
				
				#construct list to store the corresponding exp(H)
				sequence_expH = []
				for sequence_brick in element[0]:
					#sequence_brick[0] ~ 'x', 'y', 'z', 'dd'
					#sequence_brick[1] ~ correspnding times in units of the energy scale J in case of 'dd', corresponding angles in case 'x','y','z'
					
					# the sequence part is given by the dipolar Hamiltonian
					if sequence_brick[0]=='dd':

						# time is given in units of 1/self.energy_scale
						time = sequence_brick[1] 

						# H_dd is alread rescaled in units of self.energy_scale
						H=H_dd*time 
						

						# check for detuning 
						# detuning , in units of self.energy_scale
						if detuning != None:
							if type(detuning)==list:
								assert len(detuning)==L, 'not enough elements given in detuning: L={0:d}, length of detuning ={1:d}'.format(*(L,len(detuning)))
								detuing_list=[[detuning[j]*time,j] for j in range(L)]
							else:
								detuing_list=[[detuning*time,j] for j in range(L)]

							H +=  self.construct_Hamiltonian(basis,[['z',detuing_list]]).tocsr()



						# check for AC
						# amplitudes appearing in AC function are assumed to be given in units of self.energy_scale
						if AC_function != None:
														
							function = AC_function[0]
							if len(AC_function)>1:
								params = tuple(AC_function[1:]) # all parameters of interest such as amplitude, frequency etc
								#integrate the AC function from current_time to current_time + time
								AC_coupling=integrate.quad(lambda x: function(x,*params),current_time,current_time+time)[0]
							else:
								AC_coupling=integrate.quad(lambda x: function(x),current_time,current_time+time)[0]

							kick_list = [[AC_coupling,j] for j in range(L)]
							static_z = [['z',kick_list]]
							H += self.construct_Hamiltonian(basis, static_z).tocsr()


						expH = expm_multiply_parallel(H,a=-1j)
						sequence_expH += [(sequence_brick[0],time,expH)]
						
						current_time += time
					

					# the sequence part is given by a kick
					else:

						# rabi_frequency is given in units of self.energy_scale
						# the kick_time is given in units of the 1/energy_scale
						# kick_amplitude = rabi_frequency * kick_time

						time = sequence_brick[1]

						kick_amplitude = time*rabi_freq
						kick_direction = sequence_brick[0]

						kick_coupling = [[kick_amplitude,j] for j in range(L)]
						static = [[kick_direction,kick_coupling]]


						#check for detuning
						if detuning != None:
							if type(detuning)==list:
								assert len(detuning)==L, 'not enough elements given in detuning: L={0:d}, length of detuning ={1:d}'.format(*(L,len(detuning)))
								detuing_list=[[detuning[j]*time,j] for j in range(L)]
							else:
								detuing_list=[[detuning*time,j] for j in range(L)]

							static += [['z',detuing_list]]
						

						H = self.construct_Hamiltonian(basis, static )
						expH = expm_multiply_parallel(H.tocsr(),a=-1j)
						sequence_expH += [(sequence_brick[0],time,expH)]

						current_time += time

				blocks += [  [sequence_expH,element[1]]  ]
		
		# noise != None
		else:
			for element in kick_building_blocks:
				current_time = 0.0
				
				#construct list to store the corresponding exp(H)
				sequence_expH = []
				for sequence_brick in element[0]:
					#sequence_brick[0] ~ 'x', 'y', 'z', 'dd'
					#sequence_brick[1] ~ correspnding times in units of the energy scale J in case of 'dd', corresponding angles in case 'x','y','z'

					# the sequence part is given by the dipolar Hamiltonian
					if sequence_brick[0]=='dd':

						time = sequence_brick[1] 
						sequence_expH += [('dd',time,None)]
						current_time += time
					
					# the sequence part is given by a kick
					else:

						# rabi_frequency is given in units of self.energy_scale
						# the kick_time is given in units of the 1/energy_scale
						# kick_amplitude = rabi_frequency * kick_time

						time = sequence_brick[1]

						kick_amplitude = time*rabi_freq

						kick_direction = sequence_brick[0]

						kick_coupling = [[kick_amplitude,j] for j in range(L)]
						static = [[kick_direction,kick_coupling]]


						#check for detuning
						if detuning != None:
							if type(detuning)==list:
								assert len(detuning)==L, 'not enough elements given in detuning: L={0:d}, length of detuning ={1:d}'.format(*(L,len(detuning)))
								detuing_list=[[detuning[j]*time,j] for j in range(L)]
							else:
								detuing_list=[[detuning*time,j] for j in range(L)]

							static += [['z',detuing_list]]
						

						H = self.construct_Hamiltonian(basis, static )
						expH = expm_multiply_parallel(H.tocsr(),a=-1j)
						sequence_expH += [(sequence_brick[0],time,expH)]

						current_time += time

				blocks += [  [sequence_expH,element[1]]  ]
		

		return blocks


	def copy_attributes(self,inpt):
		output = inpt.copy()
		return output

	def update_building_blocks(self,element,L,basis,H_dd,rabi_freq,detuning,AC_function,noise):

		
		if noise==None:

			current_time = 0.0
				
			#construct list to store the corresponding exp(H)
			sequence_expH = []
			for sequence_brick in element[0]:
				#sequence_brick[0] ~ 'x', 'y', 'z', 'dd'
				#sequence_brick[1] ~ correspnding times in units of the energy scale J in case of 'dd', corresponding angles in case 'x','y','z'
				
				# the sequence part is given by the dipolar Hamiltonian
				if sequence_brick[0]=='dd':

					# time is given in units of 1/self.energy_scale
					time = sequence_brick[1] 

					# H_dd is alread rescaled in units of self.energy_scale
					H=H_dd*time 
						

					# check for detuning 
					# detuning , in units of self.energy_scale
					if detuning != None:
						if type(detuning)==list:
							assert len(detuning)==L, "not enough elements given in detuning:\
							 						L={0:d}, length of detuning ={1:d}".format(*(L,len(detuning)))
							detuing_list=[[detuning[j]*time,j] for j in range(L)]
						else:
							detuing_list=[[detuning*time,j] for j in range(L)]

						H +=  self.construct_Hamiltonian(basis,[['z',detuing_list]]).tocsr()



					# check for AC
					# amplitudes appearing in AC function are assumed to be given in units of self.energy_scale
					if AC_function != None:
								
						function = AC_function[0]
						if len(AC_function)>1:
							params = tuple(AC_function[1:]) # all parameters of interest such as amplitude, frequency etc
							#integrate the AC function from current_time to current_time + time
							AC_coupling=integrate.quad(lambda x: function(x,*params),current_time,current_time+time)[0]
						else:
							AC_coupling=integrate.quad(lambda x: function(x),current_time,current_time+time)[0]

						kick_list = [[AC_coupling,j] for j in range(L)]
						static_z = [['z',kick_list]]
						H += self.construct_Hamiltonian(basis, static_z).tocsr()


					expH = expm_multiply_parallel(H,a=-1j)
					sequence_expH += [(sequence_brick[0],time,expH)]
						
					current_time += time
					

				# the sequence part is given by a kick
				else:

					# rabi_frequency is given in units of self.energy_scale
					# the kick_time is given in units of the 1/energy_scale
					# kick_amplitude = rabi_frequency * kick_time

					time = sequence_brick[1]

					kick_amplitude = time*rabi_freq
					kick_direction = sequence_brick[0]

					kick_coupling = [[kick_amplitude,j] for j in range(L)]
					static = [[kick_direction,kick_coupling]]


					#check for detuning
					if detuning != None:
						if type(detuning)==list:
							assert len(detuning)==L, "not enough elements given in detuning:\
									 L={0:d}, length of detuning ={1:d}".format(*(L,len(detuning)))
							detuing_list=[[detuning[j]*time,j] for j in range(L)]
						else:
							detuing_list=[[detuning*time,j] for j in range(L)]

						static += [['z',detuing_list]]
						

					H = self.construct_Hamiltonian(basis, static )
					expH = expm_multiply_parallel(H.tocsr(),a=-1j)
					sequence_expH += [(sequence_brick[0],time,expH)]
					current_time += time

				element = [sequence_expH,element[1]]
			
			return element
			
		
		# noise != None
		else:

			current_time = 0.0
			
			#construct list to store the corresponding exp(H)
			sequence_expH = []
			for sequence_brick in element[0]:
				#sequence_brick[0] ~ 'x', 'y', 'z', 'dd'
				#sequence_brick[1] ~ correspnding times in units of the energy scale J in case of 'dd', corresponding angles in case 'x','y','z'

				# the sequence part is given by the dipolar Hamiltonian
				if sequence_brick[0]=='dd':

					time = sequence_brick[1] 
					sequence_expH += [('dd',time,None)]
					current_time += time
				
				# the sequence part is given by a kick
				else:

					# rabi_frequency is given in units of self.energy_scale
					# the kick_time is given in units of the 1/energy_scale
					# kick_amplitude = rabi_frequency * kick_time

					time = sequence_brick[1]

					kick_amplitude = time*rabi_freq

					kick_direction = sequence_brick[0]

					kick_coupling = [[kick_amplitude,j] for j in range(L)]
					static = [[kick_direction,kick_coupling]]


					#check for detuning
					if detuning != None:
						if type(detuning)==list:
							assert len(detuning)==L, "not enough elements given in detuning:\
										 L={0:d}, length of detuning ={1:d}".format(*(L,len(detuning)))
							detuing_list=[[detuning[j]*time,j] for j in range(L)]
						else:
							detuing_list=[[detuning*time,j] for j in range(L)]

						static += [['z',detuing_list]]
					

					H = self.construct_Hamiltonian(basis, static )
					expH = expm_multiply_parallel(H.tocsr(),a=-1j)
					sequence_expH += [(sequence_brick[0],time,expH)]

					current_time += time

				element= [sequence_expH,element[1]] 

			return element
	



	def build_noisy_expH(self,L,basis,H_dd,rabi_freq,detuning,AC_function,current_time,time,noise,random_num):
		# time is given in units of 1/self.energy_scale
		# H_dd is alread rescaled in units of self.energy_scale
		time += time*noise*random_num
		
		H=H_dd*time 
						

		# check for detuning 
		# detuning , in units of self.energy_scale
		if detuning != None:
			if type(detuning)==list:
				assert len(detuning)==L, "not enough elements given in detuning:\
								 L={0:d}, length of detuning ={1:d}".format(*(L,len(detuning)))
				detuing_list=[[detuning[j]*time,j] for j in range(L)]
			else:
				detuing_list=[[detuning*time,j] for j in range(L)]

			H +=  self.construct_Hamiltonian(basis,[['z',detuing_list]]).tocsr()


		# check for AC
		# amplitudes appearing in AC function are assumed to be given in units of self.energy_scale
		if AC_function != None:
						
			function = AC_function[0]
			if len(AC_function)>1:
				params = tuple(AC_function[1:]) # all parameters of interest such as amplitude, frequency etc
				#integrate the AC function from current_time to current_time + time
				AC_coupling=integrate.quad(lambda x: function(x,*params),current_time,current_time+time)[0]
			else:
				AC_coupling=integrate.quad(lambda x: function(x),current_time,current_time+time)[0]

			kick_list = [[AC_coupling,j] for j in range(L)]
			static_z = [['z',kick_list]]
			H += self.construct_Hamiltonian(basis, static_z).tocsr()


		expH = expm_multiply_parallel(H,a=-1j)
		
		return expH


