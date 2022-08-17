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

class nv_system():
	
	""" sets up a random graph of L spins where each spin has a min_dist to all other spins
		and is at least connected to one other spin at no further than max_dist """
	
	def __init__(self,dimension,L,min_dist,max_dist,seed):
		with temp_seed(seed):
			interactions_x, interactions_z, spin_positions, couplings = sampling_points(dimension,min_dist,max_dist,L)

		self.seed =seed
		self.L = L
		self.interactions_x = interactions_x
		self.interactions_z = interactions_z
		self.couplings = couplings
		self.spin_positions = spin_positions


class nv_hamiltonian(nv_system):
	
	""" all the directions are rotated by 90deg around the y-axis to have a simple iniital state """
	
	def __init__(self,kick_seq,RK=False,*system_params):
		
		super().__init__(*system_params)
		
		self.ratio = kick_seq[0]
		self.kick_angle = kick_seq[1]
		time_ratio = kick_seq[2]
		self.kick_dir = kick_seq[3]
		self.nr_of_reps = kick_seq[4]
		self.amplitude_AC= kick_seq[5]
		self.function = kick_seq[6]
		detuning = kick_seq[7]
		
		self.system_params = (system_params)
		
		if len(kick_seq) > 8:
			self.function_params = kick_seq[8]
		else:
			self.function_params = None

		
		self.basis = spin_basis_1d(L=self.L,pauli=True)

		H_dd = construct_dipolar(self.basis,self.couplings,self.interactions_x,self.interactions_z,self.L,self.kick_dir,self.seed)


		self.median_coupling = estimate_scales(self.basis,self.L,H_dd,delta_t=0.0005,time_steps=1000)
		

		self.off_set_z = detuning * self.median_coupling


		H_detuning = construct_detuning(self.basis,self.L,self.off_set_z)

		self.H_dipolar = H_dd + H_detuning



		self.application_time_dipolar = (1- time_ratio) * self.ratio / self.median_coupling 
		self.appl_time_kick = time_ratio * self.ratio / self.median_coupling
		


		self.kick_params = (self.kick_angle,self.nr_of_reps,self.amplitude_AC,detuning,self.ratio,self.median_coupling,self.application_time_dipolar,self.appl_time_kick)


		#print(self.median_coupling)
		#print(self.application_time_dipolar)
		#exit()

		H_dyn, H_kick = construct_dynamic(self.basis,self.L,self.application_time_dipolar,self.appl_time_kick,self.nr_of_reps,self.amplitude_AC,self.median_coupling,self.function,self.kick_dir, self.kick_angle,self.off_set_z,RK)

		self.H_dyn = H_dyn

		self.H_kick = H_kick


	


	def define_observables(self):
		
		Ox=hamiltonian([['x',[[1.0,j] for j in range(self.L)] ],],[],basis=self.basis,dtype=np.complex128,check_symm=False)
		Oy=hamiltonian([['y',[[1.0,j] for j in range(self.L)] ],],[],basis=self.basis,dtype=np.complex128,check_symm=False)
		Oz=hamiltonian([['z',[[1.0,j] for j in range(self.L)] ],],[],basis=self.basis,dtype=np.complex128,check_symm=False)

		O = [Ox,Oy,Oz]

		return O






	def decayed_init_state(self,psi_i,t_init):
		
		""" time evolves psi_i for t_init with H_dipolar """
		
		psi_i = psi_i.astype(np.complex128)
		w_a=np.zeros((2*len(psi_i),), dtype=psi_i.dtype) # twice as long because complex-valued
		# construct unitaries
		expH_init = expm_multiply_parallel(self.H_dipolar.tocsr(),a=-1j*t_init) # does NOT compute matrix exponential
		expH_init.dot(psi_i,work_array=w_a,overwrite_v=True)

		return psi_i

	




	def save_data(self,ell,j,N_d,store_cycles,N_steps,noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head):

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		param_tuple = (self.seed,self.L,N_steps,self.ratio,self.kick_angle,self.nr_of_reps,self.amplitude_AC,noise)


		func_params =''
		if self.function_params != None:
			for p in range(len(self.function_params)):
				func_params += '_{0:0.4f}'.format(self.function_params[p])


		if not save_state:
			params = '--data--seed={0:03d}_L={1:d}_N_steps={2:d}_JT={3:0.4f}_JT_kick={4:0.4f}_N_rep={5:d}_amp_AC={6:0.4f}_noise={7:0.4f}'.format(*param_tuple)
			file_name = file_name_head + params + func_params
			

			if j//self.nr_of_reps-store_cycles[ell] == N_d-1 or j//self.nr_of_reps==N_steps-1:
				with open(save_dir + file_name+'.pkl', 'wb') as handle:
					parameters = self.system_params + self.kick_params
					pickle.dump([*parameters, observable,psi_i], handle, protocol=pickle.HIGHEST_PROTOCOL)
				if j!=N_steps-1:
					ell +=1


		else:
			params = '--data_with_states--seed={0:03d}_L={1:d}_N_steps={2:d}_JT={3:0.4f}_JT_kick={4:0.4f}_N_rep={5:d}_amp_AC={6:0.4f}_noise={7:0.4f}'.format(*param_tuple)
			file_name = file_name_head + params + func_params
			
			if 0 <= j//self.nr_of_reps-store_cycles[ell] < N_d and ell<=store_cycles.shape[0]-1:
				print(j,ell)
				psi_t[...,ell,j//self.nr_of_reps-store_cycles[ell]]=psi

			elif j//self.nr_of_reps-store_cycles[ell] == N_d-1 or j//self.nr_of_reps==N_steps-1:
				with open(save_dir + file_name+'.pkl', 'wb') as handle:
					parameters = self.system_params + self.kick_params
					pickle.dump([*parameters, observable,psi_i,psi_t], handle, protocol=pickle.HIGHEST_PROTOCOL)
				if j!=N_steps-1:
					ell +=1

		return ell





    


	def evolve(self,psi_i,N_steps,save_dir,file_name_head,sequence_seed=1, N_d=10, save_every=1000,kick_func=None, Noise=None, save_state=False, drive='periodic', multipole='standard'):

		if multipole == 'standard':
			if drive=='random':
				seq = random_seq(N_steps,sequence_seed)

			elif drive=='fibonacci':
				len_fib =0
				f = 0
				while len_fib<N_steps:
					f +=1
					seq = fibonacci_seq(f)
					len_fib = len(seq)
			else:
				seq = np.ones(N_steps)
			    
			observable = self.evolve_standard(psi_i,N_steps,save_dir, seq, file_name_head,N_d,save_every,Noise,save_state,kick_func=kick_func)

		if multipole=='dipolar':
			if drive=='random':
				seq = random_seq(N_steps,sequence_seed)

			elif drive=='fibonacci':
				len_fib =0
				f = 0
				while len_fib<N_steps:
					f +=1
					seq = fibonacci_seq(f)
					len_fib = len(seq)
			else:
				seq       = np.ones(N_steps)
				seq[1::2] = 0
			observable=self.evolve_dipolar(psi_i,N_steps,save_dir, seq, file_name_head,N_d,save_every,Noise,save_state,kick_func=kick_func)
		return observable

	def evolve_dipolar(self,psi_i,N_steps,save_dir, seq, file_name_head,N_d,save_every,Noise,save_state,kick_func=None):
		


		#H_extra = construct_extra(self.basis,self.L,np.pi/2)
		#expH_extra = expm_multiply_parallel(H_extra.tocsr(),a=-1j)

		store_cycles = np.array([save_every*k for k in range(N_steps//save_every+1)])
		observable = np.zeros((3,N_steps*self.nr_of_reps+1),dtype=np.float64)
		O = self.define_observables()
		
		compute_observables(-1,psi_i,self.L,observable,O)
        
		if Noise is None:
			
			# auxiliary array for efficiency
			psi=psi_i.copy().astype(np.complex128)
			work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued

			# construct unitaries
			expH_dipolar = expm_multiply_parallel(self.H_dipolar.tocsr(),a=-1j*self.application_time_dipolar) 
			#expH_step_kick = expm_multiply_parallel(self.H_step_kick.tocsr(),a=-1j)

			expHz = []
			expH_step_kick = []
			for n in range(self.nr_of_reps):
				expHz.append( expm_multiply_parallel(self.H_dyn[n].tocsr(),a=-1j) )
				expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j) )


			ti=time.time()
			
			ell=0
			if save_state:
				psi_t=np.zeros((self.basis.Ns,store_cycles.shape[0],N_d,),dtype=np.complex128)
				psi_t[:,ell,0]=psi_i
			else:
				psi_t=None
			
			for j in range(N_steps*self.nr_of_reps):
				
				if kick_func != None:
					expH_step_kick = []
					for n in range(self.nr_of_reps):
						kick_value = kick_func(j/(N_steps*self.nr_of_reps))
						expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j*kick_value) )



				n = j % self.nr_of_reps
				
				if 0 <= j//self.nr_of_reps-store_cycles[ell] < N_d and ell<=store_cycles.shape[0]-1:
					ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)


				if seq[j//self.nr_of_reps]==1:
					expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
					expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
					expHz[n].dot(psi,work_array=work_array,overwrite_v=True) 
                    
				else:
					expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
					expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
					expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
                    
				compute_observables(j,psi,self.L,observable,O)

				if n ==0:
					print('finished Floquet cycle {0:d}'.format(j//self.nr_of_reps))


			ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)


			tf=time.time()
			print('\nfinished evolving state in {0:0.4f} secs.\n'.format(tf-ti))


		else:
			# auxiliary array for efficiency
			psi=psi_i.copy().astype(np.complex128)
			work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued


			# construct unitaries
			expHz = []
			expH_step_kick = []
			for n in range(self.nr_of_reps):
				expHz.append( expm_multiply_parallel(self.H_dyn[n].tocsr(),a=-1j) )
				expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j) )


			ti=time.time()

			ell=0
			if save_state:
				psi_t=np.zeros((self.basis.Ns,store_cycles.shape[0],N_d,),dtype=np.complex128)
				psi_t[:,ell,0]=psi_i
			else:
				psi_t=None
			

			for j in range(N_steps*self.nr_of_reps):
				 
				if kick_func != None:
					expH_step_kick = []
					for n in range(self.nr_of_reps):
						kick_value = kick_func(j/(N_steps*self.nr_of_reps))
						expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j*kick_value) )


				noise_strength=Noise*np.random.uniform(-1,1)
				expH_dipolar = expm_multiply_parallel(self.H_dipolar.tocsr(),a=-1j*(self.application_time_dipolar+self.application_time_dipolar*noise_strength))  
			
				n = j % self.nr_of_reps


				if 0 <= j//self.nr_of_reps-store_cycles[ell] < N_d and ell<=len(store_cycles)-1:
					ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)



				if ((j//self.nr_of_reps) % 1 ==0 and j % self.nr_of_reps == self.nr_of_reps-1 and j!= self.nr_of_reps-1):
					#print('true j = {0:d}'.format(j))
					#	expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True) 
					if seq[j//self.nr_of_reps]==1:
					
						expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
						#expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 					
						
					else:
						#expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
						expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
                        
						#expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True)


				else:

					if seq[j//self.nr_of_reps]==1:

						expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True) 

						#if (j//self.nr_of_reps) % 10 ==0  and j != 0 and n==0:
							#print('extra kick')
							#expH_extra.dot(psi,work_array=work_array,overwrite_v=True) 
							#expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
							#expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
					else:
						
						#expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
						expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True)
                        
				compute_observables(j,psi,self.L,observable,O)

				if n ==0:
					print('finished Floquet cycle {0:d}'.format(j//self.nr_of_reps))
			
			ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)


			tf=time.time()
			print('\nfinished evolving state in {0:0.4f} secs.\n'.format(tf-ti))

	
		return observable
        
        
	def evolve_standard(self,psi_i,N_steps,save_dir, seq, file_name_head,N_d,save_every,Noise,save_state,kick_func=None):
		#H_extra = construct_extra(self.basis,self.L,np.pi/2)
		#expH_extra = expm_multiply_parallel(H_extra.tocsr(),a=-1j)

		store_cycles = np.array([save_every*k for k in range(N_steps//save_every+1)])
		observable = np.zeros((3,N_steps*self.nr_of_reps+1),dtype=np.float64)
		O = self.define_observables()
		
		compute_observables(-1,psi_i,self.L,observable,O)
        
		if Noise is None:
			
			# auxiliary array for efficiency
			psi=psi_i.copy().astype(np.complex128)
			work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued

			# construct unitaries
			expH_dipolar = expm_multiply_parallel(self.H_dipolar.tocsr(),a=-1j*self.application_time_dipolar) 
			#expH_step_kick = expm_multiply_parallel(self.H_step_kick.tocsr(),a=-1j)

			expHz = []
			expH_step_kick = []
			for n in range(self.nr_of_reps):
				expHz.append( expm_multiply_parallel(self.H_dyn[n].tocsr(),a=-1j) )
				expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j) )


			ti=time.time()
			
			ell=0
			if save_state:
				psi_t=np.zeros((self.basis.Ns,store_cycles.shape[0],N_d,),dtype=np.complex128)
				psi_t[:,ell,0]=psi_i
			else:
				psi_t=None
			
			for j in range(N_steps*self.nr_of_reps):
				
				if kick_func != None:
					expH_step_kick = []
					for n in range(self.nr_of_reps):
						kick_value = kick_func(j/(N_steps*self.nr_of_reps))
						expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j*kick_value) )


				n = j % self.nr_of_reps
				
				if 0 <= j//self.nr_of_reps-store_cycles[ell] < N_d and ell<=store_cycles.shape[0]-1:
					ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)


				if seq[j//self.nr_of_reps]==1:
					expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
					expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
					expHz[n].dot(psi,work_array=work_array,overwrite_v=True) 
				else:
					#expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
					expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
					expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
				
				compute_observables(j,psi,self.L,observable,O)

				if n ==0:
					print('finished Floquet cycle {0:d}'.format(j//self.nr_of_reps))


			ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)


			tf=time.time()
			print('\nfinished evolving state in {0:0.4f} secs.\n'.format(tf-ti))


		else:
			# auxiliary array for efficiency
			psi=psi_i.copy().astype(np.complex128)
			work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued


			# construct unitaries
			expHz = []
			expH_step_kick = []
			for n in range(self.nr_of_reps):
				expHz.append( expm_multiply_parallel(self.H_dyn[n].tocsr(),a=-1j) )
				expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j) )
				

			ti=time.time()

			ell=0
			if save_state:
				psi_t=np.zeros((self.basis.Ns,store_cycles.shape[0],N_d,),dtype=np.complex128)
				psi_t[:,ell,0]=psi_i
			else:
				psi_t=None
			
			for j in range(N_steps*self.nr_of_reps):
				

				if kick_func != None:
					expH_step_kick = []
					for n in range(self.nr_of_reps):
						kick_value = kick_func(j/(N_steps*self.nr_of_reps))
						expH_step_kick.append( expm_multiply_parallel(self.H_kick[n].tocsr(),a=-1j*kick_value) )
						print(kick_value)

				

				noise_strength=Noise*np.random.uniform(-1,1)
				expH_dipolar = expm_multiply_parallel(self.H_dipolar.tocsr(),a=-1j*(self.application_time_dipolar+self.application_time_dipolar*noise_strength))  
			
				n = j % self.nr_of_reps


				if 0 <= j//self.nr_of_reps-store_cycles[ell] < N_d and ell<=len(store_cycles)-1:
					ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)



				if ((j//self.nr_of_reps) % 1 ==0 and j % self.nr_of_reps == self.nr_of_reps-1 and j!= self.nr_of_reps-1):
					#print('true j = {0:d}'.format(j))
					#	expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True) 
					if seq[j//self.nr_of_reps]==1:
					
						expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
						#expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 					
						
					else:
						#expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
						
						#expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True)


				else:

					if seq[j//self.nr_of_reps]==1:

						expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True) 

						#if (j//self.nr_of_reps) % 10 ==0  and j != 0 and n==0:
							#print('extra kick')
							#expH_extra.dot(psi,work_array=work_array,overwrite_v=True) 
							#expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
							#expH_step_kick[n].dot(psi,work_array=work_array,overwrite_v=True) 
					else:
						
						#expH_step_kick.dot(psi,work_array=work_array,overwrite_v=True) 
						expH_dipolar.dot(psi,work_array=work_array,overwrite_v=True)
						expHz[n].dot(psi,work_array=work_array,overwrite_v=True)
					
				compute_observables(j,psi,self.L,observable,O)

				if n ==0:
					print('finished Floquet cycle {0:d}'.format(j//self.nr_of_reps))
			
			ell = self.save_data(ell,j,N_d,store_cycles,N_steps,Noise,observable,psi_i,psi,psi_t,save_state,save_dir,file_name_head)


			tf=time.time()
			print('\nfinished evolving state in {0:0.4f} secs.\n'.format(tf-ti))

	
		return observable








@contextlib.contextmanager
def temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield	
	finally:
		np.random.set_state(state)



def sampling_points(dimension,min_dist,max_dist,L):
	box_size=10 #will be updated if necessary
	
	if dimension==3:
		B = np.array([0,1,0])
	else:
		B=np.array([1])
	
	interactions_z = []
	interactions_x = []
	couplings = []
	spin_positions = np.ones((L,dimension))*min_dist*(-1)
	
	ell = 0
	counter = 0
	while ell <= L-1:
		new_position = np.random.uniform(low=0.0,high =box_size,size = dimension)
		distance = np.linalg.norm(spin_positions-new_position,axis=1)

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
					interactions_x.append([-1*coupling_constant,k,ell])
					k -=1

				ell += 1

	return interactions_x, interactions_z, spin_positions, couplings






def construct_dipolar(basis,couplings,interactions_x,interactions_z,L,kick_dir,seed):
	with temp_seed(seed):
		if L>1:
			mean_coupling = stat.median(np.abs(couplings))
			std_c = 10.0*mean_coupling
			c_x = np.random.normal(loc = mean_coupling,scale=std_c,size=L)
		else:
			mean_coupling = 0.1
			std_c = 10.0*mean_coupling
			c_x = np.random.normal(loc = mean_coupling,scale=std_c,size=L)
		
	z_list=[[c_x[j],j] for j in range(L)]
	static = [['xx',interactions_z],['zz',interactions_x],['yy',interactions_x],['x',z_list]]

	#kick_list = [[1.0,j] for j in range(L)]
	#kick_list_corr = [[0.1,j] for j in range(L)]

	#static_x = [[kick_dir,kick_list]]


	H = hamiltonian(static,[],basis=basis,dtype=np.complex128,check_symm=False)
	#Hx = hamiltonian(static_x,[],basis=basis,dtype=np.complex128)

	return H #, Hx


def construct_detuning(basis,L,off_set_z):
		
	z_list=[[off_set_z,j] for j in range(L)]
	static = [['x',z_list]]

	H_detune = hamiltonian(static,[],basis=basis,dtype=np.complex128,check_symm=False)

	return H_detune 


def construct_extra(basis,L,off_set_z):
		
	z_list=[[off_set_z,j] for j in range(L)]
	static = [['x',z_list]]

	H_detune = hamiltonian(static,[],basis=basis,dtype=np.complex128,check_symm=False)

	return H_detune 


def construct_dynamic(basis,L,application_time_dipolar,application_time_kick,nr_of_reps,amplitude_AC,median_coupling,function,kick_dir,kick_angle,off_set_z,runge_kutta=False):

	# get the time span during which we need to integrate 'function'	
	# the sequence looks like ( U_dd U_x )^N
	# dynamic kick applied during U_dd
	
	amplitude_AC = amplitude_AC * median_coupling 		# amplitude AC measured in units of the median coupling

	omega = 2*np.pi/((nr_of_reps)*(application_time_dipolar+application_time_kick))
	
	Hz_lst = []
	cplngs = np.zeros(nr_of_reps)
	
	s = 0
	for n in range(nr_of_reps):
		t_low = n*(application_time_dipolar+application_time_kick)+ application_time_kick
		t_high = t_low + application_time_dipolar

		# integrate 'function' from t_low to t_high
		cplng=integrate.quad(lambda x: function(omega*x),t_low,t_high)[0]
		kick_list = [[amplitude_AC*cplng,j] for j in range(L)]
		static_z = [['x',kick_list]]
		Hz_lst.append(hamiltonian(static_z,[],basis=basis,dtype=np.complex128,check_symm=False))
		print(amplitude_AC*cplng/np.pi)
		s += amplitude_AC*cplng/np.pi
		
	print('integrated amplitude sum is: ', s)
	#exit()
	# kick part
	H_kick_lst = []
	if runge_kutta ==False:
		kick_list = [[kick_angle,j] for j in range(L)]
		kick_list_off_set =[[off_set_z*application_time_kick,j] for j in range(L)]
		static_x = [[kick_dir,kick_list],['x',kick_list_off_set]]
		for n in range(nr_of_reps):
			H_kick_lst.append( hamiltonian(static_x,[],basis=basis,dtype=np.complex128,check_symm=False) )

	elif runge_kutta==True:
		ampl_z = amplitude_AC
		ampl_x = kick_angle / application_time_kick
		if kick_dir == 'z':
			sigma = np.array([[1,0],[0,-1]])
		elif kick_dir == 'y':
			sigma = np.array([[0,-1j],[1j,0]])
		else:
			sigma = np.array([[0,1],[1,0]])
			print('-- kick dir probably wrong --')

		tls = lambda x,psi_in,ampl_x,ampl_z : -1j*( ampl_x * sigma+ ampl_z*np.array([[0,1],[1,0]])*(function(omega*x)+off_set_z/ampl_z) ) @ psi_in


		for n in range(nr_of_reps):
			t_low =n*(application_time_dipolar+application_time_kick)
			t_high = t_low +application_time_kick
			_, ampl = reconstruct_matrix(t_low, t_high, 0.001,tls,ampl_x,ampl_z)
 			

			kick_list_x = [[ampl[2].real,j] for j in range(L)]
			kick_list_y = [[ampl[3].real,j] for j in range(L)]
			kick_list_z = [[ampl[1].real,j] for j in range(L)]
			static_xyz = [['x',kick_list_x],['y',kick_list_y],['z',kick_list_z]] 
			H_kick_lst.append( hamiltonian(static_xyz,[],basis=basis,dtype=np.complex128,check_symm=False) )

	return Hz_lst, H_kick_lst




def compute_observables(j,psi,L,obs,O):
	# updates variables in-place
	for k in range(len(obs)):
		obs[k,j+1] = O[k].expt_value(psi).real/L 




def estimate_scales(basis,L,H,delta_t=0.0005,time_steps=1000):
		
	"""estimates relevant time scales via decay of spins using the fully polarized state as init state """
	
	psi_i = np.zeros(basis.Ns)
	psi_i[basis.index('1'*L)]=1.0

	O=[hamiltonian([['z',[[1.0,j] for j in range(L)] ],],[],basis=basis,dtype=np.complex128,check_symm=False)]	

	observable=np.zeros((1,time_steps+1),dtype=np.float64)

	compute_observables(-1,psi_i,L,observable,O)
	psi=psi_i.copy().astype(np.complex128)
		
	expH = expm_multiply_parallel(H.tocsr(),a=-1j*delta_t)
	work_array=np.zeros((2*len(psi),), dtype=psi.dtype)
	for j in range(time_steps):
		expH.dot(psi,work_array=work_array,overwrite_v=True)
				
		compute_observables(j,psi,L,observable,O)

	intersect=np.abs(np.abs(observable[0]) - np.exp(-1)).argmin()
	median_coupling = 1/(intersect*delta_t)

	return median_coupling  


def fibonacci_seq(n):
	f_min1 =np.array([0])
	f_current =np.array([1])
	for n in range(n):
		f_temp = f_current
		f_current = np.concatenate((f_current,f_min1))
		f_min1 = f_temp

	return f_current


def random_seq(n,seed):
	with temp_seed(seed):
		seq = np.array(list(map(np.round,np.random.uniform(0,1,n))))

	return seq



def Runge_Kutta(time_step,TLS,psi_init,t_init,ampl_x,ampl_z):
	
	# computes a single RK time-step given some two-level-system at t_init with psi_init

	k1 = TLS(t_init, psi_init,ampl_x,ampl_z)
	k2=  TLS(t_init +time_step/2, psi_init + time_step/2*k1,ampl_x,ampl_z)
	k3 = TLS(t_init +time_step/2, psi_init + time_step/2*k2,ampl_x,ampl_z)
	k4 = TLS(t_init + time_step, psi_init + time_step*k3,ampl_x,ampl_z)

	new_psi = psi_init + 1/6*time_step*(k1+2*k2+2*k3+k4)
	new_time = t_init + time_step
	return new_time, new_psi

def time_evolve_RK(t_init, t_final, time_step, TLS, psi_init,ampl_x,ampl_z):

	# time-evolution with RK approximation

	n_of_steps = int((t_final -t_init)//time_step)

	time = t_init
	for n in range(n_of_steps):
		time, psi_init = Runge_Kutta(time_step,TLS,psi_init,time,ampl_x,ampl_z)

	return psi_init / np.linalg.norm(psi_init)

def decompose_TLS(matrix):
	
	# decomposes a 2x2 matrix into a*sigma_0 + b*sigma_z + c*sigma_x + d*sigma_y and returns [a,b,c,d]
	m = np.linalg.inv(np.array([[1,1,0,0],[0,0,1,-1j],[0,0,1,1j],[1,-1,0,0]]))
	return m @ np.array([matrix[0,0],matrix[0,1],matrix[1,0],matrix[1,1]])

def reconstruct_matrix(t_init, t_final, time_step,TLS,ampl_x,ampl_z):

	# reconstruct the effective amplitudes in the different driections x,y,z after solving with Runge-Kutta
	
	time_evolution_matrix = np.zeros((2,2),dtype=np.complex128)
	for i in range(2):
		psi_init = np.zeros(2)
		psi_init[i]=1

		psi = time_evolve_RK(t_init, t_final, time_step, TLS, psi_init,ampl_x,ampl_z)

		time_evolution_matrix[i]=psi

	time_evolution_matrix = np.transpose(time_evolution_matrix)

	Heff = 1j*logm(time_evolution_matrix)

	amplitudes = decompose_TLS(Heff)

	return Heff, amplitudes




if __name__ == '__main__':

	# solution for a generic timedependent two-level-system using a Runge-Kutta approximation
	'''
	def TLS(t,psi_in,ampl_x,ampl_z):
		#ampl_z*np.array([[1,0],[0,-1]])*np.sin(t) 
		return -1j*( ampl_x * np.array([[0,1],[1,0]])+ ampl_z*np.array([[1,0],[0,-1]])*np.sin(t) ) @ psi_in

	H_eff, ampl = reconstruct_matrix(0, 1, 0.001,TLS,np.pi/4,0.0)
	
	print(H_eff)
	print(ampl[2])
	print(np.pi/4)
	exit()
	'''
	

	# system parameters -- parameters to build the random graph 
	dimension=3
	L=12
	min_dist=0.9
	max_dist=1.1
	seed =10


	# driving parameters -- parameters to specify the convolved drive
	# smooth funtion to be used -- any function with a single argument, for the simulation we use values x in [0,2*np.pi]
	
	
	def continuous(x):
		'''
		if x > 2*np.pi*0.99:
			#return 0.67
			#return 2*0.67
			#return 4*0.67
			return 1
		else:
			return 0
		'''
		return 0
	
	'''
	def continuous(x):
		return np.sin(x)
	'''


	angle = np.pi			# kick angle of the for the step-kick -- note we are using pauli matrices not spin matrices 
	direction='z' 			# direction of the step kick (note that we use a pi/2 rotation to the experiment for simulation purposes i.e. x->z, z->-x, y->y)
	t = 0.66 				# the percent of one period that goes down to perform the kick -- ranges from almost 0 to < 1 -- needs to be adjusted with real values
	ratio=0.25				# J*tau = ratio -- dimensionless value quantifying the energy scales

	repetition=1		# number of kicks to be done to complete a full period of the smooth function 
	amplitude_AC=0#87#200	# amplitude of the smooth function in units of the median coupling
	detuning = 0.0#0.5#1			# in units of the median coupling
	continuous_params = (detuning,)

	kick_seq = [ratio,angle,t,direction,repetition,amplitude_AC,continuous,detuning,continuous_params]
	runge_kutta = False

	nv_sys = nv_system(dimension,L,min_dist,max_dist,seed)
	nv_ham= nv_hamiltonian(kick_seq,runge_kutta,dimension,L,min_dist,max_dist,seed)


	psi_i=np.zeros(nv_ham.basis.Ns,dtype=np.complex128)
	psi_i[nv_ham.basis.index('1'*L)]=1.0



	save_dir = '../ARP_dir/'
	file_name_head = 'ARP_data_long'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	alldata = []
	a_list = []
	for j in range(9,10):
		
		a=0.49*j/10
		a_list += [a]
		def kick_func(x):
			if x<a:
				return 0.8
			elif x>=a and x<1-a:
				return 0.8+(x-a)*0.4/(1-2*a)
			else:
				return 1.2

		observable=nv_ham.evolve(psi_i,2000,save_dir,file_name_head,N_d=1,save_every=1000,Noise=0.05,kick_func = kick_func, save_state=False,drive='periodic', multipole='standard')
	
		alldata += [observable]


	with open(save_dir + file_name_head +'.pkl', 'wb') as handle:
		pickle.dump([alldata,a_list], handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	for i in range(len(alldata)):	
		plt.plot(alldata[i][2],label='a={}'.format(a_list[i]))
	plt.legend()
	plt.xscale('log')
	plt.show()

	plt.plot(observable[1])
	plt.xscale('log')
	plt.show()

	plt.plot(observable[2])
	plt.xscale('log')
	plt.show()

	exit()
	for r in range(repetition):
		plt.plot(observable[0,r::repetition])
	plt.xscale('log')
	plt.xlabel('$t  f_{{AC}}$',fontsize=16)
	plt.ylabel('$ I_x $',fontsize=16)
	plt.show()



	#plt.plot(observable[0,5::repetition],observable[1,5::repetition],'s')
	#plt.show()


	plt.plot(observable[0,-150*4:-150*4+100],observable[1,-150*4:-150*4+100],'-s')
	plt.show()

	plt.plot(observable[2,-150*4:-150*4+100],observable[1,-150*4:-150*4+100],'-s')
	plt.show()
	
	plt.plot(observable[0,-150*4:-150*4+100],observable[2,-150*4:-150*4+100],'-s')
	plt.show()


	plt.plot(np.abs(observable[0,-100:]),observable[1,-100:],'-s')
	plt.show()

	A = np.transpose(np.array( [ np.zeros(len(observable[0,-150*4:-150*4+100])),np.zeros(len(observable[0,-150*4:-150*4+100])),np.zeros(len(observable[0,-150*4:-150*4+100])),observable[2,-150*4:-150*4+100],observable[1,-150*4:-150*4+100],observable[0,-150*4:-150*4+100] ] ) ) 

	def get_fix_mins_maxs(mins, maxs):
	    deltas = (maxs - mins) / 12.
	    mins = mins + deltas / 4.
	    maxs = maxs - deltas / 4.
	    
	    return [mins, maxs]

	x,y,z,u,v,w = zip(*A)
	#from mpl_toolkits.mplot3d import Axes3D 
	fig = plt.figure()
	#ax = fig.gca(projection='3d')
	ax = fig.add_subplot(111, projection='3d')
	minmax = get_fix_mins_maxs(-0.9, 0.9)
	ax.set_xlim(minmax)
	ax.set_ylim(minmax) 
	ax.set_zlim(minmax) 

	ax.quiver(x, y, z, u, v, w,arrow_length_ratio=0.1)
	plt.show()




