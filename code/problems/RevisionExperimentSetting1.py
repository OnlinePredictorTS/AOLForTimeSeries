import jax
import jax.numpy as np
import jax.random as random

import tigerforecast
from tigerforecast.utils import generate_key
from tigerforecast.problems import Problem


class RevisionExperimentSetting1(Problem):
	"""
	Description: Simulates an autoregressive moving-average time-series.
	"""

	compatibles = set(['RevisionExperimentSetting1-v0', 'TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.has_regressors = False

	def initialize(self, p=5, q=2, n = 10, d=1, noise_list = None, c=0, noise_magnitude=0.1, noise_distribution = 'normal'):
		"""
		Description: Randomly initialize the hidden dynamics of the system.
		Args:
			p (int/numpy.ndarray): Autoregressive dynamics. If type int then randomly
				initializes a Gaussian length-p vector with L1-norm bounded by 1.0. 
				If p is a 1-dimensional numpy.ndarray then uses it as dynamics vector.
			q (int/numpy.ndarray): Moving-average dynamics. If type int then randomly
				initializes a Gaussian length-q vector (no bound on norm). If p is a
				1-dimensional numpy.ndarray then uses it as dynamics vector.
			n (int): Dimension of values.
			c (float): Default value follows a normal distribution. The ARMA dynamics 
				follows the equation x_t = c + AR-part + MA-part + noise, and thus tends 
				to be centered around mean c.
		Returns:
			The first value in the time-series
		"""
		self.initialized = True
		self.T = 0
		self.max_T = -1
		self.n = n
		self.d = d
		self.p = p
		self.q = q

		self.phi = [random.uniform(generate_key(), shape = (10, 10), minval = -0.025, maxval = 0.025),
			  random.uniform(generate_key(), shape = (10, 10), minval = -0.025, maxval = 0.025),
			  random.uniform(generate_key(), shape = (10, 10), minval = -0.025, maxval = 0.025),
			  random.uniform(generate_key(), shape = (10, 10), minval = -0.025, maxval = 0.025),
			  random.uniform(generate_key(), shape = (10, 10), minval = -0.025, maxval = 0.025),
			  ]
		
		self.psi = [random.uniform(generate_key(), shape = (10, 10), minval = -0.025, maxval = 0.025),
			  random.uniform(generate_key(), shape = (10, 10), minval = -0.025, maxval = 0.025)]


		parts = []
		s_sum = 0.0
		for i in range(len(self.phi)):
			u, s, vh = np.linalg.svd(self.phi[i])
			parts.append((u, s, vh))
			s_sum += np.sum(s)
		
		if(s_sum > 1):
			for i in range(len(self.phi)):
				self.phi[i] = (parts[i][0] * (parts[i][1] / s_sum)) @ parts[i][2]

		parts = []
		s_sum = 0.0
		for i in range(len(self.psi)):
			u, s, vh = np.linalg.svd(self.psi[i])
			parts.append((u, s, vh))
			s_sum += np.sum(s)
		
		if(s_sum > 1):
			for i in range(len(self.psi)):
				self.phi[i] = (parts[i][0] * (parts[i][1] / s_sum)) @ parts[i][2]

		self.phi = np.array(self.phi)
		self.psi = np.array(self.psi)

		self.noise_magnitude, self.noise_distribution = noise_magnitude, noise_distribution
		self.c = random.normal(generate_key(), shape=(self.n,)) if c == None else c
		#self.x = random.normal(generate_key(), shape=(self.p + d, self.n))
		self.x = random.normal(generate_key(), shape=(self.p + self.d, self.n))
		#if self.d>1:
		#	self.delta_i_x = random.normal(generate_key(), shape=(self.d-1, self.n)) 
		#else:
		#	self.delta_i_x = None
		
		self.noise_list = None
		if(noise_list is not None):
			self.noise_list = noise_list
			self.noise = np.array(noise_list[0:self.q])
		elif(noise_distribution == 'normal'):
			self.noise = self.noise_magnitude * random.normal(generate_key(), shape=(self.q, self.n)) 
		elif(noise_distribution == 'unif'):
			self.noise = self.noise_magnitude * random.uniform(generate_key(), shape=(self.q, self.n), \
				minval=-1., maxval=1.)
		
		self.feedback=0.0
		
		def _step(x, noise, eps):
			# sum_{i = 1}^q(beta_t-i * epsilon_-i) + eps_t
			x_ma = eps
			for i in range(len(self.psi)):
				x_ma += np.dot(self.psi[i], noise[i])

			next_noise = np.roll(noise, self.n)
			next_noise = jax.ops.index_update(next_noise, 0, eps) # equivalent to self.noise[0] = eps

			#noise sum done
			#add the rest (AR)

			delta_x = np.zeros((self.d + 1, self.p + self.d, self.n))

			#first row (zeroth difference is the original time series)
			delta_x = jax.ops.index_update(delta_x, jax.ops.index[0, 0:x.shape[0], 0: x.shape[1]], x)

			#fill the next rows
			for k in range(1, self.d + 1):
				delta_x = jax.ops.index_update(delta_x, jax.ops.index[k, 0 : self.p + self.d - k, 0 : self.n], delta_x[k - 1, 0 : self.p + self.d - k, 0:self.n] - delta_x[k - 1, 1 : self.p + self.d - k + 1, 0:self.n])

			delta_x_d = delta_x[self.d][0 : self.p]
			x_ar = np.dot(self.phi[0], delta_x_d[0])
			for i in range(1, len(self.phi)):
				x_ar += np.dot(self.phi[i], delta_x_d[i])

			x_new = self.c + x_ar + x_ma
			
			next_x = np.roll(x, self.n)
			
			next_x = jax.ops.index_update(next_x, 0, x_new) # equivalent to self.x[0] = x_new
	
			return (next_x, next_noise, x_new)

		self._step = jax.jit(_step)
		#if self.delta_i_x is not None:
		#	x_delta_sum= np.sum(self.delta_i_x, axis = 0)
		#else:
		#	x_delta_sum= 0
		return self.x[0]#+x_delta_sum

	def step(self):
		"""
		Description: Moves the system dynamics one time-step forward.
		Args:
			None
		Returns:
			The next value in the ARMA time-series.
		"""
		assert self.initialized
						
		self.T += 1
		if(self.noise_list is not None):
			self.x, self.noise, x_new = self._step(self.x, self.noise, self.noise_list[self.q + self.T - 1])
		else:
			if(self.noise_distribution == 'normal'):
				self.x, self.noise, x_new = self._step(self.x, self.noise, \
					self.noise_magnitude * random.normal(generate_key(), shape=(self.n,)))
			elif(self.noise_distribution == 'unif'):
				self.x, self.noise, x_new = self._step(self.x, self.noise, \
					self.noise_magnitude * random.uniform(generate_key(), shape=(self.n,), minval=-1., maxval=1.))

		return x_new

	def hidden(self):
		"""
		Description: Return the hidden state of the system.
		Args:
			None
		Returns:
			(x, eps): The hidden state consisting of the last p x-values and the last q
			noise-values.
		"""
		assert self.initialized
		return (self.x, self.noise)

	def __str__(self):
		return "<ARIMA Problem>"