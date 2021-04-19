import tigerforecast
import jax
import jax.numpy as np
import jax.experimental.stax as stax
from tigerforecast.utils.random import generate_key
from tigerforecast.methods import Method
from tigerforecast.utils.optimizers import * 
from tigerforecast.utils.optimizers.losses import mse

class ArimaAutoRegressor(Method):
	"""
	Description: Implements the autoregressor extended to use the arima model.
	"""
	
	compatibles = set(['TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.uses_regressors = False

	def initialize(self, n = 1, m = None, p = 3, d = 1, optimizer = OGD):
		"""
		Description: Initializes autoregressive method parameters

		Args:
			n (int): dimension of the data
			p (int): Length of history used for prediction
			d (int): number of difference orders to use. For zero the original autoregressor is used.
			optimizer (class): optimizer choice
		"""
		self.initialized = True
		self.n = n
		self.original_p = p
		self.p = p + d
		self.d = d

		self.lastx = None

		self.past = np.zeros((self.p * self.n, )) #store the last d x values to compute the differences

		glorot_init = stax.glorot() # returns a function that initializes weights

		# self.params = glorot_init(generate_key(), (p+1,1))
		#self.params = {'phi' : glorot_init(generate_key(), (p + 1,1))}
		#self.params = {'phi' : glorot_init(generate_key(), (n, self.original_p * n))}
		self.params = {'phi' : np.zeros((n, self.original_p * n))}
		def _update_past(self_past, x):
			new_past = np.roll(self_past, self.n)
			new_past = jax.ops.index_update(new_past, jax.ops.index[0: self.n], x)
			return new_past
		self._update_past = jax.jit(_update_past)

		'''	unused (easy, but also inefficient version)
		def _computeDifference(x, d):
			if(d < 0):
				return 0.0
			if(d == 0):
				return x[0]
			else:
				return _computeDifference(x, d - 1) - _computeDifference(x[1:], d - 1)
		self._computeDifference = jax.jit(_computeDifference)'''

		def _getDifference(index, d, matrix):
			if(d < 0):
				return np.zeros(matrix[0, 0].shape)
			else:
				return matrix[d, index]
		self._getDifference = jax.jit(_getDifference)

		def _computeDifferenceMatrix(x):
			x_block = x.reshape(self.p, self.n)
			result = np.zeros((self.d + 1, x_block.shape[0], x_block.shape[1]))

			#first row (zeroth difference is the original time series)
			result = jax.ops.index_update(result, jax.ops.index[0, 0:x_block.shape[0], 0:x_block.shape[1]], x_block)
			
			#fill the next rows
			for k in range(1, self.d + 1):
				result = jax.ops.index_update(result, jax.ops.index[k, 0:x_block.shape[0] - k], result[k - 1, 0:x_block.shape[0] - k] - result[k - 1, 1 : len(x_block) - k + 1])

			return result
		self._computeDifferenceMatrix = jax.jit(_computeDifferenceMatrix)

		def _predict(params, x):
			if(x.shape[0]==self.n*self.original_p):
				phi = list(params.values())[0]
				return np.dot(phi, x).squeeze()
			else:
				phi = list(params.values())[0]
				differences = _computeDifferenceMatrix(x)
				#x_plus_bias = np.vstack((np.ones((1, self.n)), np.array([_getDifference(i, self.d, differences) for i in range(0, p)])))
				x_plus_bias =  np.array([_getDifference(i, self.d, differences) for i in range(0, self.original_p)]).flatten()
				return np.dot(phi, x_plus_bias).squeeze() + np.sum(np.array([_getDifference(0, k, differences) for k in range(self.d)]), axis = 0)
		
		self._predict = jax.jit(_predict)

		"""
		def _getUpdateValues(x):
			diffMatrix = _computeDifferenceMatrix(x)
			differences = np.array([_getDifference(i, self.d, diffMatrix) for i in range(0, self.original_p)])
			#differences = np.array([_getDifference(i, self.d, diffMatrix) for i in range(1, self.p + 1)])
			label = _getDifference(0, self.d, diffMatrix)
			return differences, label
		self._getUpdateValues = jax.jit(_getUpdateValues)
		"""

		self._store_optimizer(optimizer, self._predict)


	def predict(self, x):
		"""
		Description: Predict next value given observation
		Args:
			x (int/numpy.ndarray): Observation
		Returns:
			Predicted value for the next time-step
		"""
		assert self.initialized, "ERROR: Method not initialized!"

		self.past = self._update_past(self.past, x) # squeeze to remove extra dimensions
		#print(self.past)
		#print(x)
		#print(self._computeDifferenceMatrix(self.past))
		#pred = self._predict(self.params, self.past)
		#print(pred.shape)
		#print("1")
		#print(x)
		return self._predict(self.params, self.past)

	def forecast(self, x, timeline = 1):
		"""
		Description: Forecast values 'timeline' timesteps in the future
		Args:
			x (int/numpy.ndarray):  Value at current time-step
			timeline (int): timeline for forecast
		Returns:
			Forecasted values 'timeline' timesteps in the future
		"""
		assert self.initialized, "ERROR: Method not initialized!"

		self.past = self._update_past(self.past, x)
		past = self.past.copy()
		pred = []

		for t in range(timeline):
			x = self._predict(self.params, past)
			pred.append(x)
			past = self._update_past(past, x) 

		return np.array(pred)

	def update(self, y):
		"""
		Description: Updates parameters using the specified optimizer
		Args:
			y (int/numpy.ndarray): True value at current time-step
		Returns:
			None
		"""
		assert self.initialized, "ERROR: Method not initialized!"
		
		differences = self._computeDifferenceMatrix(self.past)
		_x = np.array([differences[self.d,i] for i in range(0, self.original_p)]).flatten()
		_y=y-np.sum(np.array([differences[k,0] for k in range(self.d)]), axis = 0)
		#print(y)
		#differences, label = self._getUpdateValues(self.past)
		#self.params = self.optimizer.update(self.params, differences, label)

		#print(self.params)
		#print(differences)
		#print(label)

		self.params = self.optimizer.update(self.params,_x, _y)

	def get_past(self):
		return self.past