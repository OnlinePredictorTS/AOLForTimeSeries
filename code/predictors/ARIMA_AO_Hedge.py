import random
import tigerforecast
import jax
import jax.numpy as np
import jax.experimental.stax as stax
from tigerforecast.utils.random import generate_key
from tigerforecast.methods import Method
from tigerforecast.utils.optimizers import * 
from tigerforecast.utils.optimizers.losses import mse
from optimizers.Ftrl4 import Ftrl4
from predictors.ArimaAutoregressor import ArimaAutoRegressor

#Hedge based on http://proceedings.mlr.press/v32/steinhardtb14.pdf

class ARIMAAOHedge(Method):
	"""
	Description: Implements the equivalent of an AR(p) method - predicts a linear
	combination of the previous p observed values in a time-series
	"""
	
	compatibles = set(['TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.uses_regressors = False
		self.experts=[]
		self.p=96

	def initialize(self, n = 1, m = None, p = 96, eta = 1, loss = mse, d = -1):
		self.initialized = True
		for i in range(32):
			ar= ArimaAutoRegressor()
			optimizer = Ftrl4(loss = loss,hyperparameters={})
			ar.initialize(n=n, m=m, p=i+1, optimizer=optimizer, d = 0)        
			self.experts.append(ar)
		for i in range(32):
			ar= ArimaAutoRegressor()
			optimizer = Ftrl4(loss = loss,hyperparameters={})
			ar.initialize(n=n, m=m, p=i+1, optimizer=optimizer, d = 1)        
			self.experts.append(ar)
		for i in range(32):
			ar= ArimaAutoRegressor()
			optimizer = Ftrl4(loss = loss,hyperparameters={})
			ar.initialize(n=n, m=m, p=i+1, optimizer=optimizer, d = 2)        
			self.experts.append(ar)
		self.p = self.experts[-1].p
		self.n = n
		self.theta = np.zeros((p, ))
		self.eta = np.ones((p, )) * eta
		self.y = np.zeros((p, n))
		self.h = np.zeros((p, ))
		self.z = np.zeros((p, ))
		self.w = np.ones((p, )) / p
		self.loss = loss
		self.d = d

		self.h_z_diff_sum = 0

		self.c = np.zeros((p, ))

		self.T = 0


		def _getDifference(index, d, matrix):
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

	def predict(self, x):
		"""
		Description: Predict next value given observation
		Args:
			x (int/numpy.ndarray): Observation
		Returns:
			Predicted value for the next time-step
		"""
		assert self.initialized, "ERROR: Method not initialized!"
		for i in range(len(self.experts)):
			self.y=jax.ops.index_update(self.y, i, self.experts[i].predict(x))
		return np.dot(self.y.T, self.w)

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
		_k = random.randint(0, self.p-1)
		return self.experts[_k].forecast(x,timeline)

	def update(self, y):
		"""
		Description: Updates parameters using the specified optimizer
		Args:
			y (int/numpy.ndarray): True value at current time-step
		Returns:
			None
		"""
		assert self.initialized, "ERROR: Method not initialized!"

		self.T += 1

		for i in range(len(self.experts)):
			self.z = jax.ops.index_update(self.z, i, self.experts[i].optimizer.loss(y, self.y[i]))
			self.experts[i].update(y)

		past = self.experts[-1].get_past()

		if(self.d == 0):
			Y = np.zeros((len(self.experts), self.n))#np.sum
		else:
			differences = self._computeDifferenceMatrix(past)
			Y = np.sum(np.array([self._getDifference(0, k, differences) for k in range(self.d)]), axis = 0)

		self.h = self.loss(Y, self.y)

		self.theta -= self.z

		self.c = np.max(self.theta / self.eta)

		self.w = np.exp((self.theta - self.h) / self.eta - self.c)
		self.w = self.w/np.sum(self.w)


		self.h_z_diff_sum += (np.max(self.h - self.z) ** 2)

		self.eta = np.sqrt(1 / (2 * np.log(len(self.experts))) * self.h_z_diff_sum)