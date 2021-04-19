"""
monthly unemployment for the past few decades
"""

import tigerforecast
import os
import jax.numpy as np
import csv
from datetime import datetime
from tigerforecast.problems import Problem

class RevisionExperimentSettingReal3(Problem):
	"""
	Description: Turkey power demand.
	"""

	compatibles = set(['RevisionExperimentSettingReal3-v1', 'TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.has_regressors = False

	def initialize(self):
		# Fetch data
		self.initialized = True
		self.T = 0
		self.data = self.turkey()# get data
		self.max_T = self.data.shape[0]

		return self.data[0]

	def turkey(self):

		turkeyCsv = os.path.join("..","datasets","turkey_power_demand.csv")
		values = []
		with open(turkeyCsv) as csvfile:  
			plots = csv.reader(csvfile, delimiter=';')
			for row in list(plots)[1:]:
				values.append([])
				values[-1].append(float(row[1]) / 10000.0)
				values[-1].append(float(row[2]) / 10000.0)
				values[-1].append(float(row[3]) / 10000000000.0)
				values[-1].append(float(row[4]) / 10000000000.0)

		data = np.array(values)

		return data

	def step(self):
		"""
		Description: Moves time forward by one day and returns price of the bitcoin
		Args:
			None
		Returns:
			The next bitcoin price
		"""
		assert self.initialized
		self.T += 1
		if self.T == self.max_T: 
			raise StepOutOfBounds("Number of steps exceeded length of dataset ({})".format(self.max_T))
		return self.data[self.T]

	def hidden(self):
		"""
		Description: Return the date corresponding to the last unemployment rate value
		Args:
			None
		Returns:
			Date (string)
		"""
		assert self.initialized
		return self.data

	def close(self):
		"""
		Not implemented
		"""
		pass

	def __str__(self):
		return "<Google Flu Problem>"
