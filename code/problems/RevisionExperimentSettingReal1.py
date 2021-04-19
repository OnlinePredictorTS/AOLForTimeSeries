import jax
import numpy as np
import numpy.random as random
import os
import tigerforecast
from tigerforecast.utils import generate_key
from tigerforecast.problems import Problem
from pathlib import Path
import csv


class RevisionExperimentSettingReal1(Problem):
	"""
	Description: Simulates an autoregressive moving-average time-series.
	"""

	compatibles = set(['RevisionExperimentSettingReal1-v0', 'TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.has_regressors = False

	def initialize(self, max_T = 1000000000):
		self.initialized = True
		self.T = 0
		self.data = self.load_sp500_multi()
		self.max_T = len(self.data) - 1
		return self.step()

	def load_sp500_multi(self):
		values = []
		Path(os.path.join("..","datasets")).mkdir(parents=True, exist_ok=True)
		acercsv = os.path.join("..","datasets","acer.csv")
		amazoncsv = os.path.join("..","datasets","amazon.csv")
		applecsv = os.path.join("..","datasets","apple.csv")
		intelcsv = os.path.join("..","datasets","intel.csv")
		lenovocsv = os.path.join("..","datasets","lenovo.csv")
		microsoftcsv = os.path.join("..","datasets","microsoft.csv")
		samsungcsv = os.path.join("..","datasets","samsung.csv")
		sp500csv = os.path.join("..","datasets","sp500_short.csv")

		fileList = [acercsv, amazoncsv, applecsv, intelcsv, lenovocsv, microsoftcsv, samsungcsv, sp500csv]

		tsmatrix = []

		for file in fileList:
			values = []
			with open(file) as csvfile:  
				plots = csv.reader(csvfile, delimiter=';')
				for row in list(plots)[1:]:
					if(file == samsungcsv):
						values.append(float(row[1]) / 10000.0)
					else:
						values.append(float(row[1]) / 1000.0)
			tsmatrix.append(values)

		return np.array(tsmatrix).T

	def step(self):
		"""
		Description: Moves the system dynamics one time-step forward.
		Args:
			None
		Returns:
			The next value in the ARMA time-series.
		"""
		assert self.initialized
		if(self.T < self.max_T):
			self.T += 1
		else:
			#maybe other behaviour is wanted here
			pass

		return self.data[self.T]

	def hidden(self):
		assert self.initialized
		return (self.data)

	def __str__(self):
		return "<SP500 Problem>"
