import tigerforecast
import jax.numpy as np
from tigerforecast.utils.optimizers import *
from environment.RealExperiment import RealExperiment as Experiment
import jax.random as random
from tigerforecast.utils import generate_key
from optimizers.RealONS import RealONS
from optimizers.RealOGD import RealOGD
from optimizers.Ftrl4 import Ftrl4
from optimizers.ADAFtrl import ADAFtrl
from losses.AE import ae
from losses.RealSE import se
from predictors.ArimaAutoregressor import ArimaAutoRegressor
from tigerforecast.problems.registration import problem_registry, problem_register, problem
from tigerforecast.problems.custom import register_custom_problem, CustomProblem

from tigerforecast.methods.registration import method_registry, method_register, method
from tigerforecast.methods.custom import CustomMethod, register_custom_method

import datetime

#joblib for parallelizing the runs
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path

import numpy as np

#########################################################################################################################################################
#																																						#
#															SE Settings																					#
#																																						#
#########################################################################################################################################################

def settingSE1(p):

	n = 20
	T = 10000
	dim = 10

	print("Setting 1 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting1-v0', {}, name = 'I')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 1 SE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=se),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=se),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=se),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=se),'n': dim}, name = 'FTRL-4_p_' + str(p) + "_d_" + str(c))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': se, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': se}, name = 'ARIMA-AO-Hedge')
	
	print("Setting 1 SE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting1SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting1SE')


def settingSE2(p):
	n = 20
	T = 10000

	dim = 10

	print("Setting 2 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting2-v0', {}, name = 'II')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 2 SE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=se),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=se),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=se),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=se),'n': dim}, name = 'FTRL-4_p_' + str(p) + "_d_" + str(c))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': se, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': se}, name = 'ARIMA-AO-Hedge')
	
	print("Setting 2 SE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting2SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting2SE')


def settingSE3(p):
	n = 20
	T = 10000

	dim = 10

	print("Setting 3 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting3-v0', {}, name = 'III')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 3 SE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=se),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=se),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=se),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=se),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': se, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': se}, name = 'ARIMA-AO-Hedge')

	print("Setting 3 SE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting3SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting3SE')


def settingSE4(p):
	d = 1
	n = 1
	T = 5320

	dim = 8

	print("Setting 4 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	exp.add_problem('ExperimentSetting4-v0', {}, name = 'IV')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 4 SE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=se),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=se),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=se),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=se),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': se, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': se}, name = 'ARIMA-AO-Hedge')

	print("Setting 4 SE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting4SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting4SE')

def settingSE5(p):
	d = 1
	n = 1
	T = 590

	dim = 10

	print("Setting 5 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting5-v0', {}, name = 'V')
	
	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 5 SE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=se),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=se),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=se),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=se),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': se, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': se}, name = 'ARIMA-AO-Hedge')

	print("Setting 5 SE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting5SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting5SE')


def settingSE6(p):
	dim = 4
	n = 1
	T = 419

	print("Setting 6 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting6-v0', {}, name = 'VI')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 6 SE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=se),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=se),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': d, 'optimizer': ADAFtrl(loss=se),'n': dim}, name = 'ADAFtrl_p_' + str(p) + "_d_" + str(d))
	exp.add_method('ArimaAR', {'p' : p, 'd': d, 'optimizer': Ftrl4(loss=se),'n': dim}, name = 'FTRL-4_p_' + str(p) + "_d_" + str(d))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': se, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': se}, name = 'ARIMA-AO-Hedge')

	print("Setting 6 SE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting6SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting6SE')


#########################################################################################################################################################
#																																						#
#															AE Settings																					#
#																																						#
#########################################################################################################################################################

def settingAE1(p):
	n = 20
	T = 10000
	dim = 10

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting1-v0', {}, name = 'I')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 1 AE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=ae),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=ae),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=ae),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': c, 'optimizer': ADAFtrl(loss=ae),'n': dim}, name = 'ADAFtrl_p_' + str(p) + '_d_' + str(c))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': ae, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': ae}, name = 'ARIMA-AO-Hedge')
	
	print("Setting 1 AE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting1_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting1_AE')

def settingAE2(p):
	n = 20
	T = 10000
	dim = 10

	print("Setting ae 2 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)
	
	exp.add_problem('ExperimentSetting2-v0', name = 'II')
	
	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 2 AE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=ae),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=ae),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))


	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=ae),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=ae),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': ae, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': ae}, name = 'ARIMA-AO-Hedge')

	print("Setting AE 2 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting2_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting2_AE')


def settingAE3(p):
	n = 20
	T = 10000
	dim = 10

	print("Setting ae 3 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting3-v0', name = 'III')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 3 AE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=ae),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=ae),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))


	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=ae),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=ae),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': ae, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': ae}, name = 'ARIMA-AO-Hedge')

	print("Setting AE 3 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting3_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting3_AE')


def settingAE4(p):
	d = 1
	n = 1
	T = 5320

	dim = 8

	print("Setting ae 4 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting4-v0', name = 'IV')
	
	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 4 AE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=ae),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=ae),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=ae),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=ae),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': ae, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': ae}, name = 'ARIMA-AO-Hedge')

	print("Setting AE 4 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting4_AE.pdf", avg_regret = True, size=15, start_time = 100, metric = 'ae', dpi = 100, save_csv_path = store_directory + 'papersetting4_AE')


def settingAE5(p):
	d = 1
	n = 1
	T = 590

	dim = 10

	print("Setting ae 5 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting5-v0', name = 'V')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 5 AE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=ae),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=ae),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': ADAFtrl(loss=ae),'n': dim}, name = 'ADAFtrl_p_' + str(p))
	exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': Ftrl4(loss=ae),'n': dim}, name = 'FTRL-4_p_' + str(p))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': ae, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': ae}, name = 'ARIMA-AO-Hedge')

	print("Setting AE 5 finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting5_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting5_AE')


def settingAE6(p):
	dim = 4
	n = 1
	T = 419

	print("Setting ae 6 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting6-v0', name = 'VI')

	for eta in [1, 0.1, 0.01, 0.001]:
		for c in [1, 2, 4]:
			print("Setting 6 AE c = " + str(c) + " started at " + str(datetime.datetime.now()), flush = True)
			hyp = {'lr': eta, 'c':c}
			exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealOGD(hyperparameters=hyp, loss=ae),'n': dim}, name = 'OGD_p_' + str(p) + "_lr_" + str(eta) + '_c_' + str(c))
			for eps in [1, 0.1, 0.01, 0.001]:
				hyp = {'eta': eta, 'eps': eps, 'c':c}
				exp.add_method('ArimaAR', {'p' : p, 'd': 1, 'optimizer': RealONS(hyperparameters=hyp, loss=ae),'n': dim}, name = 'ONS_p_' + str(p) + "_eta_" + str(eta) + "_eps_" + str(eps) + '_c_' + str(c))

	exp.add_method('ArimaAR', {'p' : p, 'd': d, 'optimizer': ADAFtrl(loss=ae),'n': dim}, name = 'ADAFtrl_p_' + str(p) + "_d_" + str(d))
	exp.add_method('ArimaAR', {'p' : p, 'd': d, 'optimizer': Ftrl4(loss=ae),'n': dim}, name = 'FTRL-4_p_' + str(p) + "_d_" + str(d))
	exp.add_method('JamilAlgo1', {'n': dim, 'loss': ae, 'eta': np.sqrt(T * np.log(96))}, name = 'jamil_aggregation')
	exp.add_method('ARIMAAOHedge', {'n': dim, 'loss': ae}, name = 'ARIMA-AO-Hedge')
	print("Setting ae 6 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting6_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting6_AE')




def run_experiment(i, p):

	method_register(
		id='ArimaAR',
		entry_point='predictors.ArimaAutoregressor:ArimaAutoRegressor',
	)
	
	method_register(
		id='JamilAlgo1',
		entry_point='predictors.Jamil_algo1:JamilAlgo1',
	)

	method_register(
		id='ARIMAAOHedge',
		entry_point='predictors.ARIMA_AO_Hedge:ARIMAAOHedge',
	)
	
	problem_register(
		id='ExperimentSetting1-v0',
	    entry_point='problems.RevisionExperimentSetting1:RevisionExperimentSetting1',
	)

	problem_register(
	    id='ExperimentSetting2-v0',
	    entry_point='problems.RevisionExperimentSetting2:RevisionExperimentSetting2',
	)

	problem_register(
	    id='ExperimentSetting3-v0',
	    entry_point='problems.RevisionExperimentSetting3:RevisionExperimentSetting3',
	)

	problem_register(
	    id='ExperimentSetting4-v0',
	    entry_point='problems.RevisionExperimentSettingReal1:RevisionExperimentSettingReal1',
	)

	problem_register(
	    id='ExperimentSetting5-v0',
	    entry_point='problems.RevisionExperimentSettingReal2:RevisionExperimentSettingReal2',
	)

	problem_register(
		id='ExperimentSetting6-v0',
	    entry_point='problems.RevisionExperimentSettingReal3:RevisionExperimentSettingReal3',
	)
	
	if(i == 1):
		settingSE1(p)
	elif(i == 2):
		settingSE2(p)
	elif(i == 3):
		settingSE3(p)
	elif(i == 4):
		settingSE4(p)
	elif(i == 5):
		settingSE5(p)
	elif(i == 6):
		settingSE6(p)
	elif(i == 11):
		settingAE1(p)
	elif(i == 12):
		settingAE2(p)
	elif(i == 13):
		settingAE3(p)
	elif(i == 14):
		settingAE4(p)
	elif(i == 15):
		settingAE5(p)
	elif(i == 16):
		settingAE6(p)


store_directory = "experiments_results/"
if __name__ == '__main__':
	Path(store_directory).mkdir(parents=True, exist_ok=True)
				
	tasklist = [(1, 8), (1, 16), (1, 32), (1, 64),
				(2, 8), (2, 16), (2, 32), (2, 64),
				(3, 8), (3, 16), (3, 32), (3, 64),
				(4, 8), (4, 16), (4, 32), (4, 64),
				(5, 8), (5, 16), (5, 32), (5, 64),
				(6, 8), (6, 16), (6, 32), (6, 64),
				(11, 8), (11, 16), (11, 32), (11, 64),
				(12, 8), (12, 16), (12, 32), (12, 64),
				(13, 8), (13, 16), (13, 32), (13, 64),
				(14, 8), (14, 16), (14, 32), (14, 64),
				(15, 8), (15, 16), (15, 32), (15, 64),
				(16, 8), (16, 16), (16, 32), (16, 64)
				]
	results = Parallel(n_jobs=len(tasklist))(delayed(run_experiment)(i, p) for (i, p) in tasklist)