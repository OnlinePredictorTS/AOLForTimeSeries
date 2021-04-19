# Experiment class
# experiments init file

from tigerforecast.experiments.metrics import *
from tigerforecast.experiments.precomputed import recompute, load_prob_method_to_result
from environment.RealCore import run_experiments, get_ids, to_dict, create_full_problem_to_methods
from tigerforecast.experiments.new_experiment import NewExperiment
from tigerforecast.experiments import precomputed
import csv
import jax.numpy as np
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tigerforecast.utils.optimizers import *

class RealExperiment(object):
    ''' Description: Streamlines the process of performing experiments and comparing results of methods across
             a range of problems. '''
    def __init__(self):
        self.initialized = False
        
    #@profile
    def initialize(self, problems = None, methods = None, problem_to_methods = None, metrics = ['mse'], \
                   n_runs = 1, use_precomputed = False, timesteps = None, verbose = 0):
        '''
        Description: Initializes the experiment instance. 

        Args:
            problems (dict/list): map of the form problem_id -> hyperparameters for problem or list of problem ids;
                                  in the latter case, default parameters will be used for initialization
            methods (dict/list): map of the form method_id -> hyperparameters for method or list of method ids;
                                in the latter case, default parameters will be used for initialization
            problem_to_methods (dict) : map of the form problem_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       problem in problem_to_params
            metrics (list): Specifies metrics we are interested in evaluating.
            n_runs (int): Specifies the number of experiments to average over.
            use_precomputed (boolean): Specifies whether to use precomputed results.
            timesteps (int): Number of time steps to run experiment for
            verbose (0, 1, 2): Specifies the verbosity of the experiment instance.
        '''

        self.problems, self.methods = to_dict(problems), to_dict(methods)
        self.problem_to_methods, self.metrics = problem_to_methods, metrics
        self.n_runs, self.use_precomputed = n_runs, use_precomputed
        self.timesteps, self.verbose = timesteps, verbose

        self.n_problems, self.n_methods = {}, {}

        if(problem_to_methods is None):
            self.problem_to_methods = create_full_problem_to_methods(self.problems.keys(), self.methods.keys())

        if(use_precomputed):

            if(timesteps > precomputed.get_timesteps()):
                print("WARNING: when using precomputed results, the maximum number of timesteps is fixed. " + \
                    "Will use %d instead of the specified %d" % (precomputed.get_timesteps(), timesteps))
                self.timesteps = precomputed.get_timesteps()

            # ensure problems and methods don't have specified hyperparameters
            if(type(problems) is dict or type(methods) is dict):
                precomputed.hyperparameter_warning()

            # map of the form [metric][problem][method] -> loss series + time + memory
            self.prob_method_to_result = precomputed.load_prob_method_to_result(\
                problem_ids = list(self.problems.keys()), method_ids = list(self.methods.keys()), \
                problem_to_methods = problem_to_methods, metrics = metrics)
        else:
            self.new_experiment = NewExperiment()
            self.new_experiment.initialize(self.problems, self.methods, problem_to_methods, \
                metrics, n_runs, timesteps, verbose)
            # map of the form [metric][problem][method] -> loss series + time + memory
            self.prob_method_to_result = self.new_experiment.run_all_experiments()

    #@profile
    def add_all_method_variants(self, method_id, method_params = {}, include_boosting = True, lr_tuning = False):
        optimizers = [OGD, Adagrad, Adam]
        for optimizer in optimizers:
            method_params['optimizer'] = optimizer
            self.add_method(method_id, method_params, lr_tuning = lr_tuning, name = optimizer.__name__)
            if(include_boosting):
                self.add_method('SimpleBoost', {'method_id' : method_id, 'method_params' : method_params},\
                    name = method_id + '-' + optimizer.__name__)
    #@profile            
    def add_method(self, method_id, method_params = None, lr_tuning = False, name = None):
        '''
        Description: Add a new method to the experiment instance.
        
        Args:
            method_id (string): ID of new method.
            method_params (dict): Parameters to use for initialization of new method.
        '''
        assert method_id is not None, "ERROR: No Method ID given."

        if name is None and method_params is not None and 'optimizer' in method_params:
            name = method_params['optimizer'].__name__

        new_id = ''
        if(method_id in self.methods):
            if(method_id not in self.n_methods):
                self.n_methods[method_id] = 0
            self.n_methods[method_id] += 1
            if(name is not None):
                new_id = method_id + '-' + name
            else:
                new_id = method_id + '-' + str(self.n_methods[method_id])
            self.methods[method_id].append((new_id, method_params))
        else:
            new_id = method_id
            if(name is not None):
                new_id += '-' + name
            self.methods[method_id] = [(new_id, method_params)]
            self.n_methods[method_id] = 1

        ''' Evaluate performance of new method on all problems '''
        for metric in self.metrics:
            for problem_id in self.problems.keys():
                for (new_problem_id, problem_params) in self.problems[problem_id]:
                    ''' If method is compatible with problem, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiments((problem_id, problem_params), \
                            (method_id, method_params), metric = metric, lr_tuning = lr_tuning, \
                            n_runs = self.n_runs, timesteps = self.timesteps, verbose = self.verbose)
                    except Exception as e:
                        print(e)
                        print("ERROR: Could not run %s on %s." % (method_id, problem_id) + \
                            " Please make sure method and problem are compatible.")
                        loss, time, memory = 0, 0.0, 0.0

                    self.prob_method_to_result[(metric, new_problem_id, new_id)] = loss
                    self.prob_method_to_result[('time', new_problem_id, new_id)] = time
                    self.prob_method_to_result[('memory', new_problem_id, new_id)] = memory

    #@profile
    def add_problem(self, problem_id, problem_params = None, name = None):
        '''
        Description: Add a new problem to the experiment instance.
        
        Args:
            problem_id (string): ID of new method.
            problem_params (dict): Parameters to use for initialization of new method.
        '''
        assert problem_id is not None, "ERROR: No Problem ID given."

        new_id = ''

        # AN INSTANCE OF THE PROBLEM ALREADY EXISTS
        if(problem_id in self.problems):
            # COUNT NUMBER OF INSTANCES OF SAME MAIN PROBLEM
            if(problem_id not in self.n_problems):
                self.n_problems[problem_id] = 0
            self.n_problems[problem_id] += 1
            # GET ID OF PROBLEM INSTANCE
            if(name is not None):
                new_id = name
            else:
                new_id = problem_id[:-2] + str(self.n_problems[problem_id])
            self.problems[problem_id].append((new_id, problem_params))
        # NO INSTANCE OF THE PROBLEM EXISTS
        else:
            new_id = problem_id[:-3]
            if(name is not None):
                new_id = name
            self.problems[problem_id] = [(new_id, problem_params)]
            self.n_problems[problem_id] = 1

        ''' Evaluate performance of new method on all problems '''
        for metric in self.metrics:
            for method_id in self.methods.keys():
                for (new_method_id, method_params) in self.methods[method_id]:
                    ''' If method is compatible with problem, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiments((problem_id, problem_params), \
                            (method_id, method_params), metric = metric, n_runs = self.n_runs, \
                            timesteps = self.timesteps, verbose = self.verbose)
                    except Exception as e:
                        print(e)
                        print("ERROR: Could not run %s on %s. Please make sure method and problem are compatible." % (method_id, problem_id))
                        loss, time, memory = 0.0, 0.0, 0.0

                    self.prob_method_to_result[(metric, new_id, new_method_id)] = loss
                    self.prob_method_to_result[('time', new_id, new_method_id)] = time
                    self.prob_method_to_result[('memory', new_id, new_method_id)] = memory

    def to_csv(self, table_dict, save_as):
        ''' Save to csv file '''
        with open(save_as, 'w') as f:
            for key in table_dict.keys():
                f.write(key)
                for item in table_dict[key]:
                    f.write(",%s" % str(item))
                f.write('\n')

    def scoreboard(self, metric = 'mse', start_time = 0, n_digits = 3, save_as = None):
        '''
        Description: Show a scoreboard for the results of the experiments for specified metric.

        Args:
            save_as (string): If not None, datapath to save results as csv file.
            metric (string): Metric to compare results
            verbose (boolean): Specifies whether to print the description of the scoreboard entries
        '''

        if(self.use_precomputed and metric == 'time' and len(self.n_methods.keys()) > 0):
            print("WARNING: Time comparison between precomputed methods and" + \
                  "any added method may be irrelevant due to hardware differences.")

        print("Average " + metric + ":")
            
        table = PrettyTable()
        table_dict = {}

        problem_ids = get_ids(self.problems)
        method_ids = get_ids(self.methods)

        table_dict['Problems'] = problem_ids

        field_names = ['Method\Problems']
        for problem_id in problem_ids:
            field_names.append(problem_id)

        table.field_names = field_names

        for method_id in method_ids:
            method_scores = [method_id]
            # get scores for each problem
            for problem_id in problem_ids:
                score = np.mean(self.prob_method_to_result[(metric, problem_id, method_id)][start_time:])
                score = round(float(score), n_digits)
                if(score == 0.0):
                    score = '—'
                method_scores.append(score)
            table.add_row(method_scores)
            table_dict[method_id] = method_scores[1:]

        print(table)

        if(save_as is not None):
            self.to_csv(table_dict, save_as)

    def avg_regret(self, loss):
        avg_regret = []
        cur_avg = 0
        for i in range(len(loss)):
            cur_avg = (i / (i + 1)) * cur_avg + loss[i] / (i + 1)
            avg_regret.append(cur_avg)
        return avg_regret
    
    def regret(self, loss):
        avg_regret = []
        cur_avg = 0
        for i in range(len(loss)):
            cur_avg = cur_avg + loss[i]
            avg_regret.append(cur_avg)
        return avg_regret

    def _plot(self, ax, problem, problem_result_plus_method, n_problems, metric, \
                avg_regret, start_time, cutoffs, yscale, show_legend = True):
        for (loss, method) in problem_result_plus_method:
            if(avg_regret):
                ax.plot(self.avg_regret(loss[start_time:]), label=str(method))
            else:
                ax.plot(self.regret(loss[start_time:]), label=str(method))
        if(show_legend):
            ax.legend(loc="upper right", fontsize=5 + 6//(n_problems+2))
        ax.set_title("Problem:" + str(problem))
        #ax.set_xlabel("timesteps")
        ax.set_ylabel(metric)

        if(cutoffs is not None and problem in cutoffs.keys()):
            ax.set_ylim([0, cutoffs[problem]])

        if(yscale is not None):
            ax.set_yscale(yscale)

        return ax

    def graph(self, problem_ids = None, metric = 'mse', avg_regret = True, start_time = 0, \
            cutoffs = None, yscale = None, time = 20, save_as = None, size = 3, dpi = 100, save_csv_path = None):

        '''
        Description: Show a graph for the results of the experiments for specified metric.
        
        Args:
            save_as (string): If not None, datapath to save the figure containing the plots
            metric (string): Metric to compare results
            time (float): Specifies how long the graph should display for
        '''

        # check metric exists
        assert metric in self.metrics

        # get problem and method ids
        if(problem_ids is None):
            problem_ids = get_ids(self.problems)
        method_ids = get_ids(self.methods)

        # get number of problems
        n_problems = len(problem_ids)

        all_problem_info = []

        for problem_id in problem_ids:
            problem_result_plus_method = []
            method_list = []
            for method_id in method_ids:
                method_list.append(method_id)
                problem_result_plus_method.append((self.prob_method_to_result[(metric, problem_id, method_id)], method_id))
            all_problem_info.append((problem_id, problem_result_plus_method, method_list))

        nrows = max(int(np.sqrt(n_problems)), 1)
        ncols = n_problems // nrows + n_problems % nrows

        fig, ax = plt.subplots(figsize = (ncols * size, nrows * size), nrows=nrows, ncols=ncols)
        fig.canvas.set_window_title('Experiments')

        if n_problems == 1:
            (problem, problem_result_plus_method, method_list) = all_problem_info[0]
            if(save_csv_path is not None):
                self.problemToCSV(problem, problem_result_plus_method, metric, avg_regret, save_csv_path)
            ax = self._plot(ax, problem, problem_result_plus_method, n_problems, \
                metric, avg_regret, start_time, cutoffs, yscale)
        elif nrows == 1:
            for j in range(ncols):
                (problem, problem_result_plus_method, method_list) = all_problem_info[j]
                if(save_csv_path is not None):
                    self.problemToCSV(problem, problem_result_plus_method, metric, avg_regret, save_csv_path)
                ax[j] = self._plot(ax[j], problem, problem_result_plus_method, n_problems, \
                                          metric, avg_regret, start_time, cutoffs, yscale)
        else:
            cur_pb = 0
            for i in range(nrows):
                for j in range(ncols):

                    if(cur_pb == n_problems):
                        legend = []
                        for method_id in method_ids:
                            legend.append((0, method_id))
                        ax[i, j] = self._plot(ax[i, j], 'LEGEND', legend,\
                                n_problems, metric, False, cutoffs, None, show_legend = True)
                        continue

                    if(cur_pb > n_problems):
                        ax[i, j].plot(0, 'x', 'red', label="NO MORE \n MODELS")
                        ax[i, j].legend(loc="center", fontsize=8 + 10//n_problems)
                        continue

                    (problem, problem_result_plus_method, method_list) = all_problem_info[cur_pb]
                    if(save_csv_path is not None):
                        self.problemToCSV(problem, problem_result_plus_method, metric, avg_regret, save_csv_path)
                    cur_pb += 1
                    ax[i, j] = self._plot(ax[i, j], problem, problem_result_plus_method,\
                                n_problems, metric, avg_regret, start_time, cutoffs, yscale, show_legend = False)

        #fig.tight_layout()
        plt.yscale('log')
        if(save_as is not None):
            plt.savefig(save_as, dpi=dpi)

        if time:
            plt.show(block=False)
            plt.pause(time)
            plt.close()
        else:
            plt.show()

    def __str__(self):
        return "<Experiment Method>"

    def problemToCSV(self, problem_id, problem_result_plus_method, metric, avg_regret, save_as):
        print("Store CSV")
        for method in problem_result_plus_method:
            f = open(save_as + "_" + problem_id + "_" + method[1] + ".csv", "w+")
            header = "step" + ";" + metric
            f.write(header + os.linesep)
            step = 1
            loss = method[0]
            for x in loss:
                line = str(step) + ";" + str(x)
                step += 1
                f.write(line + os.linesep)
            f.close()
