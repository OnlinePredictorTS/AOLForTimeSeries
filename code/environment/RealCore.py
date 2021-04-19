# experiments core class

import tigerforecast
from tigerforecast.experiments import metrics as metrics_module
from tigerforecast import error
import jax
import jax.numpy as np
from tigerforecast.problems import Problem
from tigerforecast.methods import Method
from tigerforecast.utils.random import set_key
from tqdm import tqdm
import inspect
import time
import operator
from losses.AE import ae
from tigerforecast.utils.autotuning import GridSearch
import traceback

metrics = {'mse': metrics_module.mse, 'cross_entropy': metrics_module.cross_entropy, 'ae' : ae}

#@profile
def to_dict(x):
    '''
    Description: If x is not a dictionary, transforms it to one by assigning None values to entries of x;
                 otherwise, returns x.

    Args:     
        x (dict / list): either a dictionary or a list of keys for the dictionary

    Returns:
        A dictionary 'version' of x
    '''
    if(x is None):
        return {}
    elif(type(x) is not dict):
        x_dict = {}
        for key in x:
            x_dict[key] = [(key, None)]
        return x_dict
    else:
        return x

def get_ids(x):
    '''
    Description: Gets the ids of problems/methods

    Args:
        x (list / dict): list of ids of problems/methods or dictionary of problems/methods and parameters
    Returns:
        x (list): list of problem/methods ids
    '''
    if(type(x) is dict):
        ids = []
        for main_id in x.keys():
            for (custom_id, _) in x[main_id]:
                ids.append(custom_id)
        return ids
    else:
        return x

def create_full_problem_to_methods(problems_ids, method_ids):
    '''
    Description: Associate all given problems to all given methods.

    Args:
        problem_ids (list): list of problem names
        method_ids (list): list of method names
    Returns:
        full_problem_to_methods (dict): association problem -> method
    '''
    full_problem_to_methods = {}

    for problem_id in problems_ids:
        full_problem_to_methods[problem_id] = []
        for method_id in method_ids:
            full_problem_to_methods[problem_id].append(method_id)

    return full_problem_to_methods

def tune_lr(method_id, method_params, problem_id, problem_params):
        #print("Learning Rate Tuning not yet available!")
        #return method_params
        loss = lambda a, b: np.sum((a-b)**2)
        optimizer = method_params['optimizer']
        search_space = {'optimizer':[]} # parameters for ARMA method
        lr_start, lr_stop = -1, -3 # search learning rates from 10^start to 10^stop 
        learning_rates = np.logspace(lr_start, lr_stop, 1+2*np.abs(lr_start - lr_stop))
        for lr in learning_rates:
            search_space['optimizer'].append(optimizer(learning_rate=lr)) # create instance and append
        trials, min_steps = None, 100
        hpo = GridSearch() # hyperparameter optimizer
        optimal_params, optimal_loss = hpo.search(method_id, method_params, problem_id, problem_params, loss, 
            search_space, trials=trials, smoothing=10, min_steps=min_steps, verbose = 0) # run each model at least 1000 steps
        return optimal_params

#@profile
def run_experiment(problem, method, metric = 'mse', lr_tuning = True, key = 0, timesteps = None, verbose = 0):
    '''
    Description: Initializes the experiment instance.
    
    Args:
        problem (tuple): problem id and parameters to initialize the specific problem instance with
        method (tuple): method id and parameters to initialize the specific method instance with
        metric (string): metric we are interesting in computing for current experiment
        key (int): for reproducibility
        timesteps(int): number of time steps to run experiment for
    Returns:
        loss (list): loss series for the specified metric over the entirety of the experiment
        time (float): time elapsed
        memory (float): memory used
    '''
    set_key(key)

    # extract specifications
    (problem_id, problem_params) = problem
    (method_id, method_params) = method
    
    loss_fn = metrics[metric]
              
        
    # initialize problem
    problem = tigerforecast.problem(problem_id)
   
    if(problem_params is None):
        init = problem.initialize()
    else:
        init = problem.initialize(**problem_params)
    


    # get first few x and y
    if(problem.has_regressors):
        x, y = init
    else:
        x, y = init, problem.step()
        

    # initialize method
    method = tigerforecast.method(method_id)

    if(method_params is None):
        method_params = {}
    try:
        method_params['n'] = x.shape[0]
    except:
        method_params['n'] = 1
    try:
        method_params['m'] = y.shape[0]
    except:
        method_params['m'] = 1

    if(lr_tuning):
        method_params = tune_lr(method_id, method_params, problem_id, problem_params)

    method.initialize(**method_params)
    if(timesteps is None):
        if(problem.max_T == -1):
            print("WARNING: On simulated problem, the number of timesteps should be specified. Will default to 5000.")
            timesteps = 5000
        else:
            timesteps = problem.max_T -method.p- 2
    elif(problem.max_T != -1):
        timesteps = min(timesteps, problem.max_T -method.p- 2)

    for i in range(method.p):
        method.predict(x)
        new = problem.step()
        #print('x:{0}, y:{1}'.format(x,y))
        if(problem.has_regressors):
            x, y = new
        else:
            x, y = y, new
    
    #print('history:{0}'.format(method.past))
    if(verbose and key == 0):
        print("Running %s on %s..." % (method_id, problem_id))

    loss = np.zeros(timesteps)
    time_start = time.time()
    memory = 0
    load_bar = False
    if(verbose == 2):
        load_bar = True
    # get loss series
    for i in tqdm(range(timesteps), disable = (not load_bar or key != 0)):
        # get loss and update method
        try:    #this avoids exceptions usually caused by ONS
            cur_loss = float(loss_fn( method.predict(x),y))
            loss = jax.ops.index_update(loss, i, cur_loss)
            method.update(y)
            # get new pair of observation and label
            new = problem.step()
            if(problem.has_regressors):
                x, y = new
            else:
                x, y = y, new
        except:
            traceback.print_exc()
            loss = jax.ops.index_update(loss, i, float('nan'))

    return loss, time.time() - time_start, memory

def run_experiments(problem, method, metric = 'mse', lr_tuning = True, n_runs = 1, timesteps = None, verbose = 0):
    
    '''
    Description: Initializes the experiment instance.
    
    Args:
        problem (tuple): problem id and parameters to initialize the specific problem instance with
        method (tuple): method id and parameters to initialize the specific method instance with
        metric (string): metric we are interesting in computing for current experiment
        key (int): for reproducibility
        timesteps(int): number of time steps to run experiment for
    Returns:
        loss (list): loss series for the specified metric over the entirety of the experiment
        time (float): time elapsed
        memory (float): memory used
    '''
    
#    return run_experiment(problem, method, metric = metric, \
#        lr_tuning = lr_tuning, key = 0, timesteps = timesteps, verbose = verbose)

    """results = Parallel(n_jobs=n_runs)(delayed(run_experiment)(problem, method, metric = metric, \
        lr_tuning = lr_tuning, key = i, timesteps = timesteps, verbose = verbose) for i in range(0, n_runs))

    resultSum = list(results[0])

    for i in range(1, len(results)):
        for j in range(len(resultSum)):
            resultSum[j] += results[i][j]

    for i in range(len(resultSum)):
        resultSum[i] *= 1.0 / n_runs

    return resultSum"""


    results = tuple((1 / n_runs) * result for result in run_experiment(problem, method, metric = metric, \
        lr_tuning = lr_tuning, key = 0, timesteps = timesteps, verbose = verbose))

    for i in range(1, n_runs):
        new_results = tuple((1 / n_runs) * result for result in run_experiment(problem, method, metric = metric, \
        lr_tuning = lr_tuning, key = i, timesteps = timesteps, verbose = verbose))
        results = tuple(map(operator.add, results, new_results))

    return results
    
