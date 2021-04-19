'''
Newton Step optimizer
'''

from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
from jax import jit, grad
import jax.numpy as np

# regular numpy is necessary for cvxopt to work
import numpy as onp
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


class RealONS(Optimizer):
    """
    Online newton step algorithm.
    """

    def __init__(self, pred=None, loss=mse, hyperparameters={}):
        self.initialized = False
        self.hps = {'G':1, 'c':1,'D':1,'exp_con':0.5 }
        self.hps.update(hyperparameters)
        for key, value in self.hps.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.A, self.Ainv = None, None
        self.pred, self.loss = pred, loss
        self.numpyify = lambda m: onp.array(m).astype(onp.double) # maps jax.numpy to regular numpy
        if not hasattr(self, 'eta'):
            if 4*self.G*self.D>self.exp_con:
                self.eta =2*self.G*self.D
            else:
                self.eta= 0.5*self.exp_con
        if not hasattr(self, 'eps'):
            self.eps= 1.0/((self.eta**2)*(self.D**2))
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

        @jit # partial update step for every matrix in method weights list
        def partial_update(A, Ainv, grad, w):
            A = A + np.outer(grad, grad)
            inv_grad = Ainv @ grad
            Ainv = Ainv - np.outer(inv_grad, inv_grad) / (1 + grad.T @ Ainv @ grad)
            new_grad = np.reshape(Ainv @ grad, w.shape)
            return A, Ainv, new_grad
        self.partial_update = partial_update

    def reset(self):
        self.A, self.Ainv = None, None

    def norm_project(self, y, A, c):
        """ Project y using norm A on the convex set bounded by c. """
        if np.any(np.isnan(y)) or np.all(np.absolute(y) <= c):
            return y
        y_shape = y.shape
        y_reshaped = np.ravel(y)
        dim_y = y_reshaped.shape[0]
        P = matrix(self.numpyify(A))
        q = matrix(self.numpyify(-np.dot(A, y_reshaped)))
        G = matrix(self.numpyify(np.append(np.identity(dim_y), -np.identity(dim_y), axis=0)), tc='d')
        h = matrix(self.numpyify(np.repeat(c, 2 * dim_y)), tc='d')
        solution = np.array(onp.array(solvers.qp(P, q, G, h)['x'])).squeeze().reshape(y_shape)
        return solution

    def general_norm(self, x):
        x = np.asarray(x)
        if np.ndim(x) == 0:
            x = x[None]
        return np.linalg.norm(x)

    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of method pred method
            x (float): input to method
            y (float): true label
            loss (function): loss function. defaults to input value.
        Returns:
            Updated parameters in same shape as input
        """
        assert self.initialized
        assert type(params) == dict, "optimizers can only take params in dictionary format"

        # get args
        grad = self.gradient(params, x, y, loss=self.loss) # defined in optimizers core class

        # used to compute inverse matrix with respect to each parameter vector
        flat_grad = [np.ravel(dw) for dw in grad.values()] # grad is a dict, everything else is a list

        # initialize A
        if self.A is None:
            self.A = [np.eye(dw.shape[0]) * self.eps for dw in flat_grad]
            self.Ainv = [np.eye(dw.shape[0]) * (1 / self.eps) for dw in flat_grad]

            
        # partial_update automatically reshapes flat_grad into correct params shape
        new_values = [self.partial_update(A, Ainv, g, w) for (A, Ainv, g, w) in zip(self.A, self.Ainv, flat_grad, params.values())]
        self.A, self.Ainv, new_grad = list(map(list, zip(*new_values)))

        new_params = {k:w - self.eta * dw for (k, w), dw in zip(params.items(), new_grad)}
        norm = self.c
        new_params = {k:self.norm_project(p, A, norm) for (k,p), A in zip(new_params.items(), self.A)}

        return new_params

    def __str__(self):
        return "<ONS Optimizer, lr={}>".format(self.lr)
