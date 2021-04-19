'''
OGD optimizer
'''
import jax.numpy as np
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error

class RealOGD(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, hyperparameters={}):
        self.initialized = False
        self.hps = {"lr":1,"c":1}
        self.hps.update(hyperparameters)
        for key, value in self.hps.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)
        print("lr",self.lr)

    def norm_project(self, y, c):
        """ Project y using norm A on the convex set bounded by c. """
        y_norm= np.abs(y)
        solution = np.where(y_norm> c, y/y_norm*c, y)
        return solution

    def reset(self): # reset internal parameters
       pass

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

        grad = self.gradient(params, x, y, loss=loss)['phi'] # defined in optimizers core class
        w = params['phi']
        #print("OGD grad")
        #print(grad)

        lr=self.lr
        w_new = w-lr*grad
        phi=self.norm_project(w_new, self.c) 
        new_params = {'phi': phi}
        return new_params


    def __str__(self):
        return "<OGD Optimizer, lr={}>".format(self.lr)
