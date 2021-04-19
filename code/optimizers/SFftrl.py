'''
SF-FTRL optimizer
'''
import jax.numpy as np
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
import jax

class SFftrl(Optimizer):
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
        self.hyperparameters = {}
        self.hyperparameters.update(hyperparameters)
        for key, value in self.hyperparameters.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)
        self.theta = None
        self.eta = None
        self.G=None
        


    def reset(self): # reset internal parameters
        self.theta = None
        self.eta = None

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
        
        if hasattr(self, 'L'):
            L=self.L
        else:
            L=1.0
        grad = self.gradient(params, x, y, loss=loss)['phi'] # defined in optimizers core class
        d_sqrt=np.sqrt(np.shape(grad)[0])
        if self.G is None:
            self.G=np.max(np.abs(x))
            self.G=np.maximum(self.G, np.max(np.abs(y)))
        else:
            self.G=np.maximum(self.G, np.max(np.abs(y)))
        if self.theta is None:
            self.theta = -grad
        else:
            self.theta -= grad
        theta_abs = np.abs(self.theta)
        if self.eta is None:
            self.eta = (grad**2)
        else:
            self.eta+=(grad**2)
        #lr=np.sqrt(np.maximum(self.eta,theta_abs))
        lr=np.sqrt(self.eta)
        phi=np.where(lr>0, self.theta/lr,self.theta)
        new_params = {'phi': phi}
        return new_params


    def __str__(self):
        return "<SFftrl Optimizer, lr={}>".format(self.lr)



