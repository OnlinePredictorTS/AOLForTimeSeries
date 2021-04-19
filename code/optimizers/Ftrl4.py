'''
SF-FTRL optimizer
'''
import jax.numpy as np
import jax.numpy.linalg as alg
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
import jax

class Ftrl4(Optimizer):
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
        self.a = 0
        self.b = None
        self.G=None
        self.L=1
        


    def reset(self): # reset internal parameters
        self.theta = None
        self.a = 0
        self.b = None
        self.G = None
        self.L=1

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

        #L largest time series norm
        self.L=np.maximum(self.L, alg.norm(y))
        
        grad = self.gradient(params, x, y, loss=loss)['phi'] # defined in optimizers core class
        n=np.shape(grad)[0]
        
        if self.theta is None:
            self.theta = -grad
        else:
            self.theta -= grad
        #initialise b
        if self.b is None:
            x_norm =alg.norm(x)
            self.b=x_norm**4

        #update next feature vector
        x_new = np.roll(x, n)
        x_new = jax.ops.index_update(x_new, jax.ops.index[0:n], y)
        x_norm =alg.norm(x_new)

        #update regulariser for w_tx
        self.b+= x_norm**4

        #update regulariser for -y_t
        _x=np.reshape(x,(-1,1))
        _y=np.reshape(y,(-1,1))
        yx= -_y@_x.T
        g_norm =alg.norm(yx)
        self.a+= g_norm**2
        
        #normalise theta
        theta_norm =alg.norm(self.theta)
        w=self.theta/theta_norm

        b=np.sqrt(self.b)
        a=np.sqrt(self.a+(x_norm*self.L)**2)

        p=a/b
        q=-theta_norm/b
        lr=np.cbrt(-q/2+np.sqrt(q**2/4+p**3/27))+np.cbrt(-q/2-np.sqrt(q**2/4+p**3/27))

        phi=w*lr
        new_params = {'phi': phi}
        return new_params

    def __str__(self):
        return "<SFftrl Optimizer, lr={}>".format(self.lr)



