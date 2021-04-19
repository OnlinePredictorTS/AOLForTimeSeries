'''
SF-FTRL optimizer
'''
import jax.numpy as np
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
import jax

class ADAFtrl(Optimizer):
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
        self.L=None
        


    def reset(self): # reset internal parameters
        self.theta = None
        self.eta = None
        self.G=None
        self.L=None

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
        n=np.shape(grad)[0]
        ntp=np.shape(grad)[1]
        p=int(ntp/n)
        _x=np.reshape(x,(n,p))
        _x_norm=np.linalg.norm(_x,axis=0)
        _y_norm=np.linalg.norm(y)
        if self.G is None:
            self.G=_x_norm
        _grad=np.reshape(grad,(n,n,p))
        _grad_norm=np.linalg.norm(_grad,axis=(0,1))

        _G = np.roll(self.G, 1)
        _G = jax.ops.index_update(_G, 0, _y_norm)
        self.G=np.maximum(self.G,_G)
        
        if self.L is None:
            self.L=np.ones(np.shape(self.G))
        self.L=np.maximum(self.L,np.where(_x_norm>0,_grad_norm/_x_norm,_x_norm))
        
        if self.theta is None:
            self.theta = -_grad
        else:
            self.theta -= _grad
        if self.eta is None:
            self.eta = (_grad_norm**2)
        else:
            self.eta+=(_grad_norm**2)
        lr= np.reshape(np.sqrt(self.eta+(self.L*self.G)**2),(1,1,-1))
        phi=np.reshape(np.where(lr>0, self.theta/lr,self.theta),np.shape(grad))
        new_params = {'phi': phi}
        #x_new = np.roll(x, 1)
        #x_new = jax.ops.index_update(x_new, 0, y)
        #y_t = self.pred(params=new_params, x=x_new)
        
        #new_mapped_params = {k:self.norm_project(np.sqrt(eta+(self.G*L)**2),x_new, y_t,p) for (k,p), eta in  zip(new_params.items(),self.eta.values())}
        
        #y_t = self.pred(params=new_mapped_params, x=x_new)

        return new_params


    def __str__(self):
        return "<SFftrl Optimizer, lr={}>".format(self.lr)


