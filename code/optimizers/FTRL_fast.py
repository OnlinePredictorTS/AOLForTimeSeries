'''
FTRL optimizer
'''
import jax.numpy as np
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
import jax

class FTRL_fast(Optimizer):
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
        self.A = None
        self.A_inv = None
        
    def norm_project(self, a,x,y,w):
        
        """ Project y using norm A on the convex set bounded by c. """
        
        if hasattr(self, 'c'):
            if abs(y)>self.c:
                w_p= np.matmul(a,x)
                deno= np.dot(x.T, w_p)
                if y>0:
                    w= w-(abs(y)-self.c)/deno*w_p
                else:
                    w= w+(abs(y)-self.c)/deno*w_p
        return w

    def reset(self): # reset internal parameters
        self.A = None
        self.A_inv = None

    def pinv(self,a,rcond=None):
      # ported from https://github.com/numpy/numpy/blob/v1.17.0/numpy/linalg/linalg.py#L1890-L1979
        a = np.conj(a)
      # copied from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/linalg.py#L442
        if rcond is None:
            max_rows_cols = max(a.shape[-2:])
            rcond = 10. * max_rows_cols * np.finfo(a.dtype).eps
            #rcond = 1e-3
            rcond = np.asarray(rcond)
            u, s, v = np.linalg.svd(a, full_matrices=False)
      # Singular values less than or equal to ``rcond * largest_singular_value``
      # are set to zero.
        cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        s = np.divide(1, s)
        s = np.where(large, s, 0)
        vT = np.swapaxes(v, -1, -2)
        uT = np.swapaxes(u, -1, -2)
        res = np.matmul(vT, np.multiply(s[..., np.newaxis], uT))
        return np.lax.convert_element_type(res, a.dtype)
    
        
    
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
        
        x_plus_bias = np.vstack((np.ones((1, 1)), x))
        if self.A_inv is None:
            self.A= 2.0*np.matmul(x_plus_bias,x_plus_bias.T)
        else:
            self.A= self.A+2.0*np.matmul(x_plus_bias,x_plus_bias.T)
        self.A_inv= self.pinv(a=self.A)
        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class          
        new_params = {k: w-np.matmul(self.A_inv,g) for (k, w),g in zip(params.items(),grad.values())}
        
        x_new = np.roll(x, 1)
        x_new = jax.ops.index_update(x_new, 0, y)
        y_t = self.pred(params=new_params, x=x_new)
        
        x_plus_bias_new = np.vstack((np.ones((1, 1)), x_new))
        new_mapped_params = {k:self.norm_project(self.A_inv,x_plus_bias_new,y_t,p) for (k,p) in new_params.items()}
        
      #  y_t = self.pred(params=new_mapped_params, x=x_new)
       # if np.isnan(y_t):
       #     print(self.A_inv)
        return new_mapped_params


    def __str__(self):
        return "<SFftrl Optimizer, lr={}>".format(self.lr)



