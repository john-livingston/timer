"""
PyMC optimization utilities inspired by pymc3-ext approach
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.optimize import minimize
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.util import get_default_varnames
import sys


def optimize(
    start=None,
    vars=None,
    return_info=False,
    verbose=True,
    progress=True,
    maxeval=5000,
    model=None,
    **kwargs
):
    """Maximize the log prob of a PyMC model using scipy
    
    All extra arguments are passed directly to the ``scipy.optimize.minimize``
    function.
    
    Args:
        start: The PyMC coordinate dictionary of the starting position
        vars: The variables to optimize
        model: The PyMC model
        return_info: Return both the coordinate dictionary and the result of
            ``scipy.optimize.minimize``
        verbose: Print the success flag and log probability to the screen
        progress: Show progress during optimization
        maxeval: Maximum number of function evaluations
    """
    wrapper = ModelWrapper(start=start, vars=vars, model=model)
    
    if verbose:
        names = [var.name for var in wrapper.vars]
        sys.stderr.write(
            "optimizing logp for variables: [{0}]\n".format(", ".join(names))
        )
    
    # Count the number of function calls
    neval = 0
    
    # This returns the objective function and its derivatives
    def objective(vec):
        nonlocal neval
        neval += 1
        nll, grad = wrapper(vec)
        if verbose and progress:
            sys.stderr.write(f"\rEval {neval}: logp = {-nll:.3e}")
            sys.stderr.flush()
        
        if neval > maxeval:
            raise StopIteration
            
        return nll, grad
    
    # Optimize using scipy.optimize
    x0 = wrapper.bij.data
    initial = objective(x0)[0]
    kwargs["jac"] = True
    
    try:
        info = minimize(objective, x0, **kwargs)
    except (KeyboardInterrupt, StopIteration):
        info = None
    finally:
        if verbose and progress:
            sys.stderr.write("\n")
    
    # Only accept the output if it is better than it was
    x = info.x if info and np.isfinite(info.fun) and info.fun < initial else x0
    
    # Coerce the output into the right format
    point = get_point(wrapper, x)
    
    if verbose and info is not None:
        sys.stderr.write("message: {0}\n".format(info.message))
        sys.stderr.write("logp: {0} -> {1}\n".format(-initial, -info.fun))
        if not np.isfinite(info.fun):
            sys.stderr.write("WARNING: final logp not finite, returning initial point\n")
            sys.stderr.write("this suggests that something is wrong with the model\n")
    
    if return_info:
        return point, info
    return point


def get_point(wrapper, x):
    """Convert optimized parameters back to PyMC point format"""
    vars = get_default_varnames(wrapper.model.unobserved_value_vars, include_transformed=True)
    values = wrapper.model.compile_fn(vars, mode="FAST_COMPILE")(
        DictToArrayBijection.rmap(RaveledVars(x, wrapper.bij.point_map_info))
    )
    return {var.name: value for var, value in zip(vars, values)}


class ModelWrapper:
    """Wrapper class to interface PyMC model with scipy.optimize"""
    
    def __init__(self, start=None, vars=None, model=None):
        model = self.model = pm.modelcontext(model)
        
        # Work out the full starting coordinates
        if start is None:
            start = model.initial_point()
        self.start = start
        
        # Fit all the parameters by default
        if vars is None:
            vars = model.continuous_value_vars
        self.vars = vars
        
        # Work out the relevant bijection map
        self.bij = DictToArrayBijection.map({
            var.name: start[var.name] for var in vars if var.name in start
        })
        
        # Pre-compile the model logp and gradient
        logp = model.logp()
        grad_logp = pt.grad(logp, vars, disconnected_inputs="ignore")
        self.func = model.compile_fn([logp] + grad_logp, mode="FAST_COMPILE")
    
    def __call__(self, vec):
        """Evaluate negative log probability and its gradient"""
        try:
            # Convert parameter vector back to point format
            point = DictToArrayBijection.rmap(RaveledVars(vec, self.bij.point_map_info))
            
            # Evaluate function
            res = self.func(point)
            
            # Extract logp and gradients
            logp = res[0]
            grads = res[1:]
            
            # Convert gradients back to vector format
            grad_dict = {var.name: grad for var, grad in zip(self.vars, grads)}
            grad_vec = DictToArrayBijection.map(grad_dict)
            
            # Return negative logp (since scipy minimizes) and negative gradient
            return -logp, -grad_vec.data
            
        except Exception as e:
            print("array:", vec)
            print("error:", str(e))
            raise