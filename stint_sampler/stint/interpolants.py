from dataclasses import dataclass
from typing import Callable, Any, Tuple
from jax import grad,jit,vmap,random
import jax.numpy as jnp
import jax
from ml_collections import config_dict as configdict
# class Interpolant():

# @dataclass
class linearInterpolant():
    # g: Callable[[float],float]
    # r: Callable[[float],float]

    def __init__(self,g: Callable[[float],float],r: Callable[[float],float],T):
        # self.g = self.generateFn(g)
        # self.r = self.generateFn(r)
        # self.T = T
        self.g=lambda t: g(t/T)
        self.r=lambda t: r(t/T)
        self.dg = self.generateFn(grad(self.g))
        self.dr = self.generateFn(grad(self.r))
    def generateFn(self,f_scalar):
        def f(t):
            # if type(t) == int or type(t)==float:
            if hasattr(t,"ndim"):
                axes = t.ndim
                if axes == 1:
                    return vmap(f_scalar)(t)
                elif axes==2:
                    return vmap(f_scalar)(t[:, 0])[:, None]
                else:
                    return f_scalar(t)
            else:
                return f_scalar(t)

        return f






