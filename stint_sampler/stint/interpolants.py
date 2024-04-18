from dataclasses import dataclass
from typing import Callable, Any, Tuple
from jax import grad,jit,vmap,random
import jax.numpy as jnp
import jax
from ml_collections import config_dict as configdict
# class Interpolant():

@dataclass
class linearInterpolant():
    g: Callable[[float],float]
    r: Callable[[float],float]

    def __post_init__(self):
        self.dg = grad(self.g)
        self.dr = grad(self.r)


