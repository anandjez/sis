from dataclasses import dataclass
from typing import Callable, Any, Tuple
from jax import grad,jit,vmap,random
import jax.numpy as jnp
import jax

@dataclass
class backwardPDE():
    mu: Callable
    sigma: Callable
    f: Callable
    phi: Callable[[jax.Array],jax.Array]