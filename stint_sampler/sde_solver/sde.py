from dataclasses import dataclass
from typing import Callable, Any, Tuple
from jax import grad,jit,vmap,random
import jax.numpy as jnp
import jax
from omegaconf import DictConfig,OmegaConf

@dataclass
class sde():
    cfg:DictConfig
    mu: Callable
    sigma: Callable

    def __post_init__(self):
        self.T = self.cfg.get("T")
        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")

    def integrateEM(self,params,X0,N,k):
        delT = (self.T-self.eps0-self.eps1)/N
        k1,k2 = random.split(k)
        X = X0
        for i in range(N):
            W = random.normal(k1,X0.shape)
            t = i*delT+self.eps0
            X += self.mu(params,t,X)*delT+self.sigma(t)*W*jnp.sqrt(delT)
        return X
