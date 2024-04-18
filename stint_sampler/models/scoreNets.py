
import jax.numpy as jnp
from jax import grad,jit,vmap
from jax import lax, random
import jax
from typing import Any, Callable, Sequence,Optional,Union
from flax.core import freeze,unfreeze
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
import optax
import json
from flax.training import train_state
from functools import partial


from ml_collections import config_dict as configdict

class FC_DNN(nn.Module):
    features: Sequence[int]
    @nn.compact
    def __call__(self,x):
        # tin = 1/self.L0(t+1e-5)
        # x= jnp.concatenate((t,x),axis=-1)
        for i,ftr in enumerate(self.features):
            x= nn.Dense(ftr)(x)
            if i != len(self.features)-1:
                x= nn.swish(x)
        return x
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  embed_dim: int
  scale: float = 30.
  @nn.compact
  def __call__(self, x):
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                 (self.embed_dim // 2, ))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class DDSnet(nn.Module):
    target_score : Any
    dim: int
    width : int = 64
    @nn.compact
    def __call__(self, t, x):
        t_embed = GaussianFourierProjection(embed_dim=30)(t[:,0])
        t1 = FC_DNN([self.width,self.width])(t_embed)
        extend_out = jnp.concatenate((t,t1,x),axis=-1)
        out1 = FC_DNN([self.width,self.width])(extend_out)
        out2 = nn.Dense(features=self.dim,kernel_init=nn.initializers.zeros_init(),bias_init=nn.initializers.zeros_init())(out1)
        t2 = FC_DNN([self.width,self.width])(t_embed)
        t3 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(),
                        bias_init=nn.initializers.constant(1.0))(t2)
        # log_prob = FC_DNN([1])(self.target(x))
        out = out2+t3*self.target_score(x)
        return out

class DenseNet(nn.Module):
    target_score: Any
    dim: int
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(f) for f in self.features]
        self.outLayer = nn.Dense(self.dim)
        # self.L0 = nn.Dense(1)

    def __call__(self, t,x):
        # tin = 1/self.L0(t+1e-5)
        x= jnp.concatenate((t,x),axis=-1)
        for i,lyr in enumerate(self.layers):
            x= lyr(x)
            # if i != len(self.layers)-1:
            x= nn.swish(x)
        # x = jnp.clip(x,-50,50)
        x = self.outLayer(x)
        return x

class FourierEmb(nn.Module):
    dim_out: int

    @nn.compact
    def __call__(self, t):
        t_embed = GaussianFourierProjection(embed_dim=2*self.dim_out)(t[:, 0])
        out = nn.gelu(nn.Dense(features=self.dim_out)(t_embed))
        out = nn.Dense(features=self.dim_out)(out)
        return out

class DISnet(nn.Module):
    target_score: Any
    dim: int
    width: int = 64
    clip_val: float = 1000.0

    @nn.compact
    def __call__(self, t,x):
        t1 = FourierEmb(dim_out=self.width)(t)
        xt = nn.gelu(t1+nn.Dense(features=self.width)(x))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.Dense(features=self.dim,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.zeros_init())(xt)
        xt = jnp.clip(xt,-self.clip_val,self.clip_val)

        t2 =nn.gelu(FourierEmb(dim_out=self.width)(t))
        t2 = nn.gelu(nn.Dense(features=self.width)(t2))
        t2 = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.constant(1))(t2)
        t2 = jnp.clip(t2, -self.clip_val, self.clip_val)
        return xt+t2*self.target_score(jnp.clip(1/t,1,100)*x)



