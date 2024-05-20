
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
    width : int = 64
    @nn.compact
    def __call__(self, t, x):
        t_embed = GaussianFourierProjection(embed_dim=30)(t[:,0])
        t1 = FC_DNN([self.width,self.width])(t_embed)
        extend_out = jnp.concatenate((t,t1,x),axis=-1)
        out1 = FC_DNN([self.width,self.width,1])(extend_out)
        t2 = FC_DNN([self.width,self.width,1])(t_embed)
        # log_prob = FC_DNN([1])(self.target(x))
        out = out1+t2*self.target_score(x)[:,None]
        return out

class DDSnet2(nn.Module):
    width : int = 64
    @nn.compact
    def __call__(self, t, x):
        t_embed = GaussianFourierProjection(embed_dim=30)(t)[0,:]
        t1 = FC_DNN([self.width,self.width])(t_embed)
        extend_out = jnp.concatenate((t1,x),axis=-1)
        out1 = FC_DNN([self.width,self.width])(extend_out)
        t2 = FC_DNN([self.width,self.width])(t_embed)
        out = FC_DNN([self.width,1])(out1+t2)
        # log_prob = FC_DNN([1])(self.target(x))
        # out = out1+t2*self.target(x)
        return out[0]

class DDSnet_interpol(nn.Module):
    target : Any
    width : int = 64
    @nn.compact
    def __call__(self, t, x):
        t_embed = GaussianFourierProjection(embed_dim=30)(t)[0,:]
        t1 = FC_DNN([self.width,self.width])(t_embed)
        extend_out = jnp.concatenate((t,t1,x),axis=-1)
        out1 = FC_DNN([self.width,self.width,self.width,1])(extend_out)
        t2 = FC_DNN([self.width,self.width,self.width,1])(t_embed)
        # log_prob = FC_DNN([1])(self.target(x))
        out = out1+t2*(t[0]*self.target(x)+(1-t[0])*(-jnp.sum(x**2)/2))
        return out[0]



class DenseNet2(nn.Module):
    features: Sequence[int]
    target: Any
    def setup(self):
        self.layers = [nn.Dense(f) for f in self.features]
        self.embed = GaussianFourierProjection(embed_dim=10)
        self.FCDNN_t1 = FC_DNN([10,4])
        self.FCDNN_t2 = FC_DNN([64,64,1])

    def __call__(self, t, x):
        # tin = 1/self.L0(t+1e-5)
        t_embed = self.embed(t)[0, :]
        t1 = self.FCDNN_t1(t_embed)
        x2 = self.FCDNN_t2(t_embed)*self.target(x)*t
        for i, lyr in enumerate(self.layers):
            x = jnp.concatenate((t, t1, x), axis=-1)
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.swish(x)
        # x = jnp.clip(x,-50,50)
        x += x2
        return x[0]

class DenseNet0(nn.Module):

    target_score: Any
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(f) for f in self.features]
        self.embed = GaussianFourierProjection(embed_dim=10)
        self.FCDNN_t1 = FC_DNN([5,1])
        self.FCDNN_t2 = FC_DNN([64,64,1])

    def __call__(self, t, x):
        # tin = 1/self.L0(t+1e-5)
        t_embed = self.embed(t[:,0])
        t1 = self.FCDNN_t1(t_embed)
        # x2 = self.FCDNN_t2(t_embed)*self.target(x)*t
        # x = jnp.concatenate((t, t1, x), axis=-1)
        for i, lyr in enumerate(self.layers):
            x = jnp.concatenate((t, t1, x), axis=-1)
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.swish(x)
        # x = jnp.clip(x,-50,50)
        # x += x2
        return x

class FourierEmb(nn.Module):
    dim_out: int

    @nn.compact
    def __call__(self, t):
        t_embed = GaussianFourierProjection(embed_dim=2*self.dim_out)(t[:, 0])
        out = nn.gelu(nn.Dense(features=self.dim_out)(t_embed))
        out = nn.Dense(features=self.dim_out)(out)
        return out




class DISnet_init(nn.Module):
    target_score: Any
    width: int = 64
    clip_val: float = 100.0

    @nn.compact
    def __call__(self, t,x):
        t_embed = FourierEmb(dim_out=self.width)(t)
        xt = nn.gelu(t_embed+nn.Dense(features=self.width)(x))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.zeros_init())(xt)
        # xt = jnp.clip(xt,-self.clip_val,self.clip_val)

        # t2 =nn.gelu(FourierEmb(dim_out=self.width)(t))
        t2 = nn.gelu(nn.Dense(features=self.width)(t_embed))
        t2 = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.constant(1))(t2)

        t3 = nn.gelu(nn.Dense(features=self.width)(t_embed))
        t3 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.constant(1))(t3)

        # t3 =nn.gelu(FourierEmb(dim_out=self.width)(t))
        # t3 = nn.gelu(nn.Dense(features=self.width)(t3))
        # t3 = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.constant(1))(t3)
        # t3 = jnp.clip(t3, 1, 10)
        targ_init = jnp.sum(jnp.clip((1 / t) * x/2, -10, 10)*x,axis=-1,keepdims=True)
        targ = t2 * (1 - t) * targ_init + t3 * t * self.target_score(x)[:,None]
        # xscale = jnp.clip(1/t,1,10)
        # targ = self.target_score(xscale*x)[:,None]/xscale
        # targ = jnp.clip(targ, -self.clip_val, self.clip_val)
        # targ = (1-t)*jnp.sum(x**2,axis=-1,keepdims=True)/2+t*self.target_score(x)[:,None]
        # targ = self.target_score(x)[:,None]
        return xt+targ
        # return xt+t*t2*self.target_score(x)[:,None]

class DenseNet_init(nn.Module):
    target_score: Any
    features: Sequence[int]

    @nn.compact
    def __call__(self, t,x):
        # tin = 1/self.L0(t+1e-5)
        t_embed = FourierEmb(dim_out=8)(t)
        t2 = nn.gelu(nn.Dense(features=4)(t_embed))
        t2 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.constant(1))(t2)

        t3 = nn.gelu(nn.Dense(features=4)(t_embed))
        t3 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.constant(1))(t3)

        x1 = jnp.array(x)
        x1 = jnp.concatenate((t, x1), axis=-1)
        for i,lyr in enumerate(self.features):
            # if i%3 ==0:

            # x1 = jnp.concatenate((t, x1), axis=-1)
            x1= nn.Dense(features=lyr)(x1)
            # if i != len(self.features)-1:
            x1= nn.swish(x1)
        x1 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.constant(0))(x1)
        targ_init = jnp.sum(jnp.clip((1 / t) * x / 2, -20, 20) * x, axis=-1, keepdims=True)
        targ = t2 * (1 - t) * targ_init + t3 * t * self.target_score(x)[:,None]
        # x = jnp.clip(x,-50,50)
        return x1+targ


class DenseNet(nn.Module):

    target_score: Any
    features: Sequence[int]

    @nn.compact
    def __call__(self, t,x):
        # tin = 1/self.L0(t+1e-5)
        x = jnp.concatenate((t, x), axis=-1)
        for i,lyr in enumerate(self.features):
            # if i%3 ==0:

            x= nn.Dense(features=lyr)(x)
            # if i != len(self.features)-1:
            x= nn.swish(x)
        x = nn.Dense(features=1)(x)
        # x = jnp.clip(x,-50,50)
        return x

class DenseNet_targ(nn.Module):
    target_score: Any
    features: Sequence[int]

    @nn.compact
    def __call__(self, t,x):
        # tin = 1/self.L0(t+1e-5)
        t_embed = FourierEmb(dim_out=8)(t)
        t2 = nn.gelu(nn.Dense(features=4)(t_embed))
        t2 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.constant(1))(t2)

        # t3 = nn.gelu(nn.Dense(features=4)(t_embed))
        # t3 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.constant(1))(t3)

        x1 = jnp.array(x)
        x1 = jnp.concatenate((t, x1), axis=-1)
        for i,lyr in enumerate(self.features):
            # if i%3 ==0:

            # x1 = jnp.concatenate((t, x1), axis=-1)
            x1= nn.Dense(features=lyr)(x1)
            # if i != len(self.features)-1:
            x1= nn.swish(x1)
        x1 = nn.Dense(features=1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.constant(0))(x1)
        # targ_init = jnp.sum(jnp.clip((1 / t) * x / 2, -20, 20) * x, axis=-1, keepdims=True)
        targ = t2 * self.target_score(x)[:,None]
        # x = jnp.clip(x,-50,50)
        return x1+targ

class DISnet(nn.Module):
    target_score: Any
    width: int = 64
    clip_val: float = 100.0

    @nn.compact
    def __call__(self, t,x):
        t1 = FourierEmb(dim_out=self.width)(t)
        xt = nn.gelu(t1+nn.Dense(features=self.width)(x))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.zeros_init())(xt)
        # xt = jnp.clip(xt,-self.clip_val,self.clip_val)

        t2 =nn.gelu(FourierEmb(dim_out=self.width)(t))
        t2 = nn.gelu(nn.Dense(features=self.width)(t2))
        t2 = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.constant(1))(t2)

        targ = self.target_score(x)[:,None]
        return xt+t2*targ

class DISnet_invt(nn.Module):
    target_score: Any
    width: int = 64
    clip_val: float = 1000.0


    @nn.compact
    def __call__(self, t,x):
        tfor_embed = FourierEmb(dim_out=self.width)(t)
        tinv_embed = jnp.clip(1/t,-self.clip_val,self.clip_val)
        t_embed = jnp.concatenate((tfor_embed,tinv_embed*tfor_embed),axis=-1)
        xt = nn.gelu(nn.Dense(features=self.width)(t_embed)+nn.Dense(features=self.width)(x))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.zeros_init())(xt)
        # xt = jnp.clip(xt,-self.clip_val,self.clip_val)

        # t2 =nn.gelu(FourierEmb(dim_out=self.width)(t))
        t2 = nn.gelu(nn.Dense(features=self.width)(t_embed))
        t2 = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.constant(1))(t2)

        targ = self.target_score(x)[:, None]
        return xt + t2 * targ


class DISnet0(nn.Module):
    target_score: Any
    width: int = 64
    clip_val: float = 100.0

    @nn.compact
    def __call__(self, t,x):
        t1 = FourierEmb(dim_out=self.width//2)(t)
        xt = nn.gelu(t1+nn.Dense(features=self.width)(x))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.gelu(nn.Dense(features=self.width)(xt))
        xt = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.zeros_init())(xt)
        # xt = jnp.clip(xt,-self.clip_val,self.clip_val)

        t2 =nn.gelu(FourierEmb(dim_out=self.width)(t))
        t2 = nn.gelu(nn.Dense(features=self.width)(t2))
        t2 = nn.Dense(features=1,kernel_init=nn.initializers.zeros_init(),bias_init = nn.initializers.constant(1))(t2)

        targ = self.target_score(x)[:,None]
        return xt+t2*targ