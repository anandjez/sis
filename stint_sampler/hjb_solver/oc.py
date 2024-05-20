from jax import grad,jit,vmap,random
import jax.numpy as jnp
import jax
import optax
import numpy as np
from hydra.utils import instantiate, call
from omegaconf import DictConfig,OmegaConf
from copy import deepcopy
from pathlib import Path
import wandb
import logging
import time
from stint_sampler.stint.interpolants import linearInterpolant
from stint_sampler.stint.backwardPDE import backwardPDE
from collections import namedtuple
from stint_sampler.eval.plotter import plotter
import matplotlib.pyplot as plt
from typing import Callable


class oc():
    cfg:DictConfig
    pde:backwardPDE
    model:Callable
    intrplnt:linearInterpolant

    def __init__(self,cfg:DictConfig, pde:backwardPDE, model:Callable,intrplnt:linearInterpolant):
        # super().__init__(cfg=cfg)
        self.cfg = cfg
        self.T = self.cfg.get("T", 1.0)
        self.d = self.cfg.get("dim")
        self.bs = self.cfg.get("batch_size")

        self.velocity_model = model
        self.pde = pde
        self.intrplnt = intrplnt

        self.velocity_scale = lambda t: 1#/self.intrplnt.r(t)

        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")

        self.NtTrain = self.cfg.train.get("NtTrain")
        self.scale_init = self.cfg.pde_solver.get("scale_init_train")
        self.train_sde_drift = self.cfg.pde_solver.get("train_sde_drift")
        self.velocityFn = lambda params,t,x:self.velocity_scale(t)*self.velocity_model(params,t,x)

        self.terminalCond = jit(vmap(grad(lambda x: self.pde.phi(x[None, :])[0])))

    def generateTimeSteps(self,t0,t1,N,k):
        tau = 1 - random.uniform(k, (self.bs,N - 1))  # ** tmult
        tvals = tau.sort(axis=-1) * (t1-t0) + t0
        tvals = jnp.concatenate((t0*jnp.ones((self.bs,1)),tvals,t1*jnp.ones((self.bs,1))),axis=-1)
        # tvals = jnp.append(tvals, t1)
        # tvals = jnp.insert(tvals, 0, t0)
        return tvals

    def lossFn(self,params,k):
        Nt = self.NtTrain
        k1, k2 = random.split(k)
        tvals = self.generateTimeSteps(self.eps0,self.T-self.eps1,Nt,k1)
        k1, k2 = random.split(k2)
        X = random.normal(k1,(self.bs,self.d))*jnp.sqrt(self.scale_init)
        L = 0.0
        for i in range(Nt):
            k1, k2 = random.split(k2)
            t = tvals[:,i]
            delT = tvals[:,i+1]-tvals[:,i]
            W = random.normal(k1, X.shape)
            velocity = self.velocityFn(params,t[:,None],X)
            L += delT*jnp.sum(velocity**2,axis=-1)*(self.pde.sigma(t)**2)/2
            # sde_mu = self.train_sde_drift*X
            mu = (self.pde.sigma(t)**2)[:,None]*velocity+self.pde.mu(t,X)
            # mu = (self.pde.sigma(tvals[i])**2)*velocity+self.pde.mu(tvals[i],X)
            X += delT[:,None]*mu + jnp.sqrt(delT[:,None]) * self.pde.sigma(t)[:,None] * W
        L -= self.pde.phi(X)
        Z = self.velocityFn(params, jnp.ones((self.bs, 1)) * self.T, X)
        Zh = self.terminalCond(X)
        L += jnp.mean(jnp.power((Z - Zh), 2))
        metrics = {"x_max": jnp.max(X), "x_min": jnp.min(X)}
        return jnp.mean(L),metrics
