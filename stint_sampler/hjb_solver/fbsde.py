from dataclasses import dataclass
from typing import Callable, Any, Tuple
from jax import grad,jit,vmap,random
import jax.numpy as jnp
import jax
from ml_collections import config_dict as configdict
from stint_sampler.stint.interpolants import linearInterpolant
from omegaconf import DictConfig,OmegaConf

class fbsde():

    def __init__(self,cfg:DictConfig, pde, model,intrplnt):
        # super().__init__(cfg=cfg)
        self.cfg = cfg
        self.T = self.cfg.get("T", 1.0)
        self.d = self.cfg.get("dim")
        self.bs = self.cfg.get("batch_size")

        self.velocityPot_model = model
        self.pde = pde
        self.intrplnt = intrplnt

        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")

        self.NtTrain = self.cfg.train.get("NtTrain")

        self.generateGrads()

    def generateGrads(self):
        grad_xNN = grad(lambda params,t,x: self.velocityPot_model(params,t[:,None],x[None,:])[0,0], 2)

        self.velocityFn = jit(vmap(grad_xNN, (None, 0, 0), 0))

        self.velocityPot = self.velocityPot_model

        self.terminalCond = jit(vmap(grad(lambda x:self.pde.phi(x[None,:])[0])))


    def generateTimeSteps(self,t0,t1,N,k):
        tau = 1 - random.uniform(k, (N - 1,))  # ** tmult
        tvals = tau.sort() * (t1-t0) + t0
        tvals = jnp.append(tvals, t1)
        tvals = jnp.insert(tvals, 0, t0)
        return tvals

    def lossFn(self,params,k):
        # implementing dX_t = Y_tdt + dW_t, dY_t=Z_tdW_t, Y_T=g(X_T)
        Nt = self.NtTrain
        k1, k2 = random.split(k)
        tvals = self.generateTimeSteps(self.eps0, self.T - self.eps1, Nt, k1)
        k1, k2 = random.split(k2)
        X = random.normal(k1,(self.bs,self.d))
        L = 0.0
        for i in range(0, Nt):
            k1, k2 = random.split(k2)
            l,_ = self.lossBSDE(params, tvals[i], X, tvals[i + 1] - tvals[i], k1)
            t=tvals[i]
            tv = jnp.ones((self.bs, 1)) * t
            b = ((self.intrplnt.dr(t)/self.intrplnt.r(t)) * X
                 + (self.intrplnt.dg(t) * self.intrplnt.r(t) - self.intrplnt.dr(t) * self.intrplnt.g(t)) * self.velocityFn(params, tv, self.intrplnt.g(t) * X / self.intrplnt.r(t)))
            X += b * (tvals[i + 1] - tvals[i])
            X = jax.lax.stop_gradient(X)
            L += 1e8 * l

        Z = self.velocityFn(params, jnp.ones((self.bs, 1)) * self.T, X)
        Zh = self.terminalCond(X)
        L += jnp.mean(jnp.power((Z - Zh), 2))
        metrics = {"x_max":jnp.max(X),"x_min":jnp.min(X)}
        return L,metrics

    def lossBSDE(self,params,t,X,delt,k):
        time_step = 1e-6

        tv = jnp.ones((self.bs, 1)) * t

        Y = self.velocityPot(params, tv, X)
        s = self.pde.sigma(t)
        Z = s * self.velocityFn(params, tv, X)

        k1, k2 = random.split(k)
        W = random.normal(k1, X.shape)  # *tmask_mat
        mu = self.pde.mu(t,X)+self.pde.sigma(t)*Z/2
        Xn = X + mu * time_step + s * W * jnp.sqrt(time_step)  # * tmask_mat
        Y +=  jnp.sum(Z * W * jnp.sqrt(time_step), axis=-1,keepdims=True)  # -self.pde.f(t,X,Z)*time_step +
        Y_nn = self.velocityPot(params, tv + time_step, Xn)

        L = jnp.mean(jnp.power((Y_nn - Y), 2))
        X += self.pde.mu(t,X) * delt + s * W * jnp.sqrt(delt)  # * tmask_mat
        return L,jax.lax.stop_gradient(X)

