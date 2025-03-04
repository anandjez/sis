from dataclasses import dataclass
from typing import Callable, Any, Tuple
from jax import grad,jit,vmap,random
import jax.numpy as jnp
import jax
from ml_collections import config_dict as configdict
from stint_sampler.stint.interpolants import linearInterpolant
from omegaconf import DictConfig,OmegaConf

class fbsde():

    def __init__(self,cfg:DictConfig, pde, model,intrplnt,solver_params:dict):
        # super().__init__(cfg=cfg)
        self.cfg = cfg
        self.solver_params = solver_params
        self.T = self.cfg.get("T", 1.0)
        self.d = self.cfg.get("dim")
        self.bs = self.cfg.get("batch_size")


        self.velocityPot_model = model
        self.pde = pde
        self.intrplnt = intrplnt

        self.sig0 = lambda t: jnp.sqrt(2 * (
                (self.intrplnt.r(t) ** 2) * self.intrplnt.dg(t) / self.intrplnt.g(t) - self.intrplnt.r(
            t) * self.intrplnt.dr(t)))
        self.beta = self.solver_params["beta"]

        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")

        # self.NtTrain = self.cfg.train.get("NtTrain")
        self.NtTrain = self.solver_params["NtTrain"]
        self.train_sde_drift = self.cfg.pde_solver.get("train_sde_drift")
        self.time_step = self.cfg.pde_solver.get("time_step")*5e-6*(0.01*((self.d>=1000))+1.0*((self.d>=10) and (self.d<1000))+1.0*(self.d<10))#(1.0/(10**int(jnp.log10(self.d))))#
        self.velocity_scale = lambda t: 1.0  # /self.intrplnt.r(t)
        self.generateGrads()

        self.t0 = self.solver_params["t0"]
        self.t1 = self.solver_params["t1"]
        self.add_terminal_loss = self.solver_params["add_terminal_loss"]
        self.compute_process = self.solver_params["compute_process"]
        self.loss_bsde_scale = self.solver_params["loss_bsde_scale"]*(2e3)/(self.time_step*self.NtTrain)
        self.learn_pot = self.solver_params["learn_pot"]
    def generateGrads(self):

        self.velocityPot = lambda params, t, x:jnp.clip( self.velocity_scale(t) * self.velocityPot_model(params, t, x),-100000,100000)

        grad_xNN = grad(lambda params,t,x: self.velocityPot(params,t[:,None],x[None,:])[0,0], 2)

        self.velocityFn1 = jit(vmap(grad_xNN, (None, 0, 0), 0))

        self.velocityFn = lambda params,t,x:jnp.clip(self.velocityFn1(params,t,x),-100000,100000)



        self.terminalCond = jit(vmap(grad(lambda x:self.pde.phi(x[None,:])[0])))

        # def potLap_unbatched(params, t, x):
        #     return jnp.array([jnp.trace(jax.jacfwd(grad_xNN,2)(params,t,x))])
        #
        # grad_tNN = grad(lambda params, t, x: self.velocityPot(params, t[:, None], x[None, :])[0, 0], 1)
        #
        # self.potLap = jit(vmap(potLap_unbatched,(None,0,0),0))
        # self.pot_dt = jit(vmap(grad_tNN, (None, 0, 0), 0))


    def generateTimeSteps(self,t0,t1,N,k):
        if self.solver_params["mode"]:
            tau = 1 - random.uniform(k, (self.bs,N - 1))  # ** tmult
        else:
            N1 = N//2
            tau0 = 1 - random.uniform(k, (self.bs,N1))**2  # ** tmult
            tau1 = random.uniform(k, (self.bs,N- N1 - 1))**2  # ** tmult
            tau = jnp.concatenate((tau0,tau1),axis=-1)

        tvals = tau.sort(axis=-1) * (t1-t0-self.time_step) + t0
        tvals = jnp.concatenate((t0*jnp.ones((self.bs,1)),tvals,t1*jnp.ones((self.bs,1))),axis=-1)
        # tvals = jnp.append(tvals, t1)
        # tvals = jnp.insert(tvals, 0, t0)
        return tvals

    def lossFn_old1(self, params, X, k, t_range, time_step_scale):
        # implementing dX_t = Y_tdt + dW_t, dY_t=Z_tdW_t, Y_T=g(X_T)
        Nt = self.NtTrain
        k1, k2 = random.split(k)
        t0 = jnp.maximum(self.t0, t_range)
        tvals = self.generateTimeSteps(t0, self.t1, Nt, k1)
        # k1, k2 = random.split(k2)
        # X = random.normal(k1,(self.bs,self.d))
        L = 0.0
        loss_term_fbsde = 0.0
        for i in range(0, Nt):
            k1, k2 = random.split(k2)
            l, Xr, metrics_bsde = self.lossBSDE(params, tvals[:, i], X, tvals[:, i + 1] - tvals[:, i], k1,
                                                time_step_scale)
            if self.compute_process:
                t = tvals[:, i]
                # tv = jnp.ones((self.bs, 1)) * t
                b = ((self.intrplnt.dr(t) / self.intrplnt.r(t))[:, None] * X
                     + ((self.sig0(t) ** 2) / (2 * self.beta(t)))[:, None] * self.velocityFn(params, t[:, None],
                                                                                             (1 / self.beta(t))[:,
                                                                                             None] * X))
                X += (tvals[:, i + 1] - tvals[:, i])[:, None] * (b + self.train_sde_drift * X)
                X = jax.lax.stop_gradient(X)
            else:
                X = Xr
            L += self.loss_bsde_scale * l / time_step_scale
        loss_bsde = L
        # if loss_bsde>5:
        #     print("Glitch!")
        if self.add_terminal_loss:
            Z = self.velocityFn(params, jnp.ones((self.bs, 1)) * self.T, X)
            Zh = self.terminalCond(X)
            # Z = jnp.clip(Z,-100.0,100.0)
            loss_term_fbsde = jnp.mean(jnp.power((Z - Zh), 2))
            if self.learn_pot:
                Y = self.velocityPot(params, jnp.ones((self.bs, 1)) * self.T, X)
                Yh = self.pde.phi(X)[:, None]
                loss_term_fbsde += jnp.mean(jnp.power((Y - Yh), 2))
            # if loss_term_fbsde>50:
            #     print("Glitch")
            L += loss_term_fbsde
        metrics = {"loss": L, "loss_bsde": loss_bsde, "loss_term_fbsde": loss_term_fbsde, "x_max": jnp.max(X),
                   "x_min": jnp.min(X)}
        metrics.update(metrics_bsde)
        return L, metrics, X

    def lossFn_old(self,params,X,k,t_range,time_step_scale):
        # implementing dX_t = Y_tdt + dW_t, dY_t=Z_tdW_t, Y_T=g(X_T)
        Nt = self.NtTrain
        k1, k2 = random.split(k)
        t0 = jnp.maximum(self.t0,t_range)
        tvals = self.generateTimeSteps(t0, self.t1, Nt, k1)
        # k1, k2 = random.split(k2)
        # X = random.normal(k1,(self.bs,self.d))
        L = 0.0
        loss_term_fbsde = 0.0
        tc = 0.25
        for i in range(0, Nt):
            k1, k2 = random.split(k2)
            l,Xr,metrics_bsde = self.lossBSDE(params, tvals[:,i], X, tvals[:,i + 1] - tvals[:,i], k1,time_step_scale)
            if self.compute_process:
                t=tvals[:,i]
                # tv = jnp.ones((self.bs, 1)) * t
                b1 = ((self.intrplnt.dr(t)/self.intrplnt.r(t))[:,None] * X
                     + ((self.sig0(t)**2)/(2*self.beta(t)))[:,None] * self.velocityFn(params, t[:,None], (1/self.beta(t))[:,None] * X ))
                s = (1/self.beta(t))[:,None]*self.velocityFn(params, t[:,None], (1/self.beta(t))[:,None] * X )-(1/(self.intrplnt.r(t)**2))[:,None]*X
                b2 = ((self.intrplnt.dg(t)/self.intrplnt.g(t))[:,None] * X
                     + ((self.sig0(t)**2)/(2))[:,None] * s)
                b = 1.0*(t<tc)[:,None]*b1+1.0*(t>=tc)[:,None]*b2
                X += (tvals[:,i + 1] - tvals[:,i])[:,None]*(b+self.train_sde_drift*X)
                X = jax.lax.stop_gradient(X)
            else:
                X = Xr
            L += self.loss_bsde_scale * l/time_step_scale
        loss_bsde = L
        # if loss_bsde>5:
        #     print("Glitch!")
        if self.add_terminal_loss:
            Z = self.velocityFn(params, jnp.ones((self.bs, 1)) * self.T, X)
            Zh = self.terminalCond(X)
            # Z = jnp.clip(Z,-100.0,100.0)
            loss_term_fbsde = jnp.mean(jnp.power((Z - Zh), 2))
            if self.learn_pot:
                Y = self.velocityPot(params, jnp.ones((self.bs, 1)) * self.T, X)
                Yh = self.pde.phi(X)[:, None]
                loss_term_fbsde += jnp.mean(jnp.power((Y - Yh), 2))
            # if loss_term_fbsde>50:
            #     print("Glitch")
            L += loss_term_fbsde
        metrics = {"loss":L,"loss_bsde":loss_bsde,"loss_term_fbsde":loss_term_fbsde,"x_max":jnp.max(X),"x_min":jnp.min(X)}
        metrics.update(metrics_bsde)
        return L,metrics,X

    def lossFn(self,params,X,k,t_range,time_step_scale):
        # implementing dX_t = Y_tdt + dW_t, dY_t=Z_tdW_t, Y_T=g(X_T)
        Nt = self.NtTrain
        k1, k2 = random.split(k)
        t0 = jnp.maximum(self.t0,t_range)
        tvals = self.generateTimeSteps(t0, self.t1, Nt, k1)
        # k1, k2 = random.split(k2)
        # X = random.normal(k1,(self.bs,self.d))
        L = 0.0
        loss_term_fbsde = 0.0
        for i in range(0, Nt):
            k1, k2 = random.split(k2)
            l,Xr,metrics_bsde = self.lossBSDE(params, tvals[:,i], X, tvals[:,i + 1] - tvals[:,i], k1,time_step_scale)
            if self.compute_process:
                t=tvals[:,i]
                # tv = jnp.ones((self.bs, 1)) * t
                b = ((self.intrplnt.dr(t)/self.intrplnt.r(t))[:,None] * X
                     + ((self.sig0(t)**2)/(2*self.beta(t)))[:,None] * self.velocityFn(params, t[:,None], (1/self.beta(t))[:,None] * X ))
                X += (tvals[:,i + 1] - tvals[:,i])[:,None]*(b+self.train_sde_drift*X)
                X = jax.lax.stop_gradient(X)
            else:
                X = Xr
            L += self.loss_bsde_scale * l/time_step_scale
        loss_bsde = L
        # if loss_bsde>5:
        #     print("Glitch!")
        if self.add_terminal_loss:
            Z = self.velocityFn(params, jnp.ones((self.bs, 1)) * self.T, X)
            Zh = self.terminalCond(X)
            # Z = jnp.clip(Z,-100.0,100.0)
            loss_term_fbsde = jnp.mean(jnp.power((Z - Zh), 2))
            if self.learn_pot:
                Y = self.velocityPot(params, jnp.ones((self.bs, 1)) * self.T, X)
                Yh = self.pde.phi(X)[:, None]
                loss_term_fbsde += jnp.mean(jnp.power((Y - Yh), 2))
            # if loss_term_fbsde>50:
            #     print("Glitch")
            L += loss_term_fbsde
        metrics = {"loss":L,"loss_bsde":loss_bsde,"loss_term_fbsde":loss_term_fbsde,"x_max":jnp.max(X),"x_min":jnp.min(X)}
        metrics.update(metrics_bsde)
        return L,metrics,X

    def lossBSDE(self,params,t,X,delt,k,time_step_scale):
        time_step = self.time_step*time_step_scale*t[:,None]

        # tv = jnp.ones((self.bs, 1)) * t

        Y = jnp.clip(self.velocityPot(params, t[:,None], X),-100000,100000)#500 in first submission
        s = self.pde.sigma(t)[:,None]
        Z = jnp.clip(s * self.velocityFn(params, t[:,None], X),-1000,1000)#100 in first submission
        metrics = {"ymax":jnp.max(jnp.abs(Y)),"zmax":jnp.max(jnp.abs(Z))}
        k1, k2 = random.split(k)
        W = random.normal(k1, X.shape)  # *tmask_mat
        # mu = self.pde.mu(t,X)+s*Z/2
        mu = self.pde.mu(t,X)+s*Z
        Xn = X + time_step * mu   + s * jnp.sqrt(time_step) * W   # * tmask_mat
        Y += time_step * self.pde.f(t,X,Z) + jnp.sqrt(time_step) * jnp.sum(Z * W, axis=-1,keepdims=True)  # -self.pde.f(t,X,Z)*time_step +
        Y_nn = jnp.clip(self.velocityPot(params, t[:,None] + time_step, Xn),-100000,100000)#500 in first submission

        L = jnp.mean(jnp.power((Y_nn - Y), 2))#/jnp.sqrt(t[:,None])
        # if jnp.isnan(L):
        #     print("Glitch")
        X += delt[:,None]*mu +jnp.sqrt(delt[:,None]) * s * W    # * tmask_mat
        return L,jax.lax.stop_gradient(X),metrics

    def lossBSDE_old(self,params,t,X,delt,k,time_step_scale):
        time_step = self.time_step*time_step_scale

        # tv = jnp.ones((self.bs, 1)) * t

        Y = jnp.clip(self.velocityPot(params, t[:,None], X),-500,500)
        s = self.pde.sigma(t)[:,None]
        Z = jnp.clip(s * self.velocityFn(params, t[:,None], X),-100,100)
        metrics = {"ymax":jnp.max(jnp.abs(Y)),"zmax":jnp.max(jnp.abs(Z))}
        k1, k2 = random.split(k)
        W = random.normal(k1, X.shape)  # *tmask_mat
        # mu = self.pde.mu(t,X)+s*Z/2
        mu = self.pde.mu(t,X)+s*Z
        Xn = X + mu * time_step + s * W * jnp.sqrt(time_step)  # * tmask_mat
        Y += self.pde.f(t,X,Z)*time_step + jnp.sum(Z * W * jnp.sqrt(time_step), axis=-1,keepdims=True)  # -self.pde.f(t,X,Z)*time_step +
        Y_nn = jnp.clip(self.velocityPot(params, t[:,None] + time_step, Xn),-500,500)

        L = jnp.mean(jnp.power((Y_nn - Y), 2))
        # if jnp.isnan(L):
        #     print("Glitch")
        X += delt[:,None]*mu +jnp.sqrt(delt[:,None]) * s * W    # * tmask_mat
        return L,jax.lax.stop_gradient(X),metrics