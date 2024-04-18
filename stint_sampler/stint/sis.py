from dataclasses import dataclass
from typing import Callable, Any, Tuple
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
from stint_sampler.hjb_solver.oc import oc
from stint_sampler.hjb_solver.fbsde import fbsde
import matplotlib.pyplot as plt

Results = namedtuple(
    "Results",
    "samples weights log_norm_const_preds expectation_preds ts xs metrics plots",
    defaults=[{}, {}, None, None, None, None, {}, {}],)



class sampler():
    def __init__(self,cfg:DictConfig):
        self.cfg = deepcopy(cfg)
        OmegaConf.resolve(self.cfg)
        self.out_dir = Path(cfg.out_dir)

        if "seed" in self.cfg:
            self.k = random.PRNGKey(self.cfg.seed)

        # Logging and checkpoints
        self.plot_results: bool = self.cfg.get("plot_results", True)
        self.store_last_ckpt: bool = self.cfg.get("store_last_ckpt", False)
        self.restore_ckpt_from_wandb: bool | None = self.cfg.get(
            "restore_ckpt_from_wandb"
        )
        self.upload_ckpt_to_wandb: str | bool | None = self.cfg.get(
            "upload_ckpt_to_wandb"
        )
        if (
                isinstance(self.upload_ckpt_to_wandb, str)
                and self.upload_ckpt_to_wandb != "last"
        ):
            raise ValueError("Unknown upload mode.")
        self.eval_marginal_dims: list = self.cfg.get("eval_marginal_dims", [])

        self.device = self.cfg.get("device")

        # Paths
        self.ckpt_file: str | None = self.cfg.get("ckpt_file")
        self.ckpt_dir = self.out_dir / "ckpt"
        logging.info("Checkpoint directory: %s", self.ckpt_dir)
        # See https://jsonlines.org/ for the JSON Lines text file format
        self.metrics_file = self.out_dir / "metrics.jsonl"

        # Weights & Biases
        self.wandb = wandb.run
        if self.wandb is None:
            wandb.init(mode="disabled")
        else:
            self.wandb.summary["device"] = str(self.device)

        self.initialized = False
        self.initial_time = time.time()

        self.loss_grad = None

        self.log_interval = self.cfg.get("log_interval")
        self.eval_interval = self.cfg.get("eval_interval")
        self.jit_lossFn = self.cfg.get("jit_lossFn")

    def train(self):
        runs = self.cfg.train.get("epoch_steps")
        steps = self.cfg.train.get("epochs")
        lr = self.cfg.train.get("learning_rate")
        def opt_step(params, opt_state, tmult,k):
            (L,metrics), grad_par = self.loss_grad(params,k)
            updates, opt_state = optimizer.update(grad_par, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, L,metrics

        k1, k2 = random.split(random.PRNGKey(np.random.randint(0, 1000)))
        params_ema = self.params.copy()

        for run in range(runs):

            if run:
                optimizer = optax.adam(learning_rate=lr / (10 ** run))
            else:
                optimizer = optax.adam(learning_rate=lr)
                opt_state = optimizer.init(self.params)

            for step in range(steps):
                k1, k2 = random.split(k2)
                self.params, opt_state, L,metrics = opt_step(self.params, opt_state, runs - run,k1)
                params_ema = optax.incremental_update(self.params, params_ema, 0.01)
                # step_log.append(steps)
                if step % self.log_interval == 0:
                    metrics["loss"]=L
                    print(metrics)
                    wandb.log(metrics)
        self.params=params_ema
        return params_ema

    def eval(self):
        X = self.generateSamples()
        plotObj = plotter(X)
        hist_dims = self.cfg.eval.get("hist_dims")
        hist2d = plotObj.make_hist_plot2d(hist_dims)
        hist2d.savefig(Path(self.out_dir / "hist2d.png"), bbox_inches='tight', pad_inches=0.1, dpi=300)

        self.debugPlots()

    def debugPlots(self):
        name = 'marginal_velocity.png'
        debugDir = self.out_dir / "deb"/ name
        debugDir.parent.mkdir(parents=True, exist_ok=True)
        N=256
        xrange = 10
        xd1 = np.random.uniform(-1, 1, (self.d - 1,)).tolist()
        X = jnp.array([[-xrange + 2*xrange * i / N] + xd1 for i in range(N)])
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.tight_layout()
        for t in np.linspace(self.eps0, self.T, 5):
            tv = (t) * np.ones([N, 1])
            Z = self.velocityFn(self.params, tv, X)
            ax.plot(X[:, 0], Z[:, 0], label=str(t))

        Y = self.terminalCond(X)
        ax.plot(X[:, 0], Y[:, 0], marker='*', label='True 0.5')
        ax.set_ylim(-10,10)
        ax.legend()
        fig.savefig(debugDir, bbox_inches='tight', pad_inches=0.1, dpi=300)

class half_sis(sampler):

    def __init__(self,cfg:DictConfig):
        super().__init__(cfg=cfg)

        self.target,_ = call(self.cfg.target)
        self.target_score = vmap(grad(lambda x:self.target(x[None,:])[0]))
        self.intrplnt = linearInterpolant(*call(self.cfg.interpolant))
        self.beta = lambda t: self.intrplnt.r(t)/self.intrplnt.g(t)
        self.sig0 = lambda t: jnp.sqrt(2 * ((self.intrplnt.r(t) ** 2) * self.intrplnt.dg(t) / self.intrplnt.g(t) - self.intrplnt.r(t) * self.intrplnt.dr(t)))
        self.T = self.cfg.get("T", 1.0)
        self.d = self.cfg.get("dim")
        self.bs = self.cfg.get("batch_size")
        self.pde = backwardPDE(*self.generatePDEcoeffs())
        self.terminalCond = jit(vmap(grad(lambda x: self.pde.phi(x[None, :])[0])))

        solver_name = self.cfg.hjb_solver.get("name")
        if solver_name=="oc":
            self.velocity_model = instantiate(self.cfg.model,target_score=self.terminalCond)
        elif solver_name=="fbsde":
            self.velocity_model = instantiate(self.cfg.model, target_score=self.pde.phi)
        k1, self.k = random.split(self.k)
        xin = random.uniform(k1, (1,self.d))
        tin = random.uniform(k1, (1,1))
        k1, self.k = random.split(self.k)
        self.params = self.velocity_model.init(k1, tin, xin)

        self.hjb_solver = instantiate(self.cfg.hjb_solver.solver,self.cfg,self.pde,self.velocity_model.apply,self.intrplnt)
        # self.hjb_solver = instantiate(self.cfg.hjb_solver,cfg=self.cfg)
        # self.hjb_solver = oc(cfg=self.cfg,pde=self.pde,model=self.velocity_model.apply,intrplnt=self.intrplnt)
        # self.hjb_solver = fbsde(cfg=self.cfg,pde=self.pde,model=self.velocity_model,intrplnt=self.intrplnt)
        self.velocityFn = self.hjb_solver.velocityFn

        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")

        if self.jit_lossFn:
            self.loss_grad = jit(jax.value_and_grad(self.hjb_solver.lossFn,has_aux=True))
        else:
            self.loss_grad = jax.value_and_grad(self.hjb_solver.lossFn,has_aux=True)
        self.NtTrain = self.cfg.train.get("NtTrain")



    def generatePDEcoeffs(self):
        logPsi = lambda t, x: -jnp.sum(x**2)/(2*self.intrplnt.r(t)**2)-self.d*jnp.log(2*jnp.pi*self.intrplnt.r(t)**2)/2
        self.logPsi = vmap(logPsi,(None,0),0)

        lin_drift_fun = lambda t: jnp.log(self.beta(t) * self.intrplnt.g(t) / (self.intrplnt.r(t) ** 2))
        lin_drift_coeff = grad(lin_drift_fun)

        s = lambda t: self.sig0(t) / self.beta(t)

        def mu(t, X):
            return  - lin_drift_coeff(t) * X

        def f(t, X, Grad_u):
            return (s(t)**2) * jnp.sum(Grad_u,axis=-1,keepdims=True) / 2

        phi = lambda x: self.target(self.beta(self.T)*x)-self.logPsi(self.T,self.beta(self.T)*x)
        return mu,s,f,phi

    def generateSamples(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr


        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")
        k1, self.k = random.split(self.k)
        X = random.normal(k1,(Ns,self.d))*jnp.sqrt(1.0)
        delT = (self.T-self.eps1 - self.eps0) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.velocityFn(self.params, tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = 1.0
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))

        return X






