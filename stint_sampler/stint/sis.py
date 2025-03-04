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
from stint_sampler.common.utils import generateIntrplntFn
import orbax.checkpoint
from flax.training import orbax_utils
import operator
import time

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
        self.ckpt_dir.mkdir(exist_ok=True)
        (self.ckpt_dir / "model_params").mkdir(exist_ok=True)
        logging.info("Checkpoint directory: %s", self.ckpt_dir)
        # See https://jsonlines.org/ for the JSON Lines text file format
        self.metrics_file = self.out_dir / "metrics.jsonl"

        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

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
        self.ckpt_interval = self.cfg.get("ckpt_interval")
        self.jit_lossFn = self.cfg.get("jit_lossFn")

        self.results = {"Estimates":{"logZ":[],"mean_abs":[],"mean_sq":[]},
                        "Metrics":{"loss":[]}}



    def train(self):
        # runs = self.cfg.train.get("epoch_steps")
        delt = 0.05
        epochs = self.cfg.train.get("epochs")
        lr = self.cfg.train.get("learning_rate")
        sch_scale = self.cfg.train.get("scheduler_scale")
        sch_step = self.cfg.train.get("scheduler_step")
        grad_norm_max = self.cfg.train.get("grad_norm")
        tval_ramp = self.cfg.train.get("tval_ramp_time")
        schedule_boundaries = [i for i in range(sch_step, epochs, sch_step)]
        lr_schedule = optax.join_schedules([optax.constant_schedule(lr),optax.piecewise_interpolate_schedule('linear',lr,{i:sch_scale for i in schedule_boundaries})],[2*tval_ramp])
        # tval_schedule = optax.join_schedules([optax.linear_schedule(0.9,0.0,sch_step//4)]*len(schedule_boundaries),schedule_boundaries)
        #tval1_schedule = optax.linear_schedule(0.9,0.0,tval_ramp)#working
        tval1_schedule = optax.linear_schedule(self.T-delt,self.T0-delt,tval_ramp)#working
        # tval0_schedule = optax.join_schedules([optax.constant_schedule(0.95),optax.linear_schedule(0.95,0.0,sch_step)]*2,[sch_step])
        #tval0_schedule = optax.join_schedules([optax.constant_schedule(0.95),optax.linear_schedule(0.95,0.0,sch_step)]*2,[sch_step])#working
        tval0_schedule = optax.join_schedules([optax.constant_schedule(self.T0-delt),optax.linear_schedule(self.T0-delt,0.0,tval_ramp)],[tval_ramp])
        grad_clip_schedule = optax.piecewise_constant_schedule(10,{600:10.0, 2000:5.0})
        time_step_schedule = optax.piecewise_constant_schedule(1.0,{i:1.0 for i in schedule_boundaries})
        def opt_step(params, opt_state, schedules,L_prev,k):
            (L,metrics), grad_par = self.loss_grad(params,k,schedules)
            # if L/L_prev>2:
            #     L=L_prev
            #     print("Glitch!")
            # else:
            # val = L+jax.tree_util.tree_reduce(operator.add,jax.tree_util.tree_map(jnp.sum,grad_par))
            step_ok = jnp.isfinite(L) and not jnp.isnan(L)
            if step_ok:
                updates, opt_state = optimizer.update(grad_par, opt_state)
                params = optax.apply_updates(params, updates)
            else:
                print("Glitch!",L)
            return params, opt_state, L,metrics

        # k1, k2 = random.split(random.PRNGKey(np.random.randint(0, 1000)))
        k1,self.k = random.split(self.k)
        self.params_ema = self.params.copy()
        # optimizer = optax.adam(learning_rate=lr_schedule)

        # optimizer = optax.chain(optax.clip_by_global_norm(grad_norm_max),optax.adamw(lr_schedule,weight_decay=1e-7))

        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_norm_max),  # Clip by the gradient by the global norm.
           optax.scale_by_adam(),  # Use the updates from adam.
            optax.scale_by_schedule(lr_schedule),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )


        opt_state = optimizer.init(self.params)
        L = 1e6

        for step in range(epochs):
            # k1, k2 = random.split(k2)
            k1, self.k = random.split(self.k)
            sch = {"tval":((1-self.debug)*tval0_schedule(step),(1-self.debug)*tval1_schedule(step)),"grad_clip":grad_clip_schedule(step),"time_step":time_step_schedule(step)}
            self.params, opt_state, L,metrics = opt_step(self.params, opt_state, sch,L,k1)
            self.params_ema = optax.incremental_update(self.params, self.params_ema, 0.01)
            # step_log.append(steps)
            if step % self.log_interval == 0:
                # metrics["loss"]=L
                self.results["Metrics"]["loss"].append(float(metrics["loss"]))
                print(metrics)
                wandb.log(metrics)
                if self.debug:
                    trainTime = time.time()
                    if hasattr(self,"trainTime"):
                        timeAvg = (trainTime-self.trainTime)/self.log_interval
                        print("Avg time per iteration:",timeAvg)
                        wandb.log({"trainTime":timeAvg})
                    self.trainTime = trainTime

            if (step % self.eval_interval == 0) and (self.debug==0) :
                self.eval_inter(step)
            if (step % self.ckpt_interval == 0) and (self.debug==0):
                ckpt_name = "ckpt_"+str(step)
                save_args = orbax_utils.save_args_from_target(self.params_ema)
                self.orbax_checkpointer.save(self.ckpt_dir / "model_params" / ckpt_name, self.params_ema, save_args=save_args)

        # self.params=params_ema
        save_args = orbax_utils.save_args_from_target(self.params_ema)
        self.orbax_checkpointer.save(self.ckpt_dir / "model_params" / "final_ckpt" , self.params_ema, save_args=save_args)
        return self.params_ema

    def estimateFns(self,fns:list,samples,imp_wts):
        return [jnp.mean(vmap(fn)(samples)*imp_wts) for fn in fns]

    def eval_inter(self,step):
        name = 'hist_'+str(step)+'.png'
        outDir = self.out_dir / "output" / name
        outDir.parent.mkdir(parents=True, exist_ok=True)
        X,imp_wts = self.generateSamples()
        plotObj = plotter(X)
        hist_dims = self.cfg.eval.get("hist_dims")
        hist2d = plotObj.make_hist_plot2d(hist_dims)
        hist2d.savefig(outDir, bbox_inches='tight', pad_inches=0.1, dpi=300)
        est = self.estimateFns([lambda x:jnp.sum(jnp.abs(x)),lambda x:jnp.sum(x**2)],X,imp_wts)
        logZ = self.estimate_logZ()
        estimates = {"logZ":logZ,"mean_abs":est[0],"mean_sq":est[1]}
        self.updateResults(estimates)
        logging.info("Estimates: %s", estimates)
        # logging.info("logZ: %s", logZ)
        print(estimates)
        wandb.log(estimates)
        # print(logZ)
        # self.debugPlots()
    def eval(self):
        X,imp_wts = self.generateSamples()
        self.results["Samples"] = X
        self.results["Weights"] = imp_wts
        plotObj = plotter(X)
        hist_dims = self.cfg.eval.get("hist_dims")
        hist2d = plotObj.make_hist_plot2d(hist_dims)
        hist2d.savefig(Path(self.out_dir / "hist2d.png"), bbox_inches='tight', pad_inches=0.1, dpi=300)
        estimates = self.estimateFns([lambda x:jnp.sum(jnp.abs(x)),lambda x:jnp.sum(x**2)],X,imp_wts)
        logZ = self.estimate_logZ()
        logging.info("Estimates: %s", estimates)
        logging.info("logZ: %s", logZ)
        print(estimates)
        print(logZ)
        estFig = plotObj.plotEstimates(self.results["Estimates"])
        estFig.savefig(Path(self.out_dir / "estimates.png"), bbox_inches='tight', pad_inches=0.1, dpi=300)
        save_args = orbax_utils.save_args_from_target(self.results)
        self.orbax_checkpointer.save(self.ckpt_dir / "results", self.results, save_args=save_args)
        self.debugPlots()

    def debugPlots(self):
        name = 'marginal_velocity.png'
        debugDir = self.out_dir / "deb"/ name
        debugDir.parent.mkdir(parents=True, exist_ok=True)
        N=256
        xrange = 10
        xd1 = np.random.uniform(-1, 1, (self.d - 1,)).tolist()
        X = jnp.array([[-xrange + 2*xrange * i / N] + xd1 for i in range(N)])
        fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(6, 10))
        plt.tight_layout()
        for t in [self.eps0,0.25,0.5,0.75,1.0]: # np.linspace(self.eps0, self.T, 5):
            tv = (t) * np.ones([N, 1])
            if hasattr(self,"scoreFn"):
                Z = self.velocityFn(self.params_ema['params0'], tv, X)
                ax[0].plot(X[:, 0], Z[:, 0], label=str(t))
                Z = self.scoreFn(self.params_ema['params1'], tv, X)
                ax[1].plot(X[:, 0], Z[:, 0], label=str(t))
            else:
                Z = self.velocityFn(self.params_ema, tv, X)
                ax[0].plot(X[:, 0], Z[:, 0], label=str(t))

        Y0 = self.scoreFn(self.params_ema['params1'], jnp.ones((N, 1)) * self.T0, X * self.beta(self.T0)) * self.beta(self.T0) + self.terminalCondT0(X)
        Y = self.terminalCondT(X)
        ax[0].plot(X[:, 0], Y0[:, 0], marker='*', label='True 1.0')
        ax[1].plot(X[:, 0], Y[:, 0], marker='*', label='True 1.0')
        ax[0].set_ylim(-10,10)
        ax[1].set_ylim(-10,10)
        ax[0].legend()
        ax[1].legend()
        fig.savefig(debugDir, bbox_inches='tight', pad_inches=0.1, dpi=300)

    def updateResults(self,estimates):
        for key in self.results["Estimates"]:
            self.results["Estimates"][key].append(estimates[key])

class half_sis(sampler):

    def __init__(self,cfg:DictConfig):
        super().__init__(cfg=cfg)

        self.target, _ = call(self.cfg.target)
        self.target_score = vmap(grad(lambda x: self.target(x[None, :])[0]))
        self.intrplnt = linearInterpolant(*call(self.cfg.interpolant))
        self.beta = lambda t: self.intrplnt.r(t) / self.intrplnt.g(t)
        self.alpha = lambda t: 1.0
        self.sig0 = lambda t: jnp.sqrt(2 * (
                    (self.intrplnt.r(t) ** 2) * self.intrplnt.dg(t) / self.intrplnt.g(t) - self.intrplnt.r(
                t) * self.intrplnt.dr(t)))
        self.sigSampling = lambda t: 1.0
        self.T = self.cfg.get("T", 1.0)
        # self.T0 = self.cfg.get("T0", 0.5)
        self.d = self.cfg.get("dim")
        self.bs = self.cfg.get("batch_size")
        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")
        self.learn_pot = self.cfg.get("learn_potential")

        solver_name = self.cfg.hjb_solver.get("name")
        self.pde = backwardPDE(*self.generatePDEcoeffs())
        self.terminalCondT = jit(vmap(grad(lambda x: self.pde.phi(x[None, :])[0])))
        if solver_name=="oc":
            self.velocity_model = instantiate(self.cfg.velocity_model,target_score=self.terminalCondT)
        elif solver_name=="fbsde":
            self.velocity_model = instantiate(self.cfg.velocity_model, target_score=self.pde.phi)
        k1, self.k = random.split(self.k)
        xin = random.uniform(k1, (1,self.d))
        tin = random.uniform(k1, (1,1))
        k1, self.k = random.split(self.k)
        self.params = self.velocity_model.init(k1, tin, xin)
        self.params_ema = self.params.copy()

        solver_params = {"t0":self.eps0,"t1":self.T-self.eps1,"add_terminal_loss":1,"compute_process":1,"loss_bsde_scale":1.0,"learn_pot":self.learn_pot}
        self.hjb_solver = instantiate(self.cfg.hjb_solver.solver,self.cfg,self.pde,self.velocity_model.apply,self.intrplnt,solver_params)
        # self.hjb_solver = instantiate(self.cfg.hjb_solver,cfg=self.cfg)
        # self.hjb_solver = oc(cfg=self.cfg,pde=self.pde,model=self.velocity_model.apply,intrplnt=self.intrplnt)
        # self.hjb_solver = fbsde(cfg=self.cfg,pde=self.pde,model=self.velocity_model,intrplnt=self.intrplnt)
        self.velocityFn = self.hjb_solver.velocityFn


        if self.jit_lossFn:
            self.loss_grad = jit(jax.value_and_grad(self.lossFn,has_aux=True))
        else:
            self.loss_grad = jax.value_and_grad(self.lossFn,has_aux=True)
        self.NtTrain = self.cfg.train.get("NtTrain")

    def lossFn(self,params,k,schedules):

        k1, k2 = random.split(k)
        X = random.normal(k1, (self.bs, self.d))
        L,metrics,X = self.hjb_solver.lossFn(params,X,k2,0*schedules["tval"][0],schedules["time_step"])
        metrics["loss"]=L
        return L,metrics


    def generatePDEcoeffs(self):
        #only nn gives output bs*1
        logPsi = lambda t, x: -jnp.sum(x**2)/(2*self.intrplnt.r(t)**2)-self.d*jnp.log(2*jnp.pi*self.intrplnt.r(t)**2)/2
        self.logPsi = vmap(logPsi,(0,0),0)

        lin_drift_fun = lambda t: jnp.log(self.beta(t) * self.intrplnt.g(t) / (self.intrplnt.r(t) ** 2))
        lin_drift_coeff = vmap(grad(lin_drift_fun))

        s = lambda t: self.sig0(t) / self.beta(t)

        def mu(t, X):
            return  - lin_drift_coeff(t)[:,None] * X

        def f(t, X, sigGrad_u):
            return jnp.sum(sigGrad_u ** 2, axis=-1, keepdims=True) / 2

        phi = lambda x: self.target(self.beta(self.T)*x)-self.logPsi(jnp.ones((x.shape[0],))*self.T,self.beta(self.T)*x)
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
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))

        return X,jnp.ones((Ns,))

    def estimate_logZ(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr

        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")

        k1, self.k = random.split(self.k)
        X = random.normal(k1, (Ns, self.d)) * jnp.sqrt(0.0)
        delT = (self.T -self.eps1 - self.eps0) / Nt

        logZ = 0.0

        for i in range(0, Nt):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            s = self.sig0(t) / self.beta(t)
            Z = s*self.velocityFn(self.params_ema, tv, X )
            logZ += jnp.sum(Z**2,axis=-1)*delT/2
            mu = s * Z + dr(t)*X/r(t)
            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += delT * mu + jnp.sqrt(delT) * s * W
            logZ += jnp.sum(Z*W, axis=-1) * jnp.sqrt(delT)

        logZ -= self.pde.phi(X)
        return -jnp.mean(logZ)

class full_sis(sampler):

    def __init__(self,cfg:DictConfig):
        super().__init__(cfg=cfg)

        self.T = self.cfg.get("T", 1.0)
        self.T0 = self.cfg.get("T0", 0.5)

        self.target,_ = call(self.cfg.target)
        self.target_score = vmap(grad(lambda x:self.target(x[None,:])[0]))
        self.intrplnt = linearInterpolant(*call(self.cfg.interpolant),self.T)
        self.beta = lambda t: (self.intrplnt.r(t))/self.intrplnt.g(t)
        self.alpha = lambda t: 1.0
        self.sig0 = lambda t: jnp.sqrt(2 * ((self.intrplnt.r(t) ** 2) * self.intrplnt.dg(t) / self.intrplnt.g(t) - self.intrplnt.r(t) * self.intrplnt.dr(t)))
        self.sigSampling = lambda t:1.0

        self.d = self.cfg.get("dim")
        self.bs = self.cfg.get("batch_size")
        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")
        self.learn_pot = self.cfg.get("learn_potential")
        self.debug = self.cfg.get("debug")
        self.NtTrain = self.cfg.train.get("NtTrain")

        self.loss_bsde_scale = self.cfg.pde_solver.get("loss_bsde_scale",1.0)*(0.1*(self.d>=10)+1.0*(self.d<10))#*(1.0/(10**int(jnp.log10(self.d))))#

        solver_name = self.cfg.hjb_solver.get("name")

        self.pde0 = backwardPDE(*self.generatePDEcoeffs(kind=0))
        self.terminalCondT0 = jit(vmap(grad(lambda x: self.pde0.phi(x[None, :])[0])))
        if solver_name == "oc":
            self.velocity_model = instantiate(self.cfg.velocity_model, target_score=self.terminalCondT0)
        elif solver_name == "fbsde":
            self.velocity_model = instantiate(self.cfg.velocity_model, target_score=self.pde0.phi)

        solver_params = {"t0": self.eps0, "t1": self.T0, "add_terminal_loss": 0, "compute_process": 1,"loss_bsde_scale":self.loss_bsde_scale,"learn_pot":self.learn_pot,"beta":self.beta,"NtTrain":int(self.NtTrain*self.T0/self.T),"mode":0}

        self.hjb_solver_velocity = instantiate(self.cfg.hjb_solver.solver, self.cfg, self.pde0, self.velocity_model.apply,
                                            self.intrplnt, solver_params)



        self.pde1 = backwardPDE(*self.generatePDEcoeffs(kind=1))
        self.terminalCondT = jit(vmap(grad(lambda x: self.pde1.phi(x[None, :])[0])))
        if solver_name=="oc":
            self.score_model = instantiate(self.cfg.score_model,target_score=self.terminalCondT)
        elif solver_name=="fbsde":
            self.score_model = instantiate(self.cfg.score_model, target_score=self.pde1.phi)

        solver_params = {"t0":self.T0,"t1":self.T-self.eps1,"add_terminal_loss":1,"compute_process":0,"loss_bsde_scale":self.loss_bsde_scale,"learn_pot":self.learn_pot,"beta":self.alpha,"NtTrain":int(self.NtTrain*(self.T-self.T0)/self.T),"mode":1}

        self.hjb_solver_score = instantiate(self.cfg.hjb_solver.solver, self.cfg, self.pde1, self.score_model.apply,
                                      self.intrplnt,solver_params)


        k1, self.k = random.split(self.k)
        xin = random.uniform(k1, (1,self.d))
        tin = random.uniform(k1, (1,1))
        k1, self.k = random.split(self.k)
        params0 = self.velocity_model.init(k1, tin, xin)
        k1, self.k = random.split(self.k)
        params1 = self.score_model.init(k1, tin, xin)

        self.params = {"params0":params0,"params1":params1}
        self.params_ema = self.params.copy()
        # self.hjb_solver = instantiate(self.cfg.hjb_solver,cfg=self.cfg)
        # self.hjb_solver = oc(cfg=self.cfg,pde=self.pde,model=self.velocity_model.apply,intrplnt=self.intrplnt)
        # self.hjb_solver = fbsde(cfg=self.cfg,pde=self.pde,model=self.velocity_model,intrplnt=self.intrplnt)
        self.velocityFn = self.hjb_solver_velocity.velocityFn
        self.scoreFn = self.hjb_solver_score.velocityFn


        #for importance weights
        # self.pot_vel = lambda params,t,x: self.logPsi(t[:,0],x)[:,None]+self.hjb_solver_velocity.velocityPot(params,t,(1/self.beta(t))*x)
        # grad_xNN = grad(lambda params, t, x: self.pot_vel(params, t[:, None], x[None, :])[0, 0], 2)
        # def potLap_unbatched(params, t, x):
        #     return jnp.array([jnp.trace(jax.jacfwd(grad_xNN,2)(params,t,x))])
        #
        # grad_tNN = grad(lambda params, t, x: self.pot_vel(params, t[:, None], x[None, :])[0, 0], 1)
        #
        # self.potLap = jit(vmap(potLap_unbatched,(None,0,0),0))
        # self.pot_dt = jit(vmap(grad_tNN, (None, 0, 0), 0))
        ##

        if self.jit_lossFn:
            self.loss_grad = jit(jax.value_and_grad(self.lossFn,has_aux=True))
        else:
            self.loss_grad = jax.value_and_grad(self.lossFn,has_aux=True)
        self.NtTrain = self.cfg.train.get("NtTrain")

    def lossFn(self,params,k,schedules):
        t_ind = jnp.array((schedules["tval"][1]-self.T0+1)).astype(int)
        k1, k2 = random.split(k)
        X = random.normal(k1, (self.bs, self.d))
        L = 0.0
        loss_score_init = 0
        # Z0 = self.scoreFn(params['params1'], jnp.ones((self.bs, 1)) * 0.0, X)
        # loss_score_init = jnp.mean(jnp.power(((-X) - Z0), 2))#put alpha here
        L += loss_score_init
        k1, k2 = random.split(k2)
        l,velocity_metrics,X1 = self.hjb_solver_velocity.lossFn(params['params0'],X,k1,schedules["tval"][0],schedules["time_step"])
        L += l*(1-t_ind)*((1.0-2*schedules["tval"][0])**2)
        Z = self.velocityFn(params['params0'], jnp.ones((self.bs, 1)) * self.T0, X1)
        Zh = (jax.lax.stop_gradient(
            self.scoreFn(params['params1'], jnp.ones((self.bs, 1)) * self.T0, X1 * self.beta(self.T0))) *
              self.beta(self.T0) + self.terminalCondT0(X1))#include alpha here

        Y = self.hjb_solver_velocity.velocityPot(params['params0'], jnp.ones((self.bs, 1)) * self.T0, X1)
        Yh = jax.lax.stop_gradient(
            self.hjb_solver_score.velocityPot(params['params1'], jnp.ones((self.bs, 1)) * self.T0, X1 * self.beta(self.T0)))+ self.pde0.phi(X1)[:,None]-self.d*jnp.log(self.intrplnt.g(self.T0))

        loss_mid = jnp.mean(jnp.power((Z - Zh), 2)) +0*jnp.mean(jnp.power((Y - Yh), 2))
        if self.learn_pot:
            Y = self.hjb_solver_velocity.velocityPot(params['params0'], jnp.ones((self.bs, 1)) * self.T0, X1)
            Yh = jax.lax.stop_gradient(
                self.hjb_solver_score.velocityPot(params['params1'], jnp.ones((self.bs, 1)) * self.T0,
                                                  X1 * self.beta(self.T0))) + self.pde0.phi(X1)[:,
                                                                              None] - self.d * jnp.log(
                self.intrplnt.g(self.T0))
            loss_mid += jnp.mean(jnp.power((Y - Yh), 2))
        L += loss_mid*((2.0-2*schedules["tval"][1])**2)
        X1 = (1-t_ind)*X1+t_ind*X
        k1, k2 = random.split(k2)
        l, score_metrics, X = self.hjb_solver_score.lossFn(params['params1'], X1, k1,schedules["tval"][1],schedules["time_step"])
        L += l
        metrics = {"loss":L,"loss_score_init":loss_score_init,"loss_mid":loss_mid}
        metrics["velocity_metrics"] = velocity_metrics
        metrics["score_metrics"] = score_metrics
        # if jnp.isnan(L):
        #     print("glitch")
        return L,metrics

    def generatePDEcoeffs(self,kind):

        if kind==0:
            logPsi = lambda t, x: -jnp.sum(x**2)/(2*self.intrplnt.r(t)**2)-self.d*jnp.log(2*jnp.pi*self.intrplnt.r(t)**2)/2
            self.logPsi = vmap(logPsi,(0,0),0)

            lin_drift_fun = lambda t: -jnp.log(self.beta(t) * self.intrplnt.g(t) / (self.intrplnt.r(t) ** 2))
            lin_drift_coeff = vmap(grad(lin_drift_fun))

            s = lambda t: self.sig0(t) / self.beta(t)

            phi = lambda x: -self.logPsi(jnp.ones((x.shape[0],))*self.T0,self.beta(self.T0)*x)
        else:

            lin_drift_fun = lambda t: jnp.log(self.intrplnt.g(t) / self.alpha(t))
            lin_drift_coeff = vmap(grad(lin_drift_fun))

            s = lambda t: self.sig0(t) / self.alpha(t)

            phi = lambda x: self.target(self.alpha(self.T) * x) + self.d*jnp.log(self.intrplnt.g(self.T))

        def mu(t, X):
            return lin_drift_coeff(t)[:, None] * X

        def f(t, X, sigGrad_u):
            return  jnp.sum(sigGrad_u ** 2, axis=-1, keepdims=True) / 2

        return mu,s,f,phi

    def generateSamples(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr


        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")
        Nt1 = int(Nt*(self.T0/self.T))
        Nt2 = int(Nt*((self.T-self.T0)/self.T))
        k1, self.k = random.split(self.k)
        X = random.normal(k1,(Ns,self.d))*jnp.sqrt(1.0)
        delT = (self.T0 - self.eps0) / Nt1
        sampleTime = time.time()
        # imp_int = self.pot_vel(self.params_ema['params0'], jnp.ones((Ns, 1)) * 0, X )

        for i in range(0, Nt1):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)

            # imp_int += delT * (self.pot_dt(self.params_ema['params0'], tv, X) + 0.5 * (
            #             s ** 2) * self.potLap(self.params_ema['params0'], tv, X) + jnp.sum(score * mu,
            #                                                                                                 axis=-1,
            #                                                                                                 keepdims=True))
            # imp_int += jnp.sqrt(delT) * s * jnp.sum(score * W, axis=-1, keepdims=True)

            X += (mu * delT + s * W * jnp.sqrt(delT))

        delT = (self.T - self.T0 - self.eps1) / Nt2

        for i in range(0, Nt2):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.scoreFn(self.params_ema['params1'], tv, X )
            b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
            s = self.sigSampling(t)
            score = Z
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)

            # imp_int += delT*(self.hjb_solver_score.pot_dt(self.params_ema['params1'], tv, X )+0.5*(s**2)*self.hjb_solver_score.potLap(self.params_ema['params1'], tv, X )+jnp.sum(score*mu,axis=-1,keepdims=True))
            # imp_int += jnp.sqrt(delT)*s*jnp.sum(score*W,axis=-1,keepdims=True)

            X += (mu * delT + s * W * jnp.sqrt(delT))

        timeAvg = (time.time() - sampleTime) / Nt
        print("Avg time per sampling step = ", timeAvg)
        logging.info("Avg time per sampling step = %s", timeAvg)
        wandb.log({"SampleTime": timeAvg})
        # imp_wts = jnp.exp(self.target(X)-self.hjb_solver_score.velocityPot(self.params_ema['params1'],jnp.ones((Ns, 1)) * self.T,X)[:,0])
        imp_wts = jnp.ones((Ns,))
        # imp_wts = jnp.exp(self.target(X)-imp_int[:,0])
        normalization = jnp.mean(imp_wts)
        logging.info("Mean importance weight: %s", normalization)

        # for i in range(0,Nt):
        #
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     s = self.sig0(t)
        #     Z = s * self.scoreFn(self.params['params1'], tv, X)
        #     mu = (dg(t) / g(t)) * X + s * Z
        #     k1, self.k = random.split(self.k)
        #     W = random.normal(k1, X.shape)
        #     X += (mu * delT + s*W * jnp.sqrt(delT))

        return X,imp_wts/normalization
    def generateSamples_timing(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr


        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")
        Nt1 = int(Nt*(self.T0/self.T))
        Nt2 = int(Nt*((self.T-self.T0)/self.T))
        k1, self.k = random.split(self.k)
        X = random.normal(k1,(Ns,self.d))*jnp.sqrt(1.0)
        delT = (self.T0 - self.eps0) / Nt1
        sampleTime = time.time()
        # imp_int = self.pot_vel(self.params_ema['params0'], jnp.ones((Ns, 1)) * 0, X )
        # @jit
        def step0(i,X,delT,k):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            # k1, self.k = random.split(self.k)
            W = random.normal(k, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))

            return X

        for i in range(0,Nt1):
            k1, self.k = random.split(self.k)
            X = step0(i,X,delT,k1)

        # imp_int -= self.pot_vel(self.params_ema['params0'], jnp.ones((Ns, 1)) * self.T0, X)
        # imp_int += self.hjb_solver_score.velocityPot(self.params_ema['params1'], jnp.ones((Ns, 1)) * self.T0, X )
        delT = (self.T - self.T0 - self.eps1) / Nt2


        # @jit
        def step1(i,X,delT,k):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.scoreFn(self.params_ema['params1'], tv, X)
            b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
            s = self.sigSampling(t)
            score = Z
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            # k1, self.k = random.split(self.k)
            W = random.normal(k, X.shape)

            # imp_int += delT*(self.hjb_solver_score.pot_dt(self.params_ema['params1'], tv, X )+0.5*(s**2)*self.hjb_solver_score.potLap(self.params_ema['params1'], tv, X )+jnp.sum(score*mu,axis=-1,keepdims=True))
            # imp_int += jnp.sqrt(delT)*s*jnp.sum(score*W,axis=-1,keepdims=True)

            X += (mu * delT + s * W * jnp.sqrt(delT))
            return X

        for i in range(0, Nt2):
            k1, self.k = random.split(self.k)
            X = step1(i,X,delT,k1)

        print("Avg time per sampling step = ",(time.time()-sampleTime)/Nt)
        # imp_wts = jnp.exp(self.target(X)-self.hjb_solver_score.velocityPot(self.params_ema['params1'],jnp.ones((Ns, 1)) * self.T,X)[:,0])
        imp_wts = jnp.ones((Ns,))
        # imp_wts = jnp.exp(self.target(X)-imp_int[:,0])
        normalization = jnp.mean(imp_wts)
        logging.info("Mean importance weight: %s", normalization)

        # for i in range(0,Nt):
        #
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     s = self.sig0(t)
        #     Z = s * self.scoreFn(self.params['params1'], tv, X)
        #     mu = (dg(t) / g(t)) * X + s * Z
        #     k1, self.k = random.split(self.k)
        #     W = random.normal(k1, X.shape)
        #     X += (mu * delT + s*W * jnp.sqrt(delT))

        return X,imp_wts/normalization

    def estimate_logZ(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr

        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler") // 2

        k1, self.k = random.split(self.k)
        X = random.normal(k1, (Ns, self.d)) * jnp.sqrt(0.0)
        delT = (self.T0 - self.eps0) / Nt

        logZ = 0.0

        for i in range(0, Nt):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            s = self.sig0(t) / self.beta(t)
            Z = s*self.velocityFn(self.params_ema['params0'], tv, X )
            logZ += jnp.sum(Z**2,axis=-1)*delT/2
            mu = s * Z + dr(t)*X/r(t)
            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += delT * mu + jnp.sqrt(delT) * s * W
            logZ += jnp.sum(Z*W, axis=-1) * jnp.sqrt(delT)



        logZ -= self.pde0.phi(X)
        logZ += self.d*jnp.log(g(self.T0))
        X = self.beta(self.T0) * X
        delT = (self.T - self.eps1-self.T0) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            s = self.sig0(t)
            Z = s*self.scoreFn(self.params_ema['params1'], tv,  X )
            logZ += jnp.sum(Z**2,axis=-1)*delT/2
            mu = s * Z + dg(t)*X/g(t)
            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += delT * mu + jnp.sqrt(delT) * s * W
            logZ += jnp.sum(Z * W, axis=-1) * jnp.sqrt(delT)
        logZ -= self.target(X)
        logZ -= self.d * jnp.log(g(self.T))
        return -jnp.mean(logZ)

    def genTraj(self,Ntraj = 100):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr


        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")//2

        samples = np.zeros((Ntraj+1,Ns,self.d))
        trajIntrvl = (2*Nt)//Ntraj

        k1, self.k = random.split(self.k)
        X = random.normal(k1,(Ns,self.d))*jnp.sqrt(1.0)
        delT = (self.T0 - self.eps0) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))
            if i%trajIntrvl==0:
                samples[int(i/trajIntrvl),:,:] = np.array(X)

        delT = (self.T - self.T0 - self.eps1) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.scoreFn(self.params_ema['params1'], tv, X )
            b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
            s = self.sigSampling(t)
            score = Z
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))
            if i%trajIntrvl==0:
                samples[int((Nt+i)/trajIntrvl),:,:] = np.array(X)

        # @jit
        # def step1(i,X,delT,k):
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     Z = self.scoreFn(self.params_ema['params1'], tv, X)
        #     b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
        #     s = self.sigSampling(t)
        #     score = Z
        #     # Z = s * score
        #     mu = b + (s ** 2) * score / 2
        #
        #     W = random.normal(k, X.shape)
        #     X += (mu * delT + s * W * jnp.sqrt(delT))
        #     return X
        #
        # for i in range(0, Nt):
        #     k1, self.k = random.split(self.k)
        #     X = step1(i,X,delT,k1)
        samples[-1,:,:] = np.array(X)
        imp_wts = jnp.exp(self.target(X)-self.hjb_solver_score.velocityPot(self.params_ema['params1'],jnp.ones((Ns, 1)) * self.T,X)[:,0])
        normalization = jnp.mean(imp_wts)
        logging.info("Mean importance weight: %s", normalization)

        # for i in range(0,Nt):
        #
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     s = self.sig0(t)
        #     Z = s * self.scoreFn(self.params['params1'], tv, X)
        #     mu = (dg(t) / g(t)) * X + s * Z
        #     k1, self.k = random.split(self.k)
        #     W = random.normal(k1, X.shape)
        #     X += (mu * delT + s*W * jnp.sqrt(delT))

        return samples,imp_wts/normalization

class half_sis2(sampler):

    def __init__(self,cfg:DictConfig):
        super().__init__(cfg=cfg)

        self.T = self.cfg.get("T", 1.0)
        self.T0 = self.cfg.get("T0", 0.5)

        self.target,_ = call(self.cfg.target)
        self.target_score = vmap(grad(lambda x:self.target(x[None,:])[0]))
        self.intrplnt = linearInterpolant(*call(self.cfg.interpolant),self.T)
        self.beta = lambda t: (self.intrplnt.r(t))/self.intrplnt.g(t)
        self.alpha = lambda t: 1.0
        self.sig0 = lambda t: jnp.sqrt(2 * ((self.intrplnt.r(t) ** 2) * self.intrplnt.dg(t) / self.intrplnt.g(t) - self.intrplnt.r(t) * self.intrplnt.dr(t)))
        self.sigSampling = lambda t:1.0

        self.d = self.cfg.get("dim")
        self.bs = self.cfg.get("batch_size")
        self.eps0 = self.cfg.get("eps0")
        self.eps1 = self.cfg.get("eps1")
        self.learn_pot = self.cfg.get("learn_potential")
        self.debug = self.cfg.get("debug")
        self.NtTrain = self.cfg.train.get("NtTrain")

        solver_name = self.cfg.hjb_solver.get("name")

        self.pde0 = backwardPDE(*self.generatePDEcoeffs(kind=0))
        self.terminalCondT0 = jit(vmap(grad(lambda x: self.pde0.phi(x[None, :])[0])))
        if solver_name == "oc":
            self.velocity_model = instantiate(self.cfg.velocity_model, target_score=self.terminalCondT0)
        elif solver_name == "fbsde":
            self.velocity_model = instantiate(self.cfg.velocity_model, target_score=self.pde0.phi)

        solver_params = {"t0": self.eps0, "t1": self.T0, "add_terminal_loss": 0, "compute_process": 1,"loss_bsde_scale":1.0,"learn_pot":self.learn_pot,"beta":self.beta,"NtTrain":int(self.NtTrain*self.T0/self.T),"mode":0}

        self.hjb_solver_velocity = instantiate(self.cfg.hjb_solver.solver, self.cfg, self.pde0, self.velocity_model.apply,
                                            self.intrplnt, solver_params)



        self.pde1 = backwardPDE(*self.generatePDEcoeffs(kind=1))
        self.terminalCondT = jit(vmap(grad(lambda x: self.pde1.phi(x[None, :])[0])))
        if solver_name=="oc":
            self.score_model = instantiate(self.cfg.score_model,target_score=self.terminalCondT)
        elif solver_name=="fbsde":
            self.score_model = instantiate(self.cfg.score_model, target_score=self.pde1.phi)

        solver_params = {"t0":self.T0,"t1":self.T-self.eps1,"add_terminal_loss":1,"compute_process":1,"loss_bsde_scale":1.0,"learn_pot":self.learn_pot,"beta":self.beta,"NtTrain":int(self.NtTrain*(self.T-self.T0)/self.T),"mode":1}

        self.hjb_solver_score = instantiate(self.cfg.hjb_solver.solver, self.cfg, self.pde1, self.score_model.apply,
                                      self.intrplnt,solver_params)


        k1, self.k = random.split(self.k)
        xin = random.uniform(k1, (1,self.d))
        tin = random.uniform(k1, (1,1))
        k1, self.k = random.split(self.k)
        params0 = self.velocity_model.init(k1, tin, xin)
        k1, self.k = random.split(self.k)
        params1 = self.score_model.init(k1, tin, xin)

        self.params = {"params0":params0,"params1":params1}
        self.params_ema = self.params.copy()
        # self.hjb_solver = instantiate(self.cfg.hjb_solver,cfg=self.cfg)
        # self.hjb_solver = oc(cfg=self.cfg,pde=self.pde,model=self.velocity_model.apply,intrplnt=self.intrplnt)
        # self.hjb_solver = fbsde(cfg=self.cfg,pde=self.pde,model=self.velocity_model,intrplnt=self.intrplnt)
        self.velocityFn = self.hjb_solver_velocity.velocityFn
        self.scoreFn = self.hjb_solver_score.velocityFn


        #for importance weights
        # self.pot_vel = lambda params,t,x: self.logPsi(t[:,0],x)[:,None]+self.hjb_solver_velocity.velocityPot(params,t,(1/self.beta(t))*x)
        # grad_xNN = grad(lambda params, t, x: self.pot_vel(params, t[:, None], x[None, :])[0, 0], 2)
        # def potLap_unbatched(params, t, x):
        #     return jnp.array([jnp.trace(jax.jacfwd(grad_xNN,2)(params,t,x))])
        #
        # grad_tNN = grad(lambda params, t, x: self.pot_vel(params, t[:, None], x[None, :])[0, 0], 1)
        #
        # self.potLap = jit(vmap(potLap_unbatched,(None,0,0),0))
        # self.pot_dt = jit(vmap(grad_tNN, (None, 0, 0), 0))
        ##

        if self.jit_lossFn:
            self.loss_grad = jit(jax.value_and_grad(self.lossFn,has_aux=True))
        else:
            self.loss_grad = jax.value_and_grad(self.lossFn,has_aux=True)
        self.NtTrain = self.cfg.train.get("NtTrain")

    def lossFn(self,params,k,schedules):
        t_ind = jnp.array((schedules["tval"][1]-self.T0+1)).astype(int)
        k1, k2 = random.split(k)
        X = random.normal(k1, (self.bs, self.d))
        L = 0.0
        loss_score_init = 0
        # Z0 = self.scoreFn(params['params1'], jnp.ones((self.bs, 1)) * 0.0, X)
        # loss_score_init = jnp.mean(jnp.power(((-X) - Z0), 2))#put alpha here
        L += loss_score_init
        k1, k2 = random.split(k2)
        l,velocity_metrics,X1 = self.hjb_solver_velocity.lossFn(params['params0'],X,k1,schedules["tval"][0],schedules["time_step"])
        L += l*(1-t_ind)*((1.0-2*schedules["tval"][0])**2)
        Z = self.velocityFn(params['params0'], jnp.ones((self.bs, 1)) * self.T0, X1)
        Zh = jax.lax.stop_gradient(
            self.scoreFn(params['params1'], jnp.ones((self.bs, 1)) * self.T0, X1 ))
             #include alpha here

        Y = self.hjb_solver_velocity.velocityPot(params['params0'], jnp.ones((self.bs, 1)) * self.T0, X1)
        Yh = jax.lax.stop_gradient(
            self.hjb_solver_score.velocityPot(params['params1'], jnp.ones((self.bs, 1)) * self.T0, X1 ))

        loss_mid = jnp.mean(jnp.power((Z - Zh), 2)) +0*jnp.mean(jnp.power((Y - Yh), 2))
        if self.learn_pot:
            Y = self.hjb_solver_velocity.velocityPot(params['params0'], jnp.ones((self.bs, 1)) * self.T0, X1)
            Yh = jax.lax.stop_gradient(
                self.hjb_solver_score.velocityPot(params['params1'], jnp.ones((self.bs, 1)) * self.T0, X1))
            loss_mid += jnp.mean(jnp.power((Y - Yh), 2))
        L += loss_mid*((2.0-2*schedules["tval"][1])**2)
        X1 = (1-t_ind)*X1+t_ind*X
        k1, k2 = random.split(k2)
        l, score_metrics, X = self.hjb_solver_score.lossFn(params['params1'], X1, k1,schedules["tval"][1],schedules["time_step"])
        L += l
        metrics = {"loss":L,"loss_score_init":loss_score_init,"loss_mid":loss_mid}
        metrics["velocity_metrics"] = velocity_metrics
        metrics["score_metrics"] = score_metrics
        # if jnp.isnan(L):
        #     print("glitch")
        return L,metrics

    def generatePDEcoeffs(self,kind):

        logPsi = lambda t, x: -jnp.sum(x ** 2) / (2 * self.intrplnt.r(t) ** 2) - self.d * jnp.log(
            2 * jnp.pi * self.intrplnt.r(t) ** 2) / 2
        self.logPsi = vmap(logPsi, (0, 0), 0)

        lin_drift_fun = lambda t: -jnp.log(self.beta(t) * self.intrplnt.g(t) / (self.intrplnt.r(t) ** 2))
        lin_drift_coeff = vmap(grad(lin_drift_fun))

        s = lambda t: self.sig0(t) / self.beta(t)

        if kind==0:
            phi = lambda x: -self.logPsi(jnp.ones((x.shape[0],))*self.T0,self.beta(self.T0)*x)
        else:
            phi = lambda x: self.target(self.beta(self.T) * x) - self.logPsi(jnp.ones((x.shape[0],)) * self.T,
                                                                             self.beta(self.T) * x)
        def mu(t, X):
            return lin_drift_coeff(t)[:, None] * X

        def f(t, X, sigGrad_u):
            return  jnp.sum(sigGrad_u ** 2, axis=-1, keepdims=True) / 2

        return mu,s,f,phi

    def generateSamples(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr


        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")
        Nt1 = int(Nt*(self.T0/self.T))
        Nt2 = int(Nt*((self.T-self.T0)/self.T))
        k1, self.k = random.split(self.k)
        X = random.normal(k1,(Ns,self.d))*jnp.sqrt(1.0)
        delT = (self.T0 - self.eps0) / Nt1

        # imp_int = self.pot_vel(self.params_ema['params0'], jnp.ones((Ns, 1)) * 0, X )

        for i in range(0, Nt1):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)

            # imp_int += delT * (self.pot_dt(self.params_ema['params0'], tv, X) + 0.5 * (
            #             s ** 2) * self.potLap(self.params_ema['params0'], tv, X) + jnp.sum(score * mu,
            #                                                                                                 axis=-1,
            #                                                                                                 keepdims=True))
            # imp_int += jnp.sqrt(delT) * s * jnp.sum(score * W, axis=-1, keepdims=True)

            X += (mu * delT + s * W * jnp.sqrt(delT))
        # @jit
        # def step0(i,X,delT,k):
        #     t = (i * delT) + self.eps0
        #     tv = jnp.ones((Ns, 1)) * t
        #     Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
        #     b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
        #     s = self.sigSampling(t)
        #     score = g(t) * Z / r(t) - X / (r(t) ** 2)
        #     # Z = s * score
        #     mu = b + (s ** 2) * score / 2
        #
        #     W = random.normal(k, X.shape)
        #     X += (mu * delT + s * W * jnp.sqrt(delT))
        #     return X
        #
        # for i in range(0,Nt):
        #     k1, self.k = random.split(self.k)
        #     X = step0(i,X,delT,k1)

        # imp_int -= self.pot_vel(self.params_ema['params0'], jnp.ones((Ns, 1)) * self.T0, X)
        # imp_int += self.hjb_solver_score.velocityPot(self.params_ema['params1'], jnp.ones((Ns, 1)) * self.T0, X )
        delT = (self.T - self.T0 - self.eps1) / Nt2

        for i in range(0, Nt2):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.scoreFn(self.params_ema['params1'], tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)

            # imp_int += delT * (self.pot_dt(self.params_ema['params0'], tv, X) + 0.5 * (
            #             s ** 2) * self.potLap(self.params_ema['params0'], tv, X) + jnp.sum(score * mu,
            #                                                                                                 axis=-1,
            #                                                                                                 keepdims=True))
            # imp_int += jnp.sqrt(delT) * s * jnp.sum(score * W, axis=-1, keepdims=True)

            X += (mu * delT + s * W * jnp.sqrt(delT))

        # @jit
        # def step1(i,X,delT,k):
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     Z = self.scoreFn(self.params_ema['params1'], tv, X)
        #     b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
        #     s = self.sigSampling(t)
        #     score = Z
        #     # Z = s * score
        #     mu = b + (s ** 2) * score / 2
        #
        #     W = random.normal(k, X.shape)
        #     X += (mu * delT + s * W * jnp.sqrt(delT))
        #     return X
        #
        # for i in range(0, Nt):
        #     k1, self.k = random.split(self.k)
        #     X = step1(i,X,delT,k1)

        # imp_wts = jnp.exp(self.target(X)-self.hjb_solver_score.velocityPot(self.params_ema['params1'],jnp.ones((Ns, 1)) * self.T,X)[:,0])
        imp_wts = jnp.ones((Ns,))
        # imp_wts = jnp.exp(self.target(X)-imp_int[:,0])
        normalization = jnp.mean(imp_wts)
        logging.info("Mean importance weight: %s", normalization)

        # for i in range(0,Nt):
        #
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     s = self.sig0(t)
        #     Z = s * self.scoreFn(self.params['params1'], tv, X)
        #     mu = (dg(t) / g(t)) * X + s * Z
        #     k1, self.k = random.split(self.k)
        #     W = random.normal(k1, X.shape)
        #     X += (mu * delT + s*W * jnp.sqrt(delT))

        return X,imp_wts/normalization

    def generateSamples_old(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr


        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")//2
        k1, self.k = random.split(self.k)
        X = random.normal(k1,(Ns,self.d))*jnp.sqrt(1.0)
        delT = (self.T0 - self.eps0) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))
        # @jit
        # def step0(i,X,delT,k):
        #     t = (i * delT) + self.eps0
        #     tv = jnp.ones((Ns, 1)) * t
        #     Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
        #     b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
        #     s = self.sigSampling(t)
        #     score = g(t) * Z / r(t) - X / (r(t) ** 2)
        #     # Z = s * score
        #     mu = b + (s ** 2) * score / 2
        #
        #     W = random.normal(k, X.shape)
        #     X += (mu * delT + s * W * jnp.sqrt(delT))
        #     return X
        #
        # for i in range(0,Nt):
        #     k1, self.k = random.split(self.k)
        #     X = step0(i,X,delT,k1)

        delT = (self.T - self.T0 - self.eps1) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.scoreFn(self.params_ema['params1'], tv, X )
            b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
            s = self.sigSampling(t)
            score = Z
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))

        # @jit
        # def step1(i,X,delT,k):
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     Z = self.scoreFn(self.params_ema['params1'], tv, X)
        #     b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
        #     s = self.sigSampling(t)
        #     score = Z
        #     # Z = s * score
        #     mu = b + (s ** 2) * score / 2
        #
        #     W = random.normal(k, X.shape)
        #     X += (mu * delT + s * W * jnp.sqrt(delT))
        #     return X
        #
        # for i in range(0, Nt):
        #     k1, self.k = random.split(self.k)
        #     X = step1(i,X,delT,k1)

        imp_wts = jnp.exp(self.target(X)-self.hjb_solver_score.velocityPot(self.params_ema['params1'],jnp.ones((Ns, 1)) * self.T,X)[:,0])
        normalization = jnp.mean(imp_wts)
        logging.info("Mean importance weight: %s", normalization)

        # for i in range(0,Nt):
        #
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     s = self.sig0(t)
        #     Z = s * self.scoreFn(self.params['params1'], tv, X)
        #     mu = (dg(t) / g(t)) * X + s * Z
        #     k1, self.k = random.split(self.k)
        #     W = random.normal(k1, X.shape)
        #     X += (mu * delT + s*W * jnp.sqrt(delT))

        return X,imp_wts/normalization

    def estimate_logZ(self):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr

        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler") // 2

        k1, self.k = random.split(self.k)
        X = random.normal(k1, (Ns, self.d)) * jnp.sqrt(0.0)
        delT = (self.T0 - self.eps0) / Nt

        logZ = 0.0

        for i in range(0, Nt):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            s = self.sig0(t) / self.beta(t)
            Z = s*self.velocityFn(self.params_ema['params0'], tv, X )
            logZ += jnp.sum(Z**2,axis=-1)*delT/2
            mu = s * Z + dr(t)*X/r(t)
            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += delT * mu + jnp.sqrt(delT) * s * W
            logZ += jnp.sum(Z*W, axis=-1) * jnp.sqrt(delT)



        # logZ -= self.pde0.phi(X)
        # logZ += self.d*jnp.log(g(self.T0))
        # X = self.beta(self.T0) * X
        delT = (self.T - self.eps1-self.T0) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            s = self.sig0(t) / self.beta(t)
            Z = s * self.scoreFn(self.params_ema['params1'], tv, X)
            logZ += jnp.sum(Z ** 2, axis=-1) * delT / 2
            mu = s * Z + dr(t) * X / r(t)
            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += delT * mu + jnp.sqrt(delT) * s * W
            logZ += jnp.sum(Z * W, axis=-1) * jnp.sqrt(delT)

        logZ -= self.pde1.phi(X)
        # logZ -= self.d * jnp.log(g(self.T))
        return -jnp.mean(logZ)

    def genTraj(self,Ntraj = 100):
        g = self.intrplnt.g
        dg = self.intrplnt.dg
        r = self.intrplnt.r
        dr = self.intrplnt.dr


        Ns = self.cfg.sampler.get("Nsamples")
        Nt = self.cfg.sampler.get("NtSampler")//2

        samples = np.zeros((Ntraj+1,Ns,self.d))
        trajIntrvl = (2*Nt)//Ntraj

        k1, self.k = random.split(self.k)
        X = random.normal(k1,(Ns,self.d))*jnp.sqrt(1.0)
        delT = (self.T0 - self.eps0) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.eps0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
            b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
            s = self.sigSampling(t)
            score = g(t) * Z / r(t) - X / (r(t) ** 2)
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))
            if i%trajIntrvl==0:
                samples[int(i/trajIntrvl),:,:] = np.array(X)

        delT = (self.T - self.T0 - self.eps1) / Nt

        for i in range(0, Nt):
            t = (i * delT) + self.T0
            tv = jnp.ones((Ns, 1)) * t
            Z = self.scoreFn(self.params_ema['params1'], tv, X )
            b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
            s = self.sigSampling(t)
            score = Z
            # Z = s * score
            mu = b + (s ** 2) * score / 2

            k1, self.k = random.split(self.k)
            W = random.normal(k1, X.shape)
            X += (mu * delT + s * W * jnp.sqrt(delT))
            if i%trajIntrvl==0:
                samples[int((Nt+i)/trajIntrvl),:,:] = np.array(X)

        # @jit
        # def step1(i,X,delT,k):
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     Z = self.scoreFn(self.params_ema['params1'], tv, X)
        #     b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
        #     s = self.sigSampling(t)
        #     score = Z
        #     # Z = s * score
        #     mu = b + (s ** 2) * score / 2
        #
        #     W = random.normal(k, X.shape)
        #     X += (mu * delT + s * W * jnp.sqrt(delT))
        #     return X
        #
        # for i in range(0, Nt):
        #     k1, self.k = random.split(self.k)
        #     X = step1(i,X,delT,k1)
        samples[-1,:,:] = np.array(X)
        imp_wts = jnp.exp(self.target(X)-self.hjb_solver_score.velocityPot(self.params_ema['params1'],jnp.ones((Ns, 1)) * self.T,X)[:,0])
        normalization = jnp.mean(imp_wts)
        logging.info("Mean importance weight: %s", normalization)

        # for i in range(0,Nt):
        #
        #     t = (i * delT) + self.T0
        #     tv = jnp.ones((Ns, 1)) * t
        #     s = self.sig0(t)
        #     Z = s * self.scoreFn(self.params['params1'], tv, X)
        #     mu = (dg(t) / g(t)) * X + s * Z
        #     k1, self.k = random.split(self.k)
        #     W = random.normal(k1, X.shape)
        #     X += (mu * delT + s*W * jnp.sqrt(delT))

        return samples,imp_wts/normalization

