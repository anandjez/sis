import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from omegaconf import DictConfig,OmegaConf
import jax.numpy as jnp
import jax
import math
from jax import grad,jit,vmap,random

import orbax.checkpoint
from flax.training import orbax_utils
from pathlib import Path

import seaborn as sns
import pandas as pd
import logging
from hydra.utils import instantiate
from stint_sampler.targets.targets import gmm,double_well,student
import time
from functools import partial

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

homePath = Path.home()

outPath = homePath / "python/sis/plots"


def genSampler(modelPath,outPath):
    configPath = modelPath / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(configPath)

    # Setup dir
    OmegaConf.set_struct(cfg, False)
    cfg.keops_build_path = "None"
    out_dir = outPath.absolute()
    logging.info("Hydra and wandb output path: %s", out_dir)
    if not cfg.get("out_dir"):
        cfg.out_dir = str(out_dir)
    logging.info("Solver output path: %s", cfg.out_dir)
    logging.info("---------------------------------------------------------------")
    logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    # try:
    # results = orbax_checkpointer.restore(genSamplePath / "ckpt" / "results")
    sampler = instantiate(cfg.solver, cfg)
    ckptPath = modelPath / "ckpt" / "model_params" / "final_ckpt"
    d = cfg.get("dim")
    params = orbax_checkpointer.restore(ckptPath)
    sampler.params_ema = params

    def genSamples(Nsamples,Nt):
        sampler.cfg.sampler.Nsamples = Nsamples
        sampler.cfg.sampler.NtSampler = Nt
        samples,weights = sampler.generateSamples_timing()

        return samples

    return genSamples

def langevin(target_score,d,Nsamples,Nt,alpha=1.0,delt=1e-4,inner_Nt = 2):#delt=1e-2
    k1,k2 = random.split(random.PRNGKey(np.random.randint(0,100)))
    X = random.normal(k1,(Nsamples,d))
    sampleTime = time.time()
    delt = delt/d
    # @jit
    def step(X,k2):
        for i in range(inner_Nt):
            k1, k2 = random.split(k2)
            W = random.normal(k1, (Nsamples, d))
            X += alpha * target_score(X) * delt + jnp.sqrt(2 * alpha * delt) * W
        return X

    for i in range(Nt):
        k1,k2 = random.split(k2)
        X = step(X,k1)

    print("Langevin: Avg time per sampling step =", (time.time()-sampleTime)/Nt)
    return X

# dists = [gmm,partial(double_well,d=10,w=3,delta=2),partial(double_well,d=20,w=5,delta=3)]
# modelPath = [homePath / "python/sis/slurm/logs/2024-05-18/13-02-11/interpolant.type.trig-seed.3-target.gmm",
#              homePath / "python/sis/slurm/logs/2024-05-18/13-02-11/interpolant.type.trig-seed.3-target.double_well",
#              homePath / "python/sis/slurm/logs/2024-05-18/13-02-11/interpolant.type.trig-seed.3-target.double_well20"
#              ]
#
# dim = [2,10,20]

dists = [student]
modelPath = [homePath / "python/sis/slurm/logs/2024-08-03/14-57-28"
             ]

dim = [20]

for j in range(len(dists)):
    target,sampler_gt = dists[j]()
    target_score = vmap(grad(lambda x:target(x[None,:])[0]))

    sampler_model = genSampler(modelPath[j],outPath)

    Ns = 10000
    Nt = [10**i for i in range(1,5)]

    estimate=1
    plot = 0
    print("Dim={0}".format(dim[j]))
    if estimate:
        est_fns = {"Mean Absolute":lambda x:jnp.sum(jnp.abs(x)),
                   "Mean Squared":lambda x:jnp.sum(x**2)}
        for i in range(len(Nt)):
            samples_model = sampler_model(Ns, Nt[i])
            samples_langevin = langevin(target_score,dim[j],Ns, Nt[i], alpha=1.0)
            print("N={0}".format(i))
            for fns in est_fns.keys():
                fn = vmap(est_fns[fns])
                mean_model = jnp.mean(fn(samples_model))
                mean_lgvn = jnp.mean(fn(samples_langevin))
                # std_model = jnp.std(fn(samples_model))
                # std_lgvn = jnp.std(fn(samples_langevin))
                print("{0}: Model-({1},), Langevin-({2},)".format(fns,mean_model,
                                    mean_lgvn))


    if plot:
        fig, ax = plt.subplots(nrows=2,ncols=len(Nt),figsize=(3*len(Nt), 3*2))
        ax[0,0].set_ylabel("Langevin")
        ax[1,0].set_ylabel("Full interpolant sampler")
        for i in range(len(Nt)):
            samples_model = sampler_model(Ns,Nt[i])
            samples_langevin = langevin(Ns,Nt[i],alpha=0.1)
            sns.kdeplot(x=samples_model[:, 0], y=samples_model[:, 1], fill=True, cmap="Blues", ax=ax[1,i],bw_adjust=1.0)
            sns.kdeplot(x=samples_langevin[:, 0], y=samples_langevin[:, 1], fill=True, cmap="Blues", ax=ax[0,i],bw_adjust=1.0)
            ax[1, i].set_xlabel("N = "+str(Nt[i]))
        fig.savefig(outPath / "langevin.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()