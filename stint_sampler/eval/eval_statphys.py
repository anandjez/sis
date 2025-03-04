import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from omegaconf import DictConfig,OmegaConf
import jax.numpy as jnp
import jax
import math



import orbax.checkpoint
from flax.training import orbax_utils
from pathlib import Path

import seaborn as sns
import pandas as pd
import logging
from hydra.utils import instantiate

logging.basicConfig(level=logging.INFO)

import jax.numpy as jnp
import pathlib

homePath = pathlib.Path.home()

path = homePath / "python/sis/plots"

evalModelPath = [homePath / "python/sis/slurm/logs/2024-05-21/12-47-47"]

def evalResults(pathList,outPath):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    samples_full = []
    weights_full = []
    evalsList = []
    for path in pathList:
        runPaths = [x for x in path.iterdir() if x.is_dir()]
        evals = []
        for runPath in runPaths:
            sweepPars = runPath.name.split("-")
            ckptPath = runPath / "ckpt" / "results"
            results = orbax_checkpointer.restore(ckptPath)
            evals_local = {}
            d = results["Samples"].shape[-1]
            q = jnp.mean(jnp.mean(results["Samples"],axis=0)**2)
            mn = jnp.mean(jnp.sum(results["Samples"]**2,axis=-1))
            mnSq = jnp.mean(jnp.sum(results["Samples"]**2,axis=-1)**2)
            var = (1/(4*d))*(mnSq-mn**2)
            mean = mn/(2*d)
            for par in sweepPars:
                attr = par.split(".")
                if attr[-2].isnumeric():
                    if attr[-3].isnumeric():
                        evals_local[".".join(attr[:-3])] = [float(".".join(attr[-3:]))]
                    else:
                        evals_local[".".join(attr[:-2])] = [float(".".join(attr[-2:]))]
                else:
                    evals_local[".".join(attr[:-1])] = [attr[-1]]
            # evals_local["Function"] = ["logZ"]
            evals_local["logZ"] = [float(results["Estimates"]["logZ"][-1])]
            # evals.append(pd.DataFrame(evals_local))
            # evals_local["Function"] = ["q"]
            evals_local["q"] = [float(q)]
            evals_local["var"] = [float(var)]
            evals_local["mean"] = [float(mean)]
            evals.append(pd.DataFrame(evals_local))
            # for fn in evalFns.keys():
            #     evals_local["Function"] = [fn]
            #     fn_evals = jax.vmap(evalFns[fn])(results["Samples"])*results["Weights"]
            #     mean = jnp.mean(fn_evals)
            #     std = jnp.sqrt(jnp.mean((fn_evals-mean)**2))
            #     evals_local["Value"] = [float(mean)]
            #     # evals_local["Std"] = [float(std)]
            #     evals.append(pd.DataFrame(evals_local))

        evals = pd.concat(evals)
        # evals.to_pickle(path / "eval_results.pkl")
        evalsList.append(evals)
    logging.info("Completed ✅")
    evals_pd = pd.concat(evalsList,ignore_index=True)
    # evals_pd.to_pickle(outPath / "eval_results_final.pkl")
    return evals_pd

def theory(beta,lbda):
    if beta<=1-lbda:
        return -0.5*jnp.log(1-lbda)
    else:
        q = (beta-(1-lbda))/(beta**2)
        return  -0.5*jnp.log(beta)+beta*q/2+(beta**2)*(q**2)/4

beta = 0.5
grad1Theory = jax.grad(theory,argnums=1)
grad2Theory = jax.grad(grad1Theory,argnums=1)
lbda_fine = np.linspace(0,1,50)
evals = evalResults(evalModelPath,path)
evals = evals.sort_values("target.lbda")
fig,ax = plt.subplots(figsize=[6,4])
ax.plot(evals["target.lbda"],evals["var"],label="FIS_var",linewidth=0.8,marker='*')
ax.plot(evals["target.lbda"],evals["mean"],label="FIS_mean",linewidth=0.8,marker='o')
# ax.plot(lbda,sp,label=r"$\beta=0.99-\lambda$")
# ax.semilogx(beta_fine,gt,ls='--',label="Theorertical",c='g',linewidth=0.8)
ax.plot(lbda_fine,[grad1Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial$Theoretical",c='r',linewidth=0.8)
ax.plot(lbda_fine,[grad2Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial^2$Theoretical",c='cyan',linewidth=0.8)
# ax.plot(lbda_fine,[grad3Theory(beta,i) for i in lbda_fine],ls='--',label="Theoretical3",c='cyan',linewidth=0.8)
# ax.set_xlabel(r"$\beta$")
# ax.set_ylabel("(1/d)logZ")
ax.axvline(0.5,0,1,c='k',linewidth=0.8,ls='--')
# plt.text(1.02,-0.5,r"$\beta=1-\lambda$")
ax.legend()
# fig.savefig(path / "logZ.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
# print(evals)