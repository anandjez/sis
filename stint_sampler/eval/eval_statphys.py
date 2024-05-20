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

evalModelPath = [homePath / "python/sis/slurm/logs/2024-05-15/23-34-42"]


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
            q = jnp.mean(jnp.mean(results["Samples"],axis=0)**2)
            for par in sweepPars:
                attr = par.split(".")
                if attr[-2].isnumeric():
                    evals_local[".".join(attr[:-2])] = [float(".".join(attr[-2:]))]
                else:
                    evals_local[".".join(attr[:-1])] = [attr[-1]]
            # evals_local["Function"] = ["logZ"]
            evals_local["logZ"] = [float(results["Estimates"]["logZ"][-1])]
            # evals.append(pd.DataFrame(evals_local))
            # evals_local["Function"] = ["q"]
            evals_local["q"] = [float(q)]
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
    evals_pd.to_pickle(outPath / "eval_results_final.pkl")
    return evals_pd

evals = evalResults(evalModelPath,path)
print(evals)