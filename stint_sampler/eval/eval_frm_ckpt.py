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

def KDEplots_old(dists,Nsamples=10000):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    fig, ax = plt.subplots(nrows=2,ncols=len(dists),figsize=(3*len(dists), 3*2))
    sns.set_style("ticks")
    for i,dist in enumerate(dists):
        path = Path("/Users/ajgeorge/python/sis/slurm/logs") / dist["date"] / dist["time"]
        subDirs = [x.name for x in path.iterdir() if x.is_dir()]
        if "ckpt" in subDirs:
            print("Checkpoint Found!")
        else:
            for sd in subDirs:
                if sd.find(dist["name"]) != -1:
                    print("Checkpoint Found!")
                    path = path / sd
                else:
                    if sd == subDirs[-1]:
                        print("Checkpoint not found!")
        path = path / "ckpt"
        results = orbax_checkpointer.restore(path / "results")
        params = orbax_checkpointer.restore(path / "model_params")
        _,sampler = dist["fn"]()
        samples_gt = np.array(sampler(Nsamples))
        samples = np.array(results["Samples"])
        sns.kdeplot(x=samples_gt[:, 0], y=samples_gt[:, 1], fill=True, cmap="Blues", ax=ax[0,i])
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True, cmap="Blues", ax=ax[1,i])
        ax[0,i].set_xlim([-10,10])
        ax[0,i].set_ylim([-10,10])
        ax[1,i].set_xlim([-10,10])
        ax[1,i].set_ylim([-10,10])
    return fig

def KDEplots(dists,Nsamples=10000):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    fig, ax = plt.subplots(nrows=2,ncols=len(dists),figsize=(3*len(dists), 3*2))
    sns.set_style("ticks")
    ax[0,0].set_ylabel("Ground Truth")
    ax[1,0].set_ylabel("Sampler")
    for i,dist in enumerate(dists):
        path = Path(dist["path"])
        subDirs = [x.name for x in path.iterdir() if x.is_dir()]
        if "ckpt" in subDirs:
            print("Checkpoint Found!")
            path = path / "ckpt"
            results = orbax_checkpointer.restore(path / "results")
            # params = orbax_checkpointer.restore(path / "model_params")
            _,sampler = dist["fn"]()
            samples_gt = np.array(sampler(Nsamples))
            samples = np.array(results["Samples"])
            sns.kdeplot(x=samples_gt[:, 0], y=samples_gt[:, 1], fill=True, cmap="Blues", ax=ax[0,i],bw_adjust=dist["bw"])
            sns.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True, cmap="Blues", ax=ax[1,i],bw_adjust=dist["bw"])
            for j in range(2):
                ax[j,i].set_xlim([-dist["plotLim"][0],dist["plotLim"][0]])
                ax[j,i].set_ylim([-dist["plotLim"][1],dist["plotLim"][1]])
            ax[1,i].set_xlabel(dist["name"])
        else:
            print("Ckeckpoint not found!")
    return fig


def genTraj(path,outPath):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    configPath = path / ".hydra" / "config.yaml"
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
    params = orbax_checkpointer.restore(path / "ckpt" / "model_params")
    sampler = instantiate(cfg.solver, cfg)
    sampler.params_ema = params
    # params = sampler.train()
    samples,weights = sampler.genTraj()
    logging.info("Completed ✅")
    return samples
def evalModel(pathList,outPath,evalFns,loadModel = True):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    samples_full = []
    weights_full = []
    evalsList = []
    for path in pathList:
        runPaths = [x for x in path.iterdir() if x.is_dir()]
        if "eval_results.pkl" in [x.name for x in path.iterdir()]:
            evals = pd.read_pickle(path / "eval_results.pkl")
            evalsList.append(evals)
        else:
            evals = []
            for runPath in runPaths:
                sweepPars = runPath.name.split("-")
                if loadModel:
                    configPath = runPath / ".hydra" / "config.yaml"
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
                    d = cfg.get("dim")
                for ckptPath in (runPath / "ckpt" / "model_params").iterdir():
                    if ckptPath.name.startswith("ckpt_"):
                        if int(ckptPath.name[5:])%1000 == 0:
                            params = orbax_checkpointer.restore(ckptPath)
                            sampler.params_ema = params
                            # params = sampler.train()
                            samples,weights = sampler.generateSamples()
                            evals_local = {}
                            evals_local["Steps"] = [int(ckptPath.name[5:])]
                            for par in sweepPars:
                                attr = par.split(".")
                                evals_local[".".join(attr[:-1])] = [attr[-1]]
                            for fn in evalFns.keys():
                                if fn == "logZ":
                                    val = [float(sampler.estimate_logZ())]
                                else:
                                    val = [float(jnp.mean(jax.vmap(evalFns[fn])(samples)*weights))]
                                evals_local["Function"] = [fn]
                                evals_local["Value"] = val
                                evals.append(pd.DataFrame(evals_local))

            evals = pd.concat(evals)
            evals.to_pickle(path / "eval_results.pkl")
            evalsList.append(evals)
    logging.info("Completed ✅")
    return pd.concat(evalsList)

def evalResults(pathList,outPath,evalFns):
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
            for par in sweepPars:
                attr = par.split(".")
                if attr[-2].isdigit():
                    evals_local[".".join(attr[:-2])] = [".".join(attr[-2:])]
                else:
                    evals_local[".".join(attr[:-1])] = [attr[-1]]
            evals_local["Function"] = ["logZ"]
            evals_local["Value"] = [float(results["Estimates"]["logZ"][-1])]
            evals.append(pd.DataFrame(evals_local))
            for fn in evalFns.keys():
                evals_local["Function"] = [fn]
                fn_evals = jax.vmap(evalFns[fn])(results["Samples"])*results["Weights"]
                mean = jnp.mean(fn_evals)
                std = jnp.sqrt(jnp.mean((fn_evals-mean)**2))
                evals_local["Value"] = [float(mean)]
                # evals_local["Std"] = [float(std)]
                evals.append(pd.DataFrame(evals_local))

        evals = pd.concat(evals)
        # evals.to_pickle(path / "eval_results.pkl")
        evalsList.append(evals)
    logging.info("Completed ✅")
    evals_pd = pd.concat(evalsList,ignore_index=True)
    # evals_pd.to_pickle(outPath / "eval_results_final.pkl")
    return evals_pd

def evalResults_logZ(pathList,outPath):
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
            for par in sweepPars:
                attr = par.split(".")
                if attr[-2].isdigit():
                    evals_local[".".join(attr[:-2])] = [".".join(attr[-2:])]
                else:
                    evals_local[".".join(attr[:-1])] = [attr[-1]]
            evals_local["score_model.width"] = [128]
            evals_local["velocity_model.width"] = [128]
            for stp in range(len(results["Estimates"]["logZ"])):
                evals_local["Steps"] = [2000*stp]
                # for fns in results["Estimates"].keys():
                evals_local["Function"] = ["logZ"]
                evals_local["Value"] = [float(results["Estimates"]["logZ"][stp])]
                evals.append(pd.DataFrame(evals_local))

        evals = pd.concat(evals)
        # evals.to_pickle(path / "eval_results.pkl")
        evalsList.append(evals)
    logging.info("Completed ✅")
    evals_pd = pd.concat(evalsList,ignore_index=True)
    # evals_pd.to_pickle(outPath / "eval_results_final.pkl")
    return evals_pd


