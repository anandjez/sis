from __future__ import annotations
import hashlib
import logging
import os
import pathlib
from pathlib import Path
import pandas as pd
import hydra
import jax
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import orbax.checkpoint
from flax.training import orbax_utils

logging.basicConfig(level=logging.INFO)

from stint_sampler.eval.plotter import plotter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stint_sampler.targets.targets import gmm,funnel,toy_rings
from stint_sampler.eval.eval_frm_ckpt import KDEplots, evalModel,genTraj,evalResults
import numpy as np
import jax.numpy as jnp

import scipy.integrate as integrate
import scipy.special as special

homePath = pathlib.Path.home()

path = homePath / "python/sis/plots"

genTrajPath = homePath / "python/sis/slurm/logs/2024-04-26/14-59-38/interpolant.type.trig2-learn_potential.0-target.gmm"

# evalModelPath = [homePath / "python/sis/slurm/logs/2024-04-29/20-33-37",
#                  homePath / "python/sis/slurm/logs/2024-04-30/13-18-58" ]
# evalModelPath = [homePath / "python/sis/slurm/logs/2024-05-12/11-08-49_plt",
#                  homePath / "python/sis/slurm/logs/2024-05-12/18-14-28_plt" ]

# evalModelPath = [homePath / "python/sis/slurm/logs/2024-05-12/21-50-17"]
evalModelPath = [
                 homePath / "python/sis/slurm/logs/2024-05-12/21-50-17",
                 homePath / "python/sis/slurm/logs/2024-05-18/13-02-11",
                 homePath / "python/sis/slurm/logs/2024-05-19/23-34-17"
                ]
# @hydra.main(version_base=None, config_path=str(configPath), config_name="config")



interpolant = 1
kde_plot = 0
traj_plot = 0
eval_model = 0
eval_results =0
logZ = 0

if interpolant:
    plotObj = plotter([])
    interpolants = [
                    {"intrplnt": "lin_inc", "name": "$g(t)=t$\n $r(t)=1+t/8$"},
                    {"intrplnt": "trig_const", "name": "$g(t)=sin(\pi t/2)$\n $r(t)=1.0$"},
                    {"intrplnt": "half_trig", "name": "$g(t)=sin(\pi t/2)$\n $r(t)=1-t/2$"},
                    {"intrplnt":"lin","name":"$g(t)=t$\n $r(t)=1-t$"},
                    {"intrplnt":"trig","name":"$g(t)=sin(\pi t/2)$\n $r(t)=cos(\pi t/2)$"},
                    {"intrplnt":"enc_sin","name":"$g(t)=sin(\pi t/2)$\n $r(t)=1.0\mathbb{1}_{\{t\leq 0.5\}}+sin(\pi t)\mathbb{1}_{\{t>0.5\}}$"},
                    ]
    fig_intrplnt = plotObj.plotInterpolants(interpolants)#,{"intrplnt":"enc_sin","name":"$g(t)=sin(\pi t/2)$\n $r(t)=1.0\mathbb{1}_{\{t\leq 0.5\}}+sin(\pi t)\mathbb{1}_{\{t>0.5\}}$"}])
    fig_intrplnt.savefig(path / "interpolants2.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

if kde_plot:
    # gmm_dict = {"name":"gmm", "date" : "2024-04-26", "time" : "14-59-38",
    #                 "interpolant":"trig2", "sampler":"fullsis","fn":gmm}

    gmm_dict = {"path":"/Users/ajgeorge/python/sis/slurm/logs/2024-04-29/20-33-37/interpolant.type.trig2-seed.8-target.gmm",
                "name":"GMM", "interpolant":"trig2", "sampler":"fullsis","fn":gmm, "plotLim":(10,10),"bw":1.0}
#"/Users/ajgeorge/python/sis/slurm/logs/2024-04-30/13-18-58/interpolant.type.trig2-seed.7-target.funnel"
    funnel_dict = {"path":"/Users/ajgeorge/python/sis/slurm/logs/2024-04-30/13-18-58/interpolant.type.trig2-seed.7-target.funnel",
                   "name":"Funnel", "interpolant":"trig2", "sampler":"fullsis","fn":funnel, "plotLim":(10,10),"bw":0.3}

    toy_rings_dict = {"path": "/Users/ajgeorge/python/sis/slurm/logs/2024-05-18/09-58-05",
                      "name":"Rings", "interpolant":"trig2", "sampler":"fullsis","fn":toy_rings, "plotLim":(3,3),"bw":0.3}

    fig_dist = KDEplots([gmm_dict, toy_rings_dict])
    fig_dist.savefig(path / "dist_kde_plots.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

if traj_plot:
    samples = genTraj(genTrajPath,path)
    trajData = samples[:, :100, 0]
    trajData_5 = trajData[:,np.nonzero(trajData[-1,:]>=2.5)[0]]
    trajData_0 = trajData[:,np.nonzero((trajData[-1,:]<2.5)*(trajData[-1,:]>-2.5))[0]]
    trajData_m5 = trajData[:,np.nonzero(trajData[-1,:]<=-2.5)[0]]
    data = samples[:, :, 0]
    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 5, 1]},figsize=[14,4])    # sns.lineplot(trajData,ax = ax[1],legend=False,markers=False,dashes=False,palette='r')
    ax[1].plot(trajData_0,c='b',linewidth=0.8)
    ax[1].plot(trajData_5,c='r',linewidth=0.8)
    ax[1].plot(trajData_m5,c='g',linewidth=0.8)
    sns.kdeplot(y=data[0,:],ax = ax[0],fill=True)
    ax[0].invert_xaxis()
    # ax[0].spines[['left', 'top', 'bottom']].set_visible(False)
    ax[0].set_axis_off()
    ax[2].set_axis_off()
    ax[0].set_ylim([-8,8])
    ax[1].set_ylim([-8,8])
    ax[1].set_ylim([-8,8])
    # ax[2].spines[['right', 'top', 'bottom']].set_visible(False)
    sns.kdeplot(y=data[-1,:],ax = ax[2],fill=True)

    ax[1].tick_params(bottom=True, top=False, left=True, right=True,
                      labelbottom=True, labeltop=False, labelleft=True, labelright=True)
    ax[1].set_yticks([-5,0,5])
    fig.savefig(path / "trajectory.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

def getDWgt(d=10,w=3,delta=2):
    mass1d = integrate.quad(lambda x: np.exp(-(x ** 2 - delta) ** 2), -np.inf, np.inf)[0]
    mean_abs1d = integrate.quad(lambda x: np.abs(x)*np.exp(-(x ** 2 - delta) ** 2), -np.inf, np.inf)[0]/mass1d
    mean_sq1d = integrate.quad(lambda x: (x**2)*np.exp(-(x ** 2 - delta) ** 2), -np.inf, np.inf)[0]/mass1d

    mean_abs1d_gauss = integrate.quad(lambda x: np.abs(x) * np.exp(-0.5*(x ** 2)), -np.inf, np.inf)[0] / np.sqrt(2*np.pi)

    logZ = np.log(mass1d)*w+np.log(np.sqrt(2*np.pi))*(d-w)
    mean_abs = mean_abs1d*w+mean_abs1d_gauss*(d-w)
    mean_sq = mean_sq1d*w+1.0*(d-w)

    return {"logZ":logZ,"Mean Absolute": mean_abs, "Mean Squared": mean_sq}

if eval_model:
    evalFns = {"logZ": lambda x: 0.0, "Mean Absolute": lambda x: jnp.sum(jnp.abs(x)),"Mean Squared": lambda x: jnp.sum(x**2)}
    groundTruth = {"gmm":{"logZ":0.0,"Mean Absolute": 7.1924677, "Mean Squared": 35.255955},
                   "funnel":{"logZ":0.0,"Mean Absolute": 25.345434, "Mean Squared": 704.64325},
                   # "double_well":{"logZ":5.9,"Mean Absolute": 9.55, "Mean Squared": 12.5}
                   "double_well":getDWgt(10,3,2),
                   "double_well20":getDWgt(20,5,3)
                   }
    evals = evalModel(evalModelPath, path,evalFns)
    evals = pd.concat([evals], ignore_index=True)
    evals["groundTruth"] = evals.apply(lambda row: groundTruth[row.target][row.Function],axis=1)
    print(evals.head())

    plt_targets = ["gmm","funnel", "double_well"]
    target_labels = {"gmm": "GMM", "funnel": "Funnel", "double_well": "DW(d=10,w=3,$\delta=2$)",
                     "double_well20": "DW(d=20,w=5,$\delta=3$)", "double_well50": "DW(d=50,w=5,$\delta=2$)"}
    plt_fns = ["logZ"]
    plt_interpolants = ["trig_const", "trig"]
    interpolant_labels = {"trig_const": "Half interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=1.0$",
                          "enc_sin": "Full interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=1.0\mathbb{1}_{\{t\leq 0.5\}}+sin(\pi t)\mathbb{1}_{\{t>0.5\}}$",
                          "trig": "Full interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=cos(\pi t/2)$"}
    evals_logZ = evals.loc[evals["interpolant.type"].isin(plt_interpolants)]
    evals_logZ = evals_logZ.loc[evals_logZ["target"].isin(plt_targets)]
    evals_logZ = evals_logZ.loc[evals_logZ["Function"].isin(plt_fns)]
    evals_logZ["target"] = evals_logZ["target"].replace(target_labels)
    # evals_logZ = evals_logZ.loc[evals["Function"]=="logZ"]
    sns.set_theme(font_scale=1.3, style="white")
    fct = sns.relplot(data=evals_logZ, x="Steps", y="Value", hue="interpolant.type", hue_order=plt_interpolants,
                      col="target", col_order=[target_labels[i] for i in plt_targets],
                      kind="line",
                      facet_kws={"sharey":False, "despine": False, "legend_out":False})
    fct.set_titles(col_template="{col_name}")
    # fct.set(ylim=(-10, None))
    ax = fct.axes
    ax[0,0].set_ylim((-5,2))
    ax[0,1].set_ylim((-10,2))
    ax[0,2].set_ylim((-10,8))
    fct.set_ylabels("logZ")
    # g.set( xticks=[10, 30, 50], yticks=[2, 6, 10])
    fct._legend.set_title("Interpolant")
    # replace labels
    # new_labels = ["Half interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=1.0$", "Full interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=cos(\pi t/2)$"]
    new_labels = [interpolant_labels[i] for i in plt_interpolants]
    for t, l in zip(fct._legend.texts, new_labels):
        t.set_text(l)
    sns.move_legend(fct, "center left", bbox_to_anchor=(0.98, 0.5),title=None, frameon=False)
    fct.map_dataframe(lambda data, **kws: plt.gca().axhline(data["groundTruth"].mean(), 0, 1, ls='--', c='g'))

    fct.savefig(path / "evals_logZ.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

    if 1:
        plt_targets = ["gmm","double_well","double_well20"]
        plt_fns = ["Mean Absolute","Mean Squared"]
        fns_labels = {"Mean Absolute":"E$|X|_1$","Mean Squared":"E$\|X\|^2$"}
        plt_interpolants = ["trig_const","enc_sin","trig"]
        # evals_est = evals.loc[evals["target"] != "funnel"]
        # evals_est = evals_est.loc[evals["Function"] != "logZ"]
        evals_est = evals.loc[evals["target"].isin(plt_targets)]
        evals_est["target"] = evals_est["target"].replace(target_labels)
        evals_est = evals_est.loc[evals["Function"].isin(plt_fns)]
        evals_est["Function"] = evals_est["Function"].replace(fns_labels)
        # sns.set_style("white")
        sns.set_theme(font_scale=1.3,style="white")
        fct = sns.relplot(data=evals_est, x="Steps", y="Value", hue="interpolant.type", hue_order=plt_interpolants,
                          row="Function", row_order=[fns_labels[i] for i in plt_fns],
                          col="target",col_order=[target_labels[i] for i in plt_targets], kind="line",
                    facet_kws={"sharey":False,"despine":False, "legend_out":False,"margin_titles":True})
        fct.set_titles(col_template="{col_name}", row_template="{row_name}")
        # g.set( xticks=[10, 30, 50], yticks=[2, 6, 10])
        fct._legend.set_title("Interpolant")
        # replace labels

        new_labels = [interpolant_labels[i] for i in plt_interpolants]
        for t, l in zip(fct._legend.texts, new_labels):
            t.set_text(l)
        # sns.move_legend(fct, "center left", bbox_to_anchor=(0.98, 0.5))
        sns.move_legend(
            fct, "lower center",
            bbox_to_anchor=(.5, -0.08), ncol=3, title=None, frameon=False,
        )
        fct.map_dataframe(lambda data,**kws:plt.gca().axhline(data["groundTruth"].mean(),0,1,ls='--',c='g'))

        fct.savefig(path / "evals.png", bbox_inches='tight', pad_inches=0.1, dpi=300)


if eval_results:
    evalFns = {"Mean Absolute": lambda x: jnp.sum(jnp.abs(x)),"Mean Squared": lambda x: jnp.sum(x**2)}
    evals = evalResults(evalModelPath, path,evalFns)
    evals["Std"] = evals["Value"]
    summary = evals.groupby(["target","Function","interpolant.type"],as_index=False).agg({"Value":["mean","std"]})
    print(summary)

if logZ:
    lbda = [0.1,0.5,0.95]
    gt = [0.05,0.35,1.5]
    sp_beta0 = [0.045,0.342,1.36]
    sp = [0.143,0.44,1.43]
    fig,ax = plt.subplots(figsize=[6,4])
    ax.plot(lbda,sp_beta0,label=r"$\beta=0.0$")
    ax.plot(lbda,sp,label=r"$\beta=0.99-\lambda$")
    ax.plot(lbda,gt,ls='--',label="Statphys prediction:$-0.5log(1-\lambda)$")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("(1/d)logZ")
    ax.legend()
    fig.savefig(path / "logZ.png", bbox_inches='tight', pad_inches=0.1, dpi=300)


