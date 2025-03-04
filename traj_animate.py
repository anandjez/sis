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
from stint_sampler.targets.targets import gmm,funnel,toy_rings,mog,mos
from stint_sampler.eval.eval_frm_ckpt import KDEplots, evalModel,genTraj,evalResults,evalResults_logZ
from stint_sampler.eval.ipm import OT
import numpy as np
import jax.numpy as jnp
from jax import grad
import matplotlib.animation as animation
from matplotlib.lines import Line2D
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

evalModelPathHD = [
                 homePath / "python/sis/slurm/logs/2024-08-04/18-33-44_hd",
                 homePath / "python/sis/slurm/logs/2024-08-04/22-08-00_hd",
                ]

trainPhasePath = [homePath / "python/sis/slurm/logs/2024-08-03/10-11-19",
                homePath / "python/sis/slurm/logs/2024-08-04/22-08-00_hd/score_model.width.128-seed.4-target.d.100-target.mog-velocity_model.width.128"]
# @hydra.main(version_base=None, config_path=str(configPath), config_name="config")



interpolant = 0
kde_plot = 0
traj_plot = 1
interpolant_plot=0
eval_model = 0
eval_results = 0
logZ = 0
trainPhase = 0
eval_logZ_hd = 0
ablations = 0
ipm =0
emc = 0

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
    t = np.linspace(0,1, trajData.shape[0])
    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 5, 1]},figsize=[14,4])    # sns.lineplot(trajData,ax = ax[1],legend=False,markers=False,dashes=False,palette='r')

    lb = ax[1].plot(t[0],trajData_0[:1,:], c='b', linewidth=0.8)
    lr = ax[1].plot(t[0],trajData_5[:1,:], c='r', linewidth=0.8)
    lg = ax[1].plot(t[0],trajData_m5[:1,:], c='g', linewidth=0.8)
    def update(frame):
        # for each frame, update the data stored on each artist.
        x = t[:frame]
        yb = trajData_0[:frame,:]
        yr = trajData_5[:frame,:]
        yg = trajData_m5[:frame,:]
        # update the line plot:
        for i,line in enumerate(lb):
            line.set_xdata(x)
            line.set_ydata(yb[:,i])
        for i, line in enumerate(lr):
            line.set_xdata(x)
            line.set_ydata(yr[:, i])
        for i, line in enumerate(lg):
            line.set_xdata(x)
            line.set_ydata(yg[:, i])

        #return (lb,lr,lg)

    # ax[1].plot(trajData_0,c='b',linewidth=0.8)
    # ax[1].plot(trajData_5,c='r',linewidth=0.8)
    # ax[1].plot(trajData_m5,c='g',linewidth=0.8)
    sns.kdeplot(y=data[0,:],ax = ax[0],fill=True)
    ax[0].invert_xaxis()
    # ax[0].spines[['left', 'top', 'bottom']].set_visible(False)
    ax[0].set_axis_off()
    ax[2].set_axis_off()
    ax[0].set_ylim([-8,8])
    ax[1].set_ylim([-8,8])
    ax[1].set_xlim([0,1])
    ax[2].set_ylim([-8,8])
    # ax[2].spines[['right', 'top', 'bottom']].set_visible(False)
    sns.kdeplot(y=data[-1,:],ax = ax[2],fill=True)

    ax[1].tick_params(bottom=True, top=False, left=True, right=True,
                      labelbottom=True, labeltop=False, labelleft=True, labelright=True)
    ax[1].set_yticks([-5,0,5])
    ani = animation.FuncAnimation(fig=fig, func=update, frames=trajData.shape[0], interval=30,repeat=False)
    ani.save(filename=path / "traj.gif", writer="pillow")
    # ani.save(filename=path / "traj.mp4", writer="ffmpeg")
    plt.show()
    # fig.savefig(path / "trajectory.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

if interpolant_plot:
    samples = genTraj(genTrajPath,path)
    trajData = samples[:, :100, 0]
    data = samples[:, :, 0]
    t = np.linspace(0,1, trajData.shape[0])
    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]},figsize=[12,4])    # sns.lineplot(trajData,ax = ax[1],legend=False,markers=False,dashes=False,palette='r')



    sns.kdeplot(x=data[0,:],ax = ax[0],fill=True)
    # ax[0].invert_xaxis()
    # ax[0].spines[['left', 'top', 'bottom']].set_visible(False)
    # ax[0].set_axis_off()
    # ax[1].set_axis_off()
    # ax[2].set_axis_off()
    # ax[0].set_xlim([-8,8])
    # ax[1].set_xlim([-8,8])
    # ax[2].set_xlim([-8,8])
    # ax[2].spines[['right', 'top', 'bottom']].set_visible(False)
    sns.kdeplot(x=data[int(trajData.shape[0]/2),:],ax = ax[1],fill=True)
    sns.kdeplot(x=data[-1,:],ax = ax[2],fill=True)

    for i in range(3):
        ax[i].set_xlim([-8, 8])
        ax[i].set_frame_on(False)
        ax[i].get_xaxis().tick_bottom()
        ax[i].axes.get_yaxis().set_visible(False)
        xmin, xmax = ax[i].get_xaxis().get_view_interval()
        ymin, ymax = ax[i].get_yaxis().get_view_interval()
        ax[i].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))

    # ax[0].tick_params(bottom=True, top=False, left=False, right=False,labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    # ax[1].tick_params(bottom=True, top=False, left=False, right=False,labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    # ax[2].tick_params(bottom=True, top=False, left=False, right=False,labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    # ax[0].set_xticks([-5,0,5])
    # ax[1].set_xticks([-5,0,5])
    # ax[2].set_xticks([-5,0,5])
    ax[0].set_xlabel("t=0")
    ax[1].set_xlabel("t=0.5")
    ax[2].set_xlabel("t=1")
    # ani = animation.FuncAnimation(fig=fig, func=update, frames=trajData.shape[0], interval=50,repeat=False)
    # ani.save(filename=path / "traj.gif", writer="pillow")
    # ani.save(filename=path / "traj.mp4", writer="ffmpeg")
    plt.show()
    fig.savefig(path / "intrplntTraj.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
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
    evalFns = {"logZ": lambda x: 0.0, "Mean Absolute": lambda x: jnp.sum(jnp.abs(x)),
               "Mean Squared": lambda x: jnp.sum(x ** 2)}
    groundTruth = {"gmm": {"logZ": 0.0, "Mean Absolute": 7.1924677, "Mean Squared": 35.255955},
                   "funnel": {"logZ": 0.0, "Mean Absolute": 25.345434, "Mean Squared": 704.64325},
                   # "double_well":{"logZ":5.9,"Mean Absolute": 9.55, "Mean Squared": 12.5}
                   "double_well": getDWgt(10, 3, 2),
                   "double_well20": getDWgt(20, 5, 3)
                   }
    if 1:
        evalFns = {"Mean Absolute": lambda x: jnp.sum(jnp.abs(x)),"Mean Squared": lambda x: jnp.sum(x**2)}
        evals = evalResults(evalModelPath, path,evalFns)
    if 0:
        evalsList = []
        for path in evalModelPath:
            runPaths = [x for x in path.iterdir() if x.is_dir()]
            if "eval_results.pkl" in [x.name for x in path.iterdir()]:
                evals = pd.read_pickle(path / "eval_results.pkl")
                evalsList.append(evals)
        evals = pd.concat(evalsList)
        evals = evals.loc[evals["Steps"]==9000]
    evals = pd.concat([evals], ignore_index=True)
    evals["groundTruth"] = evals.apply(lambda row: groundTruth[row.target][row.Function], axis=1)
    evals["Std"] = evals["Value"]
    summary = evals.groupby(["target","Function","interpolant.type"],as_index=False).agg({"Value":["mean","std"],"groundTruth":["mean"]})
    print(summary)

if logZ:
    def theory(beta,lbda):
        if beta<=1-lbda:
            return -0.5*jnp.log(1-lbda)
        else:
            q = (beta-(1-lbda))/(beta**2)
            return  -0.5*jnp.log(beta)+beta*q/2+(beta**2)*(q**2)/4

    if 0:
        grad1Theory = grad(theory,argnums=1)
        grad2Theory = grad(grad1Theory,argnums=1)
        grad3Theory = grad(grad2Theory,argnums=1)
        lbda = [0.0,0.2,0.4,0.5,0.6,0.75,0.95]
        lbda_fine = np.linspace(0,1,50)
        # gt = [-0.5*np.log(1-i) for i in lbda]
        beta = 0.5
        gt = [theory(beta,i) for i in lbda_fine]
        # sp_beta0 = [0.045,0.342,1.36]
        sp = [-0.03,0.08,0.21,0.28,0.36,0.5,0.71]
        fig,ax = plt.subplots(figsize=[6,4])
        ax.plot(lbda,sp,label="FIS",linewidth=0.8,marker='*')
        # ax.plot(lbda,sp,label=r"$\beta=0.99-\lambda$")
        ax.plot(lbda_fine,gt,ls='--',label="Theorertical",c='g',linewidth=0.8)
        ax.plot(lbda_fine,[grad1Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial$Theoretical",c='r',linewidth=0.8)
        ax.plot(lbda_fine,[grad2Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial^2$Theoretical",c='cyan',linewidth=0.8)
        # ax.plot(lbda_fine,[grad3Theory(beta,i) for i in lbda_fine],ls='--',label="Theoretical3",c='cyan',linewidth=0.8)
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("(1/d)logZ")
        ax.axvline(0.5,0,1,c='k',linewidth=0.8,ls='--')
        plt.text(0.52,0.1,r"$\beta=1-\lambda$")
        ax.legend()
        fig.savefig(path / "logZ_gsg.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

    if 1:

        beta = [0.0,0.2,0.4,0.7,0.9,1.0,1.1,1.5,2.0,4.0,7.0,10.0]
        beta_fine = np.linspace(0,10,200)
        # gt = [-0.5*np.log(1-i) for i in lbda]
        lbda = 0.0
        gt = [theory(i,lbda) for i in beta_fine]
        # sp_beta0 = [0.045,0.342,1.36]
        sp = [-0.015,-0.015,-0.02,-0.046,-0.06,-0.062,-0.068,-0.113,-0.175,-0.356,-0.582,-0.868]
        fig,ax = plt.subplots(figsize=[6,4])
        ax.semilogx(beta,sp,label="FIS",linewidth=0.8,marker='*')
        # ax.plot(lbda,sp,label=r"$\beta=0.99-\lambda$")
        ax.semilogx(beta_fine,gt,ls='--',label="Theorertical",c='g',linewidth=0.8)
        # ax.plot(lbda_fine,[grad1Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial$Theoretical",c='r',linewidth=0.8)
        # ax.plot(lbda_fine,[grad2Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial^2$Theoretical",c='cyan',linewidth=0.8)
        # ax.plot(lbda_fine,[grad3Theory(beta,i) for i in lbda_fine],ls='--',label="Theoretical3",c='cyan',linewidth=0.8)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("(1/d)logZ")
        ax.axvline(1,0,1,c='k',linewidth=0.8,ls='--')
        # plt.text(1.02,-0.5,r"$\beta=1-\lambda$")
        ax.legend()
        fig.savefig(path / "logZ.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

#For rebuttal

if trainPhase:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=[12, 4])
    for i in range(len(trainPhasePath)):
        ckptPath = trainPhasePath[i] / "ckpt" / "results"
        results = orbax_checkpointer.restore(ckptPath)
        loss = results["Metrics"]["loss"]
        trainSteps = list(range(0,10000,10))

        ax[i].plot(trainSteps, loss, label="loss", linewidth=0.8)
        # ax.plot(lbda,sp,label=r"$\beta=0.99-\lambda$")
        # ax.plot(lbda_fine,[grad1Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial$Theoretical",c='r',linewidth=0.8)
        # ax.plot(lbda_fine,[grad2Theory(beta,i) for i in lbda_fine],ls='--',label="$\partial^2$Theoretical",c='cyan',linewidth=0.8)
        # ax.plot(lbda_fine,[grad3Theory(beta,i) for i in lbda_fine],ls='--',label="Theoretical3",c='cyan',linewidth=0.8)
        ax[i].set_ylabel("Loss")
        ax[i].set_xlabel("Train steps")

        # ax.axvline(1, 0, 1, c='k', linewidth=0.8, ls='--')
        # plt.text(1.02,-0.5,r"$\beta=1-\lambda$")
        # ax.legend()
    ax[0].set_ylim((0, 1))
    ax[1].set_ylim((0, 10))
    fig.savefig(path / "trainPhase.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

def findNum(st,start,endStr):
    stop = start+1
    while st[stop] != endStr:
        stop += 1
    return stop
def extractFromLog(pathList):
    data = []
    data_local = {}
    for path in pathList:
        with open(path, "r") as file:
            for line in file:
                if line[:4] == "Step":
                    start = 5
                    stop = findNum(line,start,":")
                    step = int(line[start:stop])
                    if step%2000==0:
                        data_local["Steps"] = [step]
                        start = line.find("lnZ")+4
                        stop = findNum(line, start, ";")
                        data_local["Value"] = [float(line[start:stop])]
                        data.append(pd.DataFrame(data_local))
                elif line.find("seed") != -1:
                    data_local["seed"] = [line[line.find("seed")+5]]
                    if line.find("gaussian_mixture10") != -1:
                        data_local["target"] = ["MoG"]
                    else:
                        data_local["target"] = ["MoS"]
                    data_local["target.d"] = [int(line[line.find("target.dim") + 11:-1])]
    return pd.concat(data, ignore_index=True)


if eval_logZ_hd:
    evalFns = {"logZ": lambda x: 0.0}
    groundTruth = {"mog": {"logZ": 0.0,"Mean Absolute": 0.0, "Mean Squared": 0.0}, "mos": {"logZ": 0.0,"Mean Absolute": 0.0, "Mean Squared": 0.0}
                   }
    if 1:
        evalFns = {}
        evals = evalResults_logZ(evalModelPathHD, path)
    if 0:
        evalsList = []
        for path in evalModelPath:
            runPaths = [x for x in path.iterdir() if x.is_dir()]
            if "eval_results.pkl" in [x.name for x in path.iterdir()]:
                evals = pd.read_pickle(path / "eval_results.pkl")
                evalsList.append(evals)
        evals = pd.concat(evalsList)
        evals = evals.loc[evals["Steps"]==9000]
    evals = pd.concat([evals], ignore_index=True)
    evals["groundTruth"] = evals.apply(lambda row: groundTruth[row.target][row.Function], axis=1)
    print(evals.head())

    plt_targets = ["mos", "mog"]
    target_labels = {"mos":"MoS","mog":"MoG","gmm": "GMM", "funnel": "Funnel", "double_well": "DW(d=10,w=3,$\delta=2$)",
                     "double_well20": "DW(d=20,w=5,$\delta=3$)", "double_well50": "DW(d=50,w=5,$\delta=2$)"}
    plt_fns = ["logZ"]
    # plt_interpolants = ["trig_const", "trig"]
    interpolant_labels = {"trig_const": "Half interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=1.0$",
                          "enc_sin": "Full interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=1.0\mathbb{1}_{\{t\leq 0.5\}}+sin(\pi t)\mathbb{1}_{\{t>0.5\}}$",
                          "trig": "Full interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=cos(\pi t/2)$"}
    # evals_logZ = evals.loc[evals["interpolant.type"].isin(plt_interpolants)]
    evals_logZ = evals.loc[evals["target"].isin(plt_targets)]
    # evals_logZ = evals_logZ.loc[evals_logZ["target.d"].isin(["10","100","200"])]
    evals_logZ = evals_logZ.loc[evals_logZ["Function"].isin(plt_fns)]
    evals_logZ["target"] = evals_logZ["target"].replace(target_labels)
    evals_logZ["target.d"] = evals_logZ["target.d"].apply(lambda x:int(x))
    evals_logZ = evals_logZ.sort_values(by=['target', "target.d"])
    evals_logZ["Value"] = evals_logZ["Value"]/evals_logZ["target.d"]
    evals_logZ = evals_logZ.drop(columns=["score_model.width","velocity_model.width","Function"])
    # evals_logZ = evals_logZ.loc[evals["Function"]=="logZ"]
    evals_logZ["method"] = 'FIS'

    logPath = [homePath / "python/sis/plots/slurm-5045_s1.out",
               homePath / "python/sis/plots/slurm-5046_s23.out",
               ]
    logData = extractFromLog(logPath)
    logData["method"] = 'GBS'
    logData["groundTruth"] = 0.0
    logData["Value"] = logData["Value"] / logData["target.d"]
    data = pd.concat([evals_logZ,logData],ignore_index=True)
    data_final = data.loc[data["Steps"]==8000]
    data_final["Std"] = evals["Value"]
    data_final["Value"] = data_final["Value"] * data_final["target.d"]
    summary = data_final.groupby(["method", "target","target.d" ], as_index=False).agg(
        {"Value": ["mean", "std"], "groundTruth": ["mean"]})
    print(summary)

    sns.set_theme(font_scale=1.3, style="white")
    fct = sns.relplot(data=data, x="Steps", y="Value",
                      col="target.d", #col_order=[target_labels[i] for i in plt_targets],
                      row="target",
                      hue = "method",
                      kind="line",
                      facet_kws={"sharey": False, "despine": False, "legend_out": False})
    fct.set_titles(col_template="Dimension = {col_name}")
    # fct.set(ylim=(-10, None))
    ax = fct.axes
    # ax[0, 0].set_ylim((-5, 2))
    # ax[0, 1].set_ylim((-10, 2))
    # ax[0, 2].set_ylim((-10, 8))
    fct.set_ylabels("(1/d)logZ")
    # g.set( xticks=[10, 30, 50], yticks=[2, 6, 10])
    # fct._legend.set_title("Interpolant")
    # replace labels
    # new_labels = ["Half interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=1.0$", "Full interpolant:\n $g(t)=sin(\pi t/2)$\n $r(t)=cos(\pi t/2)$"]
    # new_labels = [interpolant_labels[i] for i in plt_interpolants]
    # for t, l in zip(fct._legend.texts, new_labels):
    #     t.set_text(l)
    sns.move_legend(fct, "center left", bbox_to_anchor=(0.98, 0.5), title=None, frameon=False)
    fct.map_dataframe(lambda data, **kws: plt.gca().axhline(data["groundTruth"].mean(), 0, 1, ls='--', c='g'))

    fct.savefig(path / "evals_logZ_HD.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

if ablations:
    var = 'sde_steps'
    if var=="delta":
        evalModelPath = [homePath / "python/sis/slurm/logs/2024-08-04/00-16-21_ablation",
                 ]
    elif var == "lambda":
        evalModelPath = [homePath / "python/sis/slurm/logs/2024-08-04/06-00-50_ablation",
                         homePath / "python/sis/slurm/logs/2024-08-05/06-23-39_ablations",
                         ]
    elif var == "sde_steps":
        evalModelPath = [homePath / "python/sis/slurm/logs/2024-08-05/08-35-44_ablations",
                         ]
    evalFns = {"logZ": lambda x: 0.0, "Mean Absolute": lambda x: jnp.sum(jnp.abs(x)),
               "Mean Squared": lambda x: jnp.sum(x ** 2)}
    groundTruth = {"gmm": {"logZ": 0.0, "Mean Absolute": 7.1924677, "Mean Squared": 35.255955},
                   "funnel": {"logZ": 0.0, "Mean Absolute": 25.345434, "Mean Squared": 704.64325},
                   # "double_well":{"logZ":5.9,"Mean Absolute": 9.55, "Mean Squared": 12.5}
                   "double_well": getDWgt(10, 3, 2),
                   "double_well20": getDWgt(20, 5, 3)
                   }
    if 1:
        evalFns = {"Mean Absolute": lambda x: jnp.sum(jnp.abs(x)), "Mean Squared": lambda x: jnp.sum(x ** 2)}
        evals = evalResults(evalModelPath, path, evalFns)
    if 0:
        evalsList = []
        for path in evalModelPath:
            runPaths = [x for x in path.iterdir() if x.is_dir()]
            if "eval_results.pkl" in [x.name for x in path.iterdir()]:
                evals = pd.read_pickle(path / "eval_results.pkl")
                evalsList.append(evals)
        evals = pd.concat(evalsList)
        evals = evals.loc[evals["Steps"] == 9000]
    evals = pd.concat([evals], ignore_index=True)
    evals = evals.drop(columns=["target","train.epochs"])
    evals["groundTruth"] = evals.apply(lambda row: groundTruth["gmm"]["logZ"], axis=1)
    evals["Std"] = evals["Value"]
    summary = evals.groupby(["train.NtTrain","Function",], as_index=False).agg(
        {"Value": ["mean", "std"], "groundTruth": ["mean"]})
    print(summary)

if ipm:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = []
    # evalPathFIS = [homePath / "python/sis/slurm/logs/2024-08-04/18-33-44_hd",
    #                homePath / "python/sis/slurm/logs/2024-08-04/22-08-00_hd",
    #                 ]
    evalPathFIS = [homePath / "python/sis/slurm/logs/2024-08-07/07-40-13",
                   ]
    if 1:
        for path in evalPathFIS:
            runPaths = [x for x in path.iterdir() if x.is_dir()]
            for runPath in runPaths:
                evals = {"method":["FIS"]}
                evals["seed"] = [int(runPath.name[runPath.name.find("seed")+5])]
                if runPath.name.find(".10-") != -1:
                    evals["target.d"] = [10]
                elif runPath.name.find(".100-") != -1:
                    evals["target.d"] = [100]
                else:
                    evals["target.d"] = [200]
                if runPath.name.find("mos") != -1:
                    evals["target"] = ["MoS"]
                    _, gt_sampler = mos(d=evals["target.d"][0], k=10)
                else:
                    evals["target"] = ["MoG"]
                    _, gt_sampler = mog(d=evals["target.d"][0], k=10)

                ckptPath = runPath / "ckpt" / "results"
                results = orbax_checkpointer.restore(ckptPath)
                model_samples = results["Samples"][:2000]
                gt_samples = gt_sampler(2000)
                ot_obj = OT(gt_samples)
                cost = ot_obj.compute_OT(model_samples)
                evals["Value"] = [float(cost)]
                print(evals)
                data.append(pd.DataFrame(evals))

        evalPathGBS = homePath / "python/sis/plots"
        filePath = evalPathGBS / "model_samples"
        runPaths = [x for x in filePath.iterdir()]
        for runPath in runPaths:
            evals = {"method": ["GBS"]}
            evals["seed"] = [int(runPath.name[runPath.name.find("seed") + 4])]
            if runPath.name.find("10D") != -1:
                evals["target.d"] = [10]
            elif runPath.name.find("100D") != -1:
                evals["target.d"] = [100]
            else:
                evals["target.d"] = [200]
            if runPath.name.find("student") != -1:
                evals["target"] = ["MoS"]
            else:
                evals["target"] = ["MoG"]
            gt_samples = np.load(evalPathGBS / "gt_samples" / runPath.name)
            model_samples = np.load(filePath / runPath.name)
            ot_obj = OT(gt_samples)
            cost = ot_obj.compute_OT(model_samples)
            evals["Value"] = [float(cost)]
            print(evals)
            data.append(pd.DataFrame(evals))

        data_agg = pd.concat(data)
        # data_agg.to_pickle(evalPathGBS / "wasserstein_distance.pkl")
    else:
        data_agg = pd.read_pickle(homePath / "python/sis/plots/wasserstein_distance.pkl")
        data_agg = data_agg[~data_agg.isin([np.nan, np.inf, -np.inf]).any(1)]
    data_agg = data_agg.sort_values(by=['target', "target.d"])
    data_agg["Std"] = data_agg["Value"]
    summary = data_agg.groupby(["method", "target", "target.d"], as_index=False).agg(
        {"Value": ["mean", "std"]})
    print(summary)

if emc:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = []
    evalPathFIS = [homePath / "python/sis/slurm/logs/2024-08-04/18-33-44_hd",
                   homePath / "python/sis/slurm/logs/2024-08-04/22-08-00_hd",
                    ]
    # evalPathFIS = [homePath / "python/sis/slurm/logs/2024-08-07/07-40-13",
    #                ]
    if 1:
        for path in evalPathFIS:
            runPaths = [x for x in path.iterdir() if x.is_dir()]
            for runPath in runPaths:
                evals = {"method":["FIS"]}
                evals["seed"] = [int(runPath.name[runPath.name.find("seed")+5])]
                if runPath.name.find(".10-") != -1:
                    evals["target.d"] = [10]
                elif runPath.name.find(".100-") != -1:
                    evals["target.d"] = [100]
                else:
                    evals["target.d"] = [200]
                d = evals["target.d"][0]
                if runPath.name.find("mos") != -1:
                    evals["target"] = ["MoS"]
                    _, gt_sampler = mos(d=evals["target.d"][0], k=10)
                    k = 10

                    var = 1.0
                    nu = 2.0
                    mean = float(k)
                    mean_vec = (jax.random.uniform(jax.random.PRNGKey(0), (d, k)) - 0.5) * 2 * mean
                    log_density_comp = lambda x: -0.5 * (nu + d) * jnp.log(
                        1 + jnp.sum((jnp.outer(x, jnp.ones((mean_vec.shape[1],))) - mean_vec) ** 2, axis=0) / (
                                    nu * var)) + jax.scipy.special.gammaln(0.5 * (nu + d)) - jax.scipy.special.gammaln(
                        nu / 2) - jnp.log(k) - jnp.log(nu * jnp.pi ** var) * d / 2
                    mode = lambda x: jnp.argmax(log_density_comp(x))
                    mode = jax.vmap(mode)
                else:
                    evals["target"] = ["MoG"]
                    _, gt_sampler = mog(d=evals["target.d"][0], k=10)
                    k=10
                    var = 1.0
                    mean = float(k)
                    # mean_vec = jnp.array([[mean * (i - 1), mean * (j - 1)] for i in range(3) for j in range(3)]).T
                    mean_vec = (jax.random.uniform(jax.random.PRNGKey(0), (d, k)) - 0.5) * 2 * mean
                    log_density_comp = lambda x: -jnp.sum(
                        (jnp.outer(x, jnp.ones((mean_vec.shape[1],))) - mean_vec) ** 2, axis=0) / (2 * var) - jnp.log(
                        2 * jnp.pi * var) * d / 2 - jnp.log(k)
                    mode = lambda x: jnp.argmax(log_density_comp(x))
                    mode = jax.vmap(mode)

                ckptPath = runPath / "ckpt" / "results"
                results = orbax_checkpointer.restore(ckptPath)
                model_samples = results["Samples"][:2000]
                gt_samples = gt_sampler(2000)
                unique_modes,counts = jnp.unique(mode(model_samples),return_counts=True)
                unique_modes_gt,counts_gt = jnp.unique(mode(gt_samples),return_counts=True)
                # ot_obj = OT(gt_samples)
                # cost = ot_obj.compute_OT(model_samples)
                evals["Value"] = [int(len(unique_modes))]
                evals["Max_prob"] = [float(jnp.max(counts/2000))]
                evals["Min_prob"] = [float(jnp.min(counts/2000))]
                # print(len(unique_modes),evals["Max_prob"][0],evals["Min_prob"][0])
                print(evals)
                data.append(pd.DataFrame(evals))

        evalPathGBS = homePath / "python/sis/plots"
        filePath = evalPathGBS / "model_samples"
        runPaths = [x for x in filePath.iterdir()]
        for runPath in runPaths:
            evals = {"method": ["GBS"]}
            evals["seed"] = [int(runPath.name[runPath.name.find("seed") + 4])]
            if runPath.name.find("10D") != -1:
                evals["target.d"] = [10]
            elif runPath.name.find("100D") != -1:
                evals["target.d"] = [100]
            else:
                evals["target.d"] = [200]
            d = evals["target.d"][0]
            if runPath.name.find("student") != -1:
                evals["target"] = ["MoS"]
                k = 10

                var = 1.0
                nu = 2.0
                mean = float(k)
                mean_vec = jax.random.uniform(jax.random.PRNGKey(0), minval=-k, maxval=k, shape=(d,k))
                log_density_comp = lambda x: -0.5 * (nu + d) * jnp.log(
                    1 + jnp.sum((jnp.outer(x, jnp.ones((mean_vec.shape[1],))) - mean_vec) ** 2, axis=0) / (
                            nu * var)) + jax.scipy.special.gammaln(0.5 * (nu + d)) - jax.scipy.special.gammaln(
                    nu / 2) - jnp.log(k) - jnp.log(nu * jnp.pi ** var) * d / 2
                mode = lambda x: jnp.argmax(log_density_comp(x))
                mode = jax.vmap(mode)
            else:
                evals["target"] = ["MoG"]
                k = 10
                var = 1.0
                mean = float(k)
                # mean_vec = jnp.array([[mean * (i - 1), mean * (j - 1)] for i in range(3) for j in range(3)]).T
                # mean_vec = (jax.random.uniform(jax.random.PRNGKey(0), (d, k)) - 0.5) * 2 * mean
                mean_vec = jax.random.uniform(jax.random.PRNGKey(0), minval=-k, maxval=k, shape=(d,k))
                log_density_comp = lambda x: -jnp.sum(
                    (jnp.outer(x, jnp.ones((mean_vec.shape[1],))) - mean_vec) ** 2, axis=0) / (2 * var) - jnp.log(
                    2 * jnp.pi * var) * d / 2 - jnp.log(k)
                mode = lambda x: jnp.argmax(log_density_comp(x))
                mode = jax.vmap(mode)
            gt_samples = np.load(evalPathGBS / "gt_samples" / runPath.name)
            model_samples = np.load(filePath / runPath.name)
            # ot_obj = OT(gt_samples)
            # cost = ot_obj.compute_OT(model_samples)

            unique_modes, counts = jnp.unique(mode(model_samples), return_counts=True)
            unique_modes_gt, counts_gt = jnp.unique(mode(gt_samples), return_counts=True)
            # ot_obj = OT(gt_samples)
            # cost = ot_obj.compute_OT(model_samples)
            evals["Value"] = [int(len(unique_modes))]
            evals["Max_prob"] = [float(jnp.max(counts / 2000))]
            evals["Min_prob"] = [float(jnp.min(counts / 2000))]
            # print(len(unique_modes), evals["Max_prob"][0], evals["Min_prob"][0])
            print(evals)
            data.append(pd.DataFrame(evals))

        data_agg = pd.concat(data)
        data_agg.to_pickle(evalPathGBS / "emc.pkl")
    else:
        data_agg = pd.read_pickle(homePath / "python/sis/plots/wasserstein_distance.pkl")
        data_agg = data_agg[~data_agg.isin([np.nan, np.inf, -np.inf]).any(1)]
    data_agg = data_agg.sort_values(by=['target', "target.d"])
    # data_agg["Std"] = data_agg["Value"]
    summary = data_agg.groupby(["method", "target", "target.d"], as_index=False).agg(
        {"Value": ["mean", "std"],"Max_prob": ["mean", "std"],"Min_prob": ["mean", "std"]})
    print(summary)
