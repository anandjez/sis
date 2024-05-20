import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from omegaconf import DictConfig,OmegaConf
import jax.numpy as jnp
import math
from stint_sampler.stint.linearInterpolants import linear



@dataclass
class plotter():
    data: Any

    def make_hist_plot2d(self,dims):
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.tight_layout()
        vals, xedges, yedges = np.histogram2d(self.data[:, dims[0]],self.data[:, dims[1]], 40)
        ax.imshow(vals, interpolation='none', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        return fig
        # ax.grid(True)

    def plotEstimates(self,estimates:dict):
        est_fns = estimates.keys()
        nrows = int(math.sqrt(len(est_fns)))
        ncols = int(len(est_fns)/nrows+1)
        fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10, 10))
        plt.tight_layout()
        for i,fn in enumerate(est_fns):
            ax[i].plot(estimates[fn])
            ax[i].set_title(fn)
        return fig

    def plotInterpolants(self,types):
        x = np.linspace(0,1,1000)
        nrows = 2
        ncols = len(types)//2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3*nrows))
        fig.subplots_adjust(hspace=0.35)
        # plt.tight_layout()
        colors = ['b','r']
        labels = ['g','r']
        for i,intrplnt in enumerate(types):
            g,r = linear(intrplnt["intrplnt"])
            for j,fn in enumerate([g,r]):
                ax[i//ncols,i%ncols].plot(x,fn(x)*jnp.ones_like(x),c=colors[j],label=labels[j],linewidth=0.8)
                ax[i//ncols,i%ncols].set_title(intrplnt["name"])
                ax[i//ncols,i%ncols].set_xlim([0,1])
                ax[i//ncols,i%ncols].set_xticks([0,0.5,1])
                ax[i//ncols,i%ncols].set_ylim([0,1.2])
                ax[i//ncols,i%ncols].set_yticks([0, 0.5, 1])
                ax[i//ncols,i%ncols].legend(loc='lower center')
        return fig


