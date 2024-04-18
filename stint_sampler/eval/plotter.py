import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from omegaconf import DictConfig,OmegaConf
import jax.numpy as jnp

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
