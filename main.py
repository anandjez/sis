from __future__ import annotations
import hashlib
import logging
import os
from pathlib import Path

import hydra
import jax
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from stint_sampler.eval.plotter import plotter


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    logging.info("---------------------------------------------------------------")
    envs = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONOPTIMIZE"]}
    logging.info("Env:\n%s", yaml.dump(envs))

    devices = jax.devices()
    n_devices = jax.local_device_count()

    logging.info("Devices:\n%s", devices)
    logging.info("Number of devices:\n%s", n_devices)

    hydra_config = HydraConfig.get()
    logging.info("Command line args:\n%s", "\n".join(hydra_config.overrides.task))

    # Setup dir
    OmegaConf.set_struct(cfg, False)
    out_dir = Path(hydra_config.runtime.output_dir).absolute()
    logging.info("Hydra and wandb output path: %s", out_dir)
    if not cfg.get("out_dir"):
        cfg.out_dir = str(out_dir)
    logging.info("Solver output path: %s", cfg.out_dir)

    # Setup wandb
    tags = [t for t in hydra_config.overrides.task if len(t) < 32]
    if "wandb" not in cfg:
        cfg.wandb = OmegaConf.create()
    if not cfg.wandb.get("tags"):
        cfg.wandb.tags = tags

    if not cfg.wandb.get("id"):
        # create id based on log directory for automatic (slurm) resuming
        sha = hashlib.sha256()
        sha.update(str(out_dir).encode())
        cfg.wandb.id = sha.hexdigest()

    if not cfg.wandb.get("name"):
        if hydra_config.mode is hydra.types.RunMode.RUN:
            name = str(out_dir.relative_to(out_dir.parents[1]))
        else:
            name = str(out_dir.parent.relative_to(out_dir.parents[2]))
        cfg.wandb.name = name + "," + ",".join([t.split("=")[-1] for t in tags])
    OmegaConf.set_struct(cfg, True)
    wandb.init(
        dir=out_dir,
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), entity="anandjez"
    )

    print(wandb.config)
    # Resume old wandb run
    # if wandb.run is not None and wandb.run.resumed:
    #     logging.info("Resume wandb run %s", wandb.run.path)
    #     if cfg.get("merge_wandb_resume_cfg"):
    #         cfg = merge_wandb_cfg(cfg)
    #         print(cfg)

    # Log config and overrides
    logging.info("---------------------------------------------------------------")
    logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    try:
        sampler = instantiate(cfg.solver, cfg)
        params = sampler.train()
        sampler.eval()
        wandb.run.summary["error"] = None
        wandb.finish()
        logging.info("Completed ✅")

    except Exception as e:
        logging.critical(e, exc_info=True)
        wandb.run.summary["error"] = str(e)
        wandb.finish(exit_code=1)
        raise

if __name__ =="__main__":
    main()