"""`@hydra.main` entry-point shim for evaluation.

Three-line delegation to ``utils.experiment.run.run_eval``; same
split-and-import pattern as ``train.py`` — ``run_eval`` can be
imported directly and called with an already-composed config. ``import conf`` is load-bearing — triggers
``conf/_resolvers.py``'s ``OmegaConf.register_new_resolver`` calls
before ``@hydra.main`` parses the composition.
"""
import hydra
import torch
from omegaconf import DictConfig

import conf  # noqa: F401  -- register OmegaConf resolvers
from utils.experiment.run import run_eval


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
