"""`@hydra.main` entry-point shim for FPCA training.

Three-line delegation to ``baselines.fpca_core.train``, which can
also be imported directly and called with an already-composed
config.
``import conf`` is load-bearing — it triggers
``conf/_resolvers.py``'s ``OmegaConf.register_new_resolver`` calls
before ``@hydra.main`` parses the composition.
"""

import hydra
from omegaconf import DictConfig

import conf  # noqa: F401  -- register OmegaConf resolvers
from baselines import fpca_core


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    fpca_core.train(cfg)


if __name__ == "__main__":
    main()
