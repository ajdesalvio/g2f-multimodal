"""`@hydra.main` entry-point shim for FPCA prediction.

Three-line delegation to ``baselines.fpca_core.predict``. Predict
mode is currently a stub — see the "Inference" note in
``baselines/README.md``.
"""

import hydra
from omegaconf import DictConfig

import conf  # noqa: F401  -- register OmegaConf resolvers
from baselines import fpca_core


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    fpca_core.predict(cfg)


if __name__ == "__main__":
    main()
