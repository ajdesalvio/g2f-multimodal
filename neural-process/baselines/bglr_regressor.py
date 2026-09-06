"""
BGLR transductive regressor — Python wrapper around bglr_compute.R.

RKHS parameterization: this Python wrapper passes one raw obs-level
N×N kernel ``K_k`` per ``feature_keys`` block to ``bglr_compute.R``
(no upstream eigen filter, no reconstruction from V·√d). The R script
then eigen-decomposes each kernel (``eigen(K, symmetric=TRUE)``) and
builds the actual ETA term ``list(V=$vectors, d=$values, model='RKHS')``
that BGLR consumes. The optional terminal env block is passed as
``list(X=Z_e, model='BRR')``.

Transductive interface: ``fit_predict_blocks`` consumes raw kernels
spanning train + test rows together because BGLR has no inductive
predict step. The ``is_transductive`` ClassVar lets ``fpca_core``
branch on it without duck-typing through ``hasattr``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "bglr_compute.R")


class BGLRRegressor:
    """Multi-kernel Bayesian RKHS via BGLR's Gibbs sampler.

    Parameters
    ----------
    n_iter : int
        Total Gibbs iterations (default 10000).
    burn_in : int
        Burn-in iterations discarded from the posterior. Must be
        strictly less than ``n_iter``.
    seed : int
        RNG seed forwarded to R via ``set.seed`` BEFORE ETA
        construction. Canonical CRAN BGLR has no ``seed=`` argument
        (would error with ``unused argument``); reproducibility
        relies on ``set.seed`` driving the base R RNG that BGLR
        samples from.
    env_brr_key : str | None
        Block name routed to the dedicated env BRR ETA term (the
        ``Z_e`` block). When set, ``fit_predict_blocks`` REQUIRES the
        caller to pass the env design matrix via the dedicated
        ``env_block`` argument rather than including it in ``K_blocks``.
    """

    is_transductive: ClassVar[bool] = True

    def __init__(
        self,
        n_iter: int = 10000,
        burn_in: int = 1000,
        seed: int = 0,
        env_brr_key: str | None = None,
    ):
        if n_iter <= 0:
            raise ValueError(f"n_iter must be > 0, got {n_iter}")
        if burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {burn_in}")
        if burn_in >= n_iter:
            raise ValueError(
                f"burn_in ({burn_in}) must be < n_iter ({n_iter})"
            )
        self.n_iter = int(n_iter)
        self.burn_in = int(burn_in)
        self.seed = int(seed)
        self.env_brr_key = env_brr_key

    # ------------------------------------------------------------------
    # Inductive predict is intentionally unsupported.
    # ------------------------------------------------------------------

    def predict(self, X):  # noqa: ARG002 — sklearn-compat signature
        raise RuntimeError(
            "BGLRRegressor is transductive — call "
            "fit_predict_blocks() instead of predict()."
        )

    # ------------------------------------------------------------------
    # Main entry point.
    # ------------------------------------------------------------------

    def fit_predict_blocks(
        self,
        K_blocks: dict[str, np.ndarray],
        y_train: np.ndarray,
        n_test: int,
        env_block: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Run BGLR transductively and return (y_pred_train, y_pred_test, meta).

        Parameters
        ----------
        K_blocks : ordered dict of raw obs-level kernels. Each value
            has shape ``(N, N)`` with ``N = len(y_train) + n_test``
            and rows in ``[train..., test...]`` order. Insertion
            order defines the ETA sequence — pass blocks in
            ``feature_keys`` order to mirror the R reference's ordering.
        y_train : ``(n_train,)`` training targets.
        n_test : number of test rows (test y is masked to NA inside R).
        env_block : optional ``(N, n_envs)`` design matrix for the
            terminal env BRR term. Required iff ``self.env_brr_key``
            is set.
        """
        n_train = len(y_train)
        N = n_train + n_test
        block_names = list(K_blocks.keys())
        if not block_names:
            raise ValueError(
                "BGLRRegressor: at least one kernel block is "
                "required (empty K_blocks)."
            )

        for name in block_names:
            K = K_blocks[name]
            if K.ndim != 2 or K.shape[0] != N or K.shape[1] != N:
                raise ValueError(
                    f"BGLRRegressor: kernel block {name!r} must be "
                    f"({N}, {N}); got shape={K.shape}."
                )

        # Env-block invariant: when env_brr_key is set, the caller
        # must route the env design matrix through env_block.
        if self.env_brr_key is not None:
            if env_block is None:
                raise ValueError(
                    f"BGLRRegressor: env_brr_key="
                    f"{self.env_brr_key!r} declared but env_block "
                    f"was not passed."
                )
            if env_block.ndim != 2 or env_block.shape[0] != N:
                raise ValueError(
                    f"BGLRRegressor: env_block must be (N={N}, p); "
                    f"got shape={env_block.shape}."
                )
            if self.env_brr_key in block_names:
                raise ValueError(
                    f"BGLRRegressor: env_brr_key="
                    f"{self.env_brr_key!r} also appears in the "
                    f"kernel block names — caller should keep env "
                    f"separate from K_blocks."
                )
        elif env_block is not None:
            raise ValueError(
                "BGLRRegressor: env_block passed but env_brr_key "
                "is None — set env_brr_key or drop env_block."
            )

        tmpdir = tempfile.mkdtemp(prefix="bglr_")
        logger.info("BGLR work_dir: %s", tmpdir)

        try:
            self._write_inputs(
                tmpdir, K_blocks, y_train, n_test, env_block,
            )
            self._run_rscript(tmpdir)
            yhat = self._read_yhat(tmpdir, N)
            meta = self._read_meta(tmpdir)
            return yhat[:n_train], yhat[n_train:], meta
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _write_inputs(
        self,
        tmpdir: str,
        K_blocks: dict[str, np.ndarray],
        y_train: np.ndarray,
        n_test: int,
        env_block: np.ndarray | None,
    ) -> None:
        os.makedirs(os.path.join(tmpdir, "blocks"), exist_ok=True)
        n_train = len(y_train)
        N = n_train + n_test

        # y_full.csv — test rows are placeholder zeros; R unconditionally
        # masks them to NA via train_mask, so the value never reaches BGLR.
        y_full = np.concatenate([
            np.asarray(y_train, dtype=np.float64),
            np.zeros(n_test, dtype=np.float64),
        ])
        np.savetxt(
            os.path.join(tmpdir, "y_full.csv"),
            y_full, delimiter=",", header="y", comments="",
        )

        is_train = np.concatenate([
            np.ones(n_train, dtype=int),
            np.zeros(n_test, dtype=int),
        ])
        np.savetxt(
            os.path.join(tmpdir, "train_mask.csv"),
            is_train,
            delimiter=",", header="is_train", comments="", fmt="%d",
        )

        block_names = list(K_blocks.keys())
        for name in block_names:
            self._write_kernel_block_bin(tmpdir, name, K_blocks[name])

        if self.env_brr_key is not None:
            X_env = np.ascontiguousarray(env_block, dtype=np.float64)
            self._write_block(tmpdir, self.env_brr_key, X_env)

        manifest = {
            "block_names": block_names,
            "env_brr_key": self.env_brr_key,
            "n_iter": self.n_iter,
            "burn_in": self.burn_in,
            "seed": self.seed,
            "n_train": n_train,
            "n_test": n_test,
        }
        with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _write_block(tmpdir: str, name: str, X: np.ndarray) -> None:
        """Serialize a non-square design matrix (Z_e) as CSV.

        Used only for the env BRR block, which is small and dense.
        """
        p_k = X.shape[1]
        header = ",".join(f"z_{i}" for i in range(p_k))
        np.savetxt(
            os.path.join(tmpdir, "blocks", f"{name}.csv"),
            X, delimiter=",", header=header, comments="",
        )

    @staticmethod
    def _write_kernel_block_bin(
        tmpdir: str, name: str, K: np.ndarray,
    ) -> None:
        """Write a raw obs-level N×N kernel as float64 (row-major).

        The caller is responsible for building each K over the full
        ``[train + test]`` row set (raw, no eigen filter). The R-side
        bridge passes ``list(V=evd$vectors, d=evd$values, model='RKHS')``
        directly to BGLR — matching the R reference pipeline.

        Memory: K is N×N float64 (~800 MB for N≈10k). Written once
        per block; the next block's K can be built after the previous
        one is GC'd.
        """
        K_c = np.ascontiguousarray(K, dtype=np.float64)
        K_c.tofile(os.path.join(tmpdir, "blocks", f"{name}.bin"))

    def _run_rscript(self, tmpdir: str) -> None:
        cmd = [
            "Rscript",
            os.path.normpath(_R_SCRIPT),
            "--work_dir", tmpdir,
        ]
        # Pin BLAS / OMP threads in the parent env so every backend
        # (OpenMP / OpenBLAS / MKL / Accelerate) sees the cap before
        # any thread pool is created in R. Setting these inside R is
        # backend-dependent and unreliable.
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
        )
        if result.stdout:
            logger.info(result.stdout.rstrip())
        if result.stderr:
            logger.warning(result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(
                f"BGLR R subprocess failed (code "
                f"{result.returncode}):\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )

    @staticmethod
    def _read_yhat(tmpdir: str, n_total: int) -> np.ndarray:
        path = os.path.join(tmpdir, "yhat.csv")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"BGLR R subprocess did not produce yhat.csv at {path}"
            )
        yhat = np.loadtxt(path, skiprows=1, delimiter=",")
        yhat = np.atleast_1d(yhat).astype(np.float64)
        if yhat.shape[0] != n_total:
            raise RuntimeError(
                f"yhat.csv has {yhat.shape[0]} rows, expected "
                f"{n_total}."
            )
        return yhat

    @staticmethod
    def _read_meta(tmpdir: str) -> dict:
        path = os.path.join(tmpdir, "bglr_meta.json")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"BGLR R subprocess did not produce bglr_meta.json "
                f"at {path}"
            )
        with open(path) as f:
            return json.load(f)
