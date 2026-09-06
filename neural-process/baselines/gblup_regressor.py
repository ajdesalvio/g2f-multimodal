"""GBLUP transductive regressor — frequentist twin of BGLRRegressor.

Same multi-kernel mixed model as the Bayesian RKHS path
(``baselines/bglr_regressor.py``), but the variance components are
estimated by REML (via ``sommer::mmer`` inside ``gblup_compute.R``)
and predictions are the BLUP / kriging point estimate rather than a
Gibbs posterior mean. Conditional on the variance components, BLUP and
the Bayesian posterior mean coincide — so this is the literal
frequentist counterpart, and a run is a one-token config swap
(``regressor.name: bglr → gblup``) over an identical kernel stack.

Env handling: the Bayesian path routes the env one-hot ``Z_e`` to a
dedicated ``BRR`` ETA term. A ridge on a one-hot is exactly an RKHS
random effect with the linear kernel ``K_env = Z_e Z_eᵀ``, so this
wrapper folds the env block into ``K_blocks`` as that kernel. The
public ``fit_predict_blocks`` signature is therefore byte-identical to
``BGLRRegressor`` — ``fpca_core`` calls both the same way — and
``gblup_compute.R`` sees a flat list of N×N kernels with no special
env code path.

Transductive interface: ``fit_predict_blocks`` consumes raw kernels
spanning train + test rows together; the ``K_k[test, train]``
cross-blocks carry the coupling that yields test predictions. The
``is_transductive`` ClassVar lets ``fpca_core`` branch on it without
duck-typing through ``hasattr``.
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

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "gblup_compute.R")


class GBLUPRegressor:
    """Multi-kernel frequentist GBLUP via REML (``sommer``) + BLUP.

    Parameters
    ----------
    reml_max_iter : int
        Maximum REML (AI) iterations forwarded to ``sommer::mmer``'s
        ``iters`` argument (default 40). REML is deterministic; this
        bounds the variance-component optimisation, not a sampler.
    tol : float
        Convergence tolerance recorded in the manifest for
        provenance. Currently informational — ``sommer`` owns its own
        stopping rule; kept so the field exists if a future driver
        wants to thread it through.
    seed : int
        RNG seed forwarded to R via ``set.seed`` for parity with the
        BGLR path. REML predictions are deterministic, so this only
        affects any internal start-value jitter.
    env_brr_key : str | None
        Block name for the env one-hot. When set, ``fit_predict_blocks``
        REQUIRES the caller to pass the env design matrix via the
        dedicated ``env_block`` argument (matching ``BGLRRegressor``);
        the wrapper converts it to ``K_env = Z_e Z_eᵀ`` and appends it
        as the terminal kernel block, so the REML model carries an env
        variance component analogous to BGLR's ``BRR(Z_e)``.
    """

    is_transductive: ClassVar[bool] = True

    def __init__(
        self,
        reml_max_iter: int = 40,
        tol: float = 1e-4,
        seed: int = 0,
        env_brr_key: str | None = None,
    ):
        if reml_max_iter <= 0:
            raise ValueError(
                f"reml_max_iter must be > 0, got {reml_max_iter}"
            )
        self.reml_max_iter = int(reml_max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.env_brr_key = env_brr_key

    # ------------------------------------------------------------------
    # Inductive predict is intentionally unsupported (transductive only).
    # ------------------------------------------------------------------

    def predict(self, X):  # noqa: ARG002 — sklearn-compat signature
        raise RuntimeError(
            "GBLUPRegressor is transductive — call "
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
        """Run GBLUP transductively and return (y_pred_train, y_pred_test, meta).

        Parameters
        ----------
        K_blocks : ordered dict of raw obs-level kernels. Each value
            has shape ``(N, N)`` with ``N = len(y_train) + n_test`` and
            rows in ``[train..., test...]`` order. Insertion order
            defines the variance-component term sequence — pass blocks
            in ``feature_keys`` order to mirror the BGLR ordering.
        y_train : ``(n_train,)`` training targets.
        n_test : number of test rows.
        env_block : optional ``(N, n_envs)`` env one-hot design matrix.
            Required iff ``self.env_brr_key`` is set; folded into
            ``K_env = Z_e Z_eᵀ`` and appended as the terminal block.
        """
        n_train = len(y_train)
        N = n_train + n_test
        block_names = list(K_blocks.keys())
        if not block_names:
            raise ValueError(
                "GBLUPRegressor: at least one kernel block is "
                "required (empty K_blocks)."
            )

        for name in block_names:
            K = K_blocks[name]
            if K.ndim != 2 or K.shape[0] != N or K.shape[1] != N:
                raise ValueError(
                    f"GBLUPRegressor: kernel block {name!r} must be "
                    f"({N}, {N}); got shape={K.shape}."
                )

        # Env-block invariant mirrors BGLRRegressor: when env_brr_key is
        # set, the caller routes the env design matrix through
        # env_block. Here we fold it into a linear kernel rather than a
        # separate BRR term.
        blocks: dict[str, np.ndarray] = dict(K_blocks)
        if self.env_brr_key is not None:
            if env_block is None:
                raise ValueError(
                    f"GBLUPRegressor: env_brr_key="
                    f"{self.env_brr_key!r} declared but env_block "
                    f"was not passed."
                )
            if env_block.ndim != 2 or env_block.shape[0] != N:
                raise ValueError(
                    f"GBLUPRegressor: env_block must be (N={N}, p); "
                    f"got shape={env_block.shape}."
                )
            if self.env_brr_key in block_names:
                raise ValueError(
                    f"GBLUPRegressor: env_brr_key="
                    f"{self.env_brr_key!r} also appears in the kernel "
                    f"block names — caller should keep env separate "
                    f"from K_blocks."
                )
            # K_env = Z_e Z_eᵀ: the RKHS form of a ridge on the env
            # one-hot. K_env[te, tr] is non-zero only for envs shared
            # between test and train, so a fully held-out env (LOEO)
            # contributes 0 to its own predictions — matching BGLR's
            # BRR coefficient shrinking to the prior mean.
            Z = np.ascontiguousarray(env_block, dtype=np.float64)
            blocks[self.env_brr_key] = Z @ Z.T
        elif env_block is not None:
            raise ValueError(
                "GBLUPRegressor: env_block passed but env_brr_key "
                "is None — set env_brr_key or drop env_block."
            )

        tmpdir = tempfile.mkdtemp(prefix="gblup_")
        logger.info("GBLUP work_dir: %s", tmpdir)

        try:
            self._write_inputs(tmpdir, blocks, y_train, n_test)
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
        blocks: dict[str, np.ndarray],
        y_train: np.ndarray,
        n_test: int,
    ) -> None:
        os.makedirs(os.path.join(tmpdir, "blocks"), exist_ok=True)
        n_train = len(y_train)

        # y_full.csv — test rows are placeholder zeros; R gates on
        # train_mask and never reads them into the REML fit.
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

        block_names = list(blocks.keys())
        for name in block_names:
            self._write_kernel_block_bin(tmpdir, name, blocks[name])

        manifest = {
            "block_names": block_names,
            "env_brr_key": self.env_brr_key,
            "reml_max_iter": self.reml_max_iter,
            "tol": self.tol,
            "seed": self.seed,
            "n_train": n_train,
            "n_test": n_test,
        }
        with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _write_kernel_block_bin(
        tmpdir: str, name: str, K: np.ndarray,
    ) -> None:
        """Write a raw obs-level N×N kernel as float64 (row-major).

        Identical wire format to ``BGLRRegressor`` so the two R drivers
        read kernels the same way. The caller builds each K over the
        full ``[train + test]`` row set.
        """
        K_c = np.ascontiguousarray(K, dtype=np.float64)
        K_c.tofile(os.path.join(tmpdir, "blocks", f"{name}.bin"))

    def _run_rscript(self, tmpdir: str) -> None:
        cmd = [
            "Rscript",
            os.path.normpath(_R_SCRIPT),
            "--work_dir", tmpdir,
        ]
        # Unlike the BGLR path, do NOT pin BLAS/OMP threads to 1. BGLR's
        # Gibbs sampler is sequential, so single-thread is free; sommer's
        # REML is dense linear algebra (Cholesky / MME solves on the
        # n_train kernel) that scales with multithreaded BLAS. Pinning to
        # 1 thread made a full-fold REML take >10× longer. REML is
        # deterministic up to convergence tolerance, so multithreading
        # does not hurt prediction reproducibility. Inherit the parent
        # env and let the BLAS backend size its own pool.
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True,
        )
        if result.stdout:
            logger.info(result.stdout.rstrip())
        if result.stderr:
            logger.warning(result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(
                f"GBLUP R subprocess failed (code "
                f"{result.returncode}):\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )

    @staticmethod
    def _read_yhat(tmpdir: str, n_total: int) -> np.ndarray:
        path = os.path.join(tmpdir, "yhat.csv")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"GBLUP R subprocess did not produce yhat.csv at {path}"
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
        path = os.path.join(tmpdir, "gblup_meta.json")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"GBLUP R subprocess did not produce gblup_meta.json "
                f"at {path}"
            )
        with open(path) as f:
            return json.load(f)
