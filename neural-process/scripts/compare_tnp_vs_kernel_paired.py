#!/usr/bin/env python
"""Paired TNP-vs-kernel-model comparison on IDENTICAL splits (same fold table, same cv_seed).

Reads per-split metrics from one artifact root:
  <root>/neural_process/<scheme>/<token>/<slug>/seed=<s>/metrics_<checkpoint>.json
  <root>/fpca/<scheme>/<token>/<slug>/seed=<s>/metrics.json
and joins TNP and FPCA on (scheme, token, cv_label, seed). Reports, per model and
CV label: macro RMSE (mean over splits), pooled RMSE (n-weighted), Tiezzi r_w, and
for every TNP-vs-kernel pair the mean paired RMSE difference with a paired t-test and
the fraction of splits where TNP has the lower RMSE.

Usage:
    python scripts/compare_tnp_vs_kernel_paired.py --root artifacts/cv/cv_0_00_2_1 \
        --seeds 1 --checkpoint best_frozen_pearson_r --out exports/tnp_vs_kernel_paired.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

TNP_NAMES = {
    "wthr.enc": "TNP full",
    "ga.rows": "TNP weather-free",
}


def tnp_label(slug: str) -> str:
    if "wthr.enc" in slug:
        return "TNP full"
    if "ga.rows" in slug:
        return "TNP weather-free"
    return "TNP VI-only"


def fpca_label(slug: str) -> str:
    family = "weather-GxE" if "Xwthr" in slug else ("env-GxE" if "Xeid" in slug else "main-effects")
    env = "env" if slug.startswith("fpca_eid") else "env-free"
    axis = "AGDD" if "vi.t=agdd" in slug else "DAP"
    return f"{family} {env} {axis}"


def load(root: str, seeds: list[int], checkpoint: str) -> pd.DataFrame:
    rows = []
    for s in seeds:
        for m in glob.glob(f"{root}/neural_process/*/*/*/seed={s}/metrics_{checkpoint}.json"):
            p = m.split(os.sep)
            scheme, token, slug = p[-5], p[-4], p[-3]
            for lab, v in json.load(open(m))["by_metric"].items():
                rows.append(dict(cls="tnp", model=tnp_label(slug), scheme=scheme, token=token,
                                 cv_label=lab, seed=s, n=v["n"], rmse=v["rmse"], r=v["pearson_r"]))
        for m in glob.glob(f"{root}/fpca/*/*/*/seed={s}/metrics.json"):
            p = m.split(os.sep)
            scheme, token, slug = p[-5], p[-4], p[-3]
            for lab, v in json.load(open(m))["by_metric"].items():
                rows.append(dict(cls="fpca", model=fpca_label(slug), scheme=scheme, token=token,
                                 cv_label=lab, seed=s, n=v["n"], rmse=v["rmse"], r=v["pearson_r"]))
    return pd.DataFrame(rows)


def tiezzi(r: np.ndarray, n: np.ndarray) -> float:
    w = (n - 2) / (1 - r**2)
    return float((w * r).sum() / w.sum())


def summarize(g: pd.DataFrame) -> pd.Series:
    return pd.Series(dict(
        splits=len(g),
        rmse_macro=g.rmse.mean(),
        rmse_pooled=float(np.sqrt((g.n * g.rmse**2).sum() / g.n.sum())),
        r_tiezzi=tiezzi(g.r.values, g.n.values),
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/cv/cv_0_00_2_1")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--checkpoint", default="best_frozen_pearson_r")
    ap.add_argument("--out", default="exports/tnp_vs_kernel_paired.csv")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    df = load(a.root, seeds, a.checkpoint)
    if df.empty:
        raise SystemExit("no metrics found")
    key = ["scheme", "token", "cv_label", "seed"]

    # ── per-model summaries ────────────────────────────────────────────────
    summ = df.groupby(["model", "cv_label"]).apply(summarize, include_groups=False).reset_index()
    pd.set_option("display.width", 220)
    print(f"root={a.root} seeds={seeds} checkpoint={a.checkpoint}")
    for lab in ["CV2", "CV1", "CV0", "CV00"]:
        sub = summ[summ.cv_label == lab].drop(columns="cv_label")
        if sub.empty:
            continue
        print(f"\n=== {lab} — summary")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── paired differences: TNP − kernel model on identical splits ──────────────────
    tnp = df[df.cls == "tnp"]
    fp = df[df.cls == "fpca"]
    pairs = []
    for tm in sorted(tnp.model.unique()):
        for fm in sorted(fp.model.unique()):
            j = tnp[tnp.model == tm].merge(fp[fp.model == fm], on=key, suffixes=("_t", "_f"))
            for lab, g in j.groupby("cv_label"):
                d = g.rmse_t - g.rmse_f
                t, pval = stats.ttest_rel(g.rmse_t, g.rmse_f) if len(g) > 1 else (np.nan, np.nan)
                pooled_t = np.sqrt((g.n_t * g.rmse_t**2).sum() / g.n_t.sum())
                pooled_f = np.sqrt((g.n_f * g.rmse_f**2).sum() / g.n_f.sum())
                pairs.append(dict(
                    tnp=tm, kernel=fm, cv_label=lab, splits=len(g),
                    n_match=bool((g.n_t == g.n_f).all()),
                    d_rmse_macro=d.mean(), d_rmse_macro_pct=100 * d.mean() / g.rmse_f.mean(),
                    d_rmse_pooled=pooled_t - pooled_f,
                    d_rmse_pooled_pct=100 * (pooled_t - pooled_f) / pooled_f,
                    tnp_wins_frac=float((d < 0).mean()),
                    paired_t=t, p=pval,
                    d_r_tiezzi=tiezzi(g.r_t.values, g.n_t.values) - tiezzi(g.r_f.values, g.n_f.values),
                ))
    pr = pd.DataFrame(pairs)
    for lab in ["CV2", "CV1", "CV0", "CV00"]:
        sub = pr[pr.cv_label == lab].drop(columns="cv_label")
        if sub.empty:
            continue
        print(f"\n=== {lab} — paired TNP − kernel (negative RMSE diff = TNP better)")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    summ.to_csv(a.out, index=False)
    pr.to_csv(a.out.replace(".csv", "_pairs.csv"), index=False)
    df.to_csv(a.out.replace(".csv", "_splits.csv"), index=False)
    print(f"\nwrote {a.out}, *_pairs.csv, *_splits.csv")


if __name__ == "__main__":
    main()
