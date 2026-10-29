"""Fig. 8: stochastic expansion vs deterministic sparse / dense expansion.

sigma_in = A = 1, so the expanded units have matched first and second moments.
(a) Pruned Eg vs alpha0 at fixed beta: deterministic sparse and dense
simulations against the stochastic replica theory (sigma_in = 1).
(b) Best pruned Eg (minimum over the alpha0 grid of the trial-averaged curve)
vs beta for stochastic, sparse and dense expansion, with the replica infimum
eg_at_capacity for the stochastic case.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.simulate import (Deterministic, Expansion, MaxMargin, NoExpansion, Stochastic,
                              TrialSpec)
from nn_expansion.sweep import Series, SeriesResult
from nn_expansion.theory import eg_at_capacity, eg_curve_sigma_in_1

from .common import (Panel, alpha0_grid, errorbar, label_axes, linspace, oracle_line, overlay_published,
                     series_arrays)

NAME = "fig08"


@dataclass(frozen=True)
class Config:
    # (a)
    a_N0: int
    a_beta: float
    a_sigma_out: float
    a_alpha0_grid: list[float]
    # (b)
    b_N0: int
    b_sigma_out: float
    b_beta_grid: list[float]
    b_alpha0_grid: list[float]    # grid each beta is minimised over
    # shared
    f_sparse: float
    f_dense: float
    A: float                      # == sigma_in
    trials: int
    seed: int
    rule: str                     # trial averaging, see nn_expansion.sweep.SweepResult.mean
    a_n_sem: float                # error bars of (a) in standard errors (published: 1)
    b_n_sem: float                # error bars of (b) in standard errors (published: 2)


KINDS_B = ("random", "sparse", "dense")


def _expansion(kind: str, beta: float, cfg: Config) -> Expansion:
    if beta == 1.0:
        return NoExpansion()
    if kind == "random":
        return Stochastic(beta, cfg.A)
    return Deterministic(beta, cfg.f_sparse if kind == "sparse" else cfg.f_dense, cfg.A)


def _specs(N0: int, sigma_out: float, a0: np.ndarray, expansion: Expansion) -> tuple[TrialSpec, ...]:
    return alpha0_grid(a0, N0=N0, teacher_a=1.0, sigma_out=sigma_out, expansion=expansion,
                       learner=MaxMargin(), student="linear", measure_unpruned=False, n_test=0)


def _b_key(kind: str, beta: float, j: int) -> str:
    return "b_unexpanded" if beta == 1.0 else f"b_{kind}{j}"   # beta = 1: one network for all kinds


def plan(cfg: Config) -> list[Series]:
    a0 = linspace(cfg.a_alpha0_grid)
    out = [Series(f"a_{kind}", _specs(cfg.a_N0, cfg.a_sigma_out, a0, _expansion(kind, cfg.a_beta, cfg)),
                  cfg.trials, cfg.seed, sid) for sid, kind in enumerate(("sparse", "dense"))]
    ga0 = linspace(cfg.b_alpha0_grid)
    sid = 10
    for kind in KINDS_B:
        for j, b in enumerate(linspace(cfg.b_beta_grid)):
            key = _b_key(kind, float(b), j)
            if any(s.key == key for s in out):
                continue
            out.append(Series(key, _specs(cfg.b_N0, cfg.b_sigma_out, ga0, _expansion(kind, float(b), cfg)),
                              cfg.trials, cfg.seed, sid))
            sid += 1
    return out


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    a0 = linspace(cfg.a_alpha0_grid)
    out: dict[str, np.ndarray] = {"a_alpha0": a0}
    for kind in ("sparse", "dense"):
        out.update(series_arrays(f"a_{kind}", sims[f"a_{kind}"].sweep, ("eg_pruned",), cfg.rule))
    out["a_theory"] = eg_curve_sigma_in_1(a0, cfg.a_beta, cfg.a_sigma_out, pruned=True)

    betas = linspace(cfg.b_beta_grid)
    out["b_beta"] = betas
    for kind in KINDS_B:
        best = np.full(betas.shape, np.nan)
        best_sem = np.full(betas.shape, np.nan)
        for j, b in enumerate(betas):
            sw = sims[_b_key(kind, float(b), j)].sweep
            m = sw.mean("eg_pruned", cfg.rule)
            if np.all(np.isnan(m)):
                raise RuntimeError(f"Fig. 8b {kind} beta={b}: no alpha0 passes the rule {cfg.rule!r}")
            k = int(np.nanargmin(m))
            best[j], best_sem[j] = m[k], sw.sem("eg_pruned")[k]
        out[f"b_{kind}"], out[f"b_{kind}_sem"] = best, best_sem
    out["b_theory"] = np.array([eg_at_capacity(float(b), cfg.b_sigma_out) for b in betas])
    return out


def _fig8a(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\alpha_0$", r"$E_g$ (pruned)",
               rf"Fig. 8a  β={cfg.a_beta:g}, $\sigma_{{out}}={cfg.a_sigma_out:g}$, "
               rf"$A=\sigma_{{in}}={cfg.A:g}$, $N_0={cfg.a_N0}$, {cfg.trials} trials")
    for lab in ("f =0.02", "f =0.5", "theory"):
        overlay_published(ax, "fig08a", lab, legend="published" if lab == "theory" else None)
    errorbar(ax, res["a_alpha0"], res, "a_sparse", "eg_pruned", cfg.a_n_sem, label=f"f={cfg.f_sparse:g}")
    errorbar(ax, res["a_alpha0"], res, "a_dense", "eg_pruned", cfg.a_n_sem, label=f"f={cfg.f_dense:g}")
    ax.plot(res["a_alpha0"], res["a_theory"], "r", label="stochastic theory")
    oracle_line(ax, cfg.a_sigma_out)


def _fig8b(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\beta$", r"$\min_{\alpha_0} E_g$ (pruned)",
               rf"Fig. 8b  $\sigma_{{out}}={cfg.b_sigma_out:g}$, $N_0={cfg.b_N0}$, {cfg.trials} trials")
    for lab in ("f=0.5", "f=0.02", "random"):
        overlay_published(ax, "fig08b", lab, legend="published" if lab == "random" else None)
    for kind, lab in (("random", "stochastic"), ("sparse", f"f={cfg.f_sparse:g}"), ("dense", f"f={cfg.f_dense:g}")):
        ax.errorbar(res["b_beta"], res[f"b_{kind}"], yerr=cfg.b_n_sem * res[f"b_{kind}_sem"], fmt=".-",
                    ms=4, capsize=2, label=lab)
    ax.plot(res["b_beta"], res["b_theory"], "r:", label=r"theory, $\alpha_0\to\alpha_{0,c}$")
    oracle_line(ax, cfg.b_sigma_out)


def panels(cfg: Config) -> dict[str, Panel]:
    return {"fig08a": partial(_fig8a, cfg=cfg), "fig08b": partial(_fig8b, cfg=cfg)}
