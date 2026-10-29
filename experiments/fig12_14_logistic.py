"""Figs. 12-14: logistic regression by full-batch gradient descent (Sec. IV B).

Student-teacher overlap R (pruned N0-block) after training, and the training
trajectories R(eta t), kappa(eta t). The "max margin" lines are the replica
max-margin values (sigma_in = 1), defined only below capacity.
Fig. 12: beta = 1, R vs alpha0 for several eta.
Fig. 13: beta = 1, R and kappa vs eta t at one alpha0 below and one above capacity.
Fig. 14: R vs alpha0 for several beta (stochastic units, sigma_in = 1) at fixed eta;
  max-margin line for the largest beta and markers at each capacity.

Conventions of the published runs:
learning rates in published units (step eta sqrt(N), see nn_expansion.logistic), random
initialisation 0.1 N(0,1)/sqrt(N), stop when |step| < 1e-6, 10001 epochs, traces
sampled every 20 epochs. The published Fig. 13 scaled the time axis of the
eta = 0.05 and 0.005 runs by 0.1 and 0.05 instead of their own eta; the overlay below undoes that (PUBLISHED_13_X_SCALE).
Its "max margin" line in the alpha0 = 8 panel is the replica R at alpha0 = 3.86
(another grid index) and is not drawn here: alpha0 = 8 is above capacity.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.simulate import Logistic, NoExpansion, Stochastic, TrialSpec
from nn_expansion.sweep import Series, SeriesResult
from nn_expansion.theory import AboveCapacity, R_curve_sigma_in_1, capacity_alpha0, saddle_sigma_in_1

from .common import (PUBLISHED_STYLE, Panel, alpha0_grid, errorbar, label_axes, linspace, overlay_published,
                     reference_series, series_arrays)

NAME = "fig12_14"

# published eta -> factor that maps the published Fig. 13 x values to eta * t
PUBLISHED_13_X_SCALE = {0.5: 1.0, 0.05: 0.05 / 0.1, 0.005: 0.005 / 0.05}


@dataclass(frozen=True)
class Config:
    sigma_out: float
    seed: int
    eta_unit: str                 # "paper" | "published"
    init_scale: float
    stop_tol: float
    rule: str                     # trial averaging (logistic trials never fail)
    n_sem: float                  # error bars in standard errors (published: 1)
    # Fig. 12
    N0_12: int
    etas_12: list[float]
    alpha0_grid_12: list[float]
    epochs_12: int
    trials_12: int
    # Fig. 13
    N0_13: int
    alpha0_13: list[float]
    etas_13: list[float]
    epochs_13: int
    record_every_13: int
    trials_13: int
    # Fig. 14
    N0_14: int
    betas_14: list[float]
    eta_14: float
    alpha0_grid_14: list[float]
    epochs_14: int
    trials_14: int


def _learner(cfg: Config, eta: float, epochs: int, record_every: int | None) -> Logistic:
    return Logistic(eta=eta, eta_unit=cfg.eta_unit, epochs=epochs, init_scale=cfg.init_scale,
                    stop_tol=cfg.stop_tol, record_every=record_every)


def _specs(N0: int, a0: np.ndarray, sigma_out: float, beta: float, learner: Logistic) -> tuple[TrialSpec, ...]:
    return alpha0_grid(a0, N0=N0, teacher_a=1.0, sigma_out=sigma_out,
                       expansion=NoExpansion() if beta == 1.0 else Stochastic(beta, 1.0),
                       learner=learner, student="linear", measure_unpruned=False, n_test=0)


def plan(cfg: Config) -> list[Series]:
    so = cfg.sigma_out
    a12, a14 = linspace(cfg.alpha0_grid_12), linspace(cfg.alpha0_grid_14)
    out = [Series(f"12_eta{k}", _specs(cfg.N0_12, a12, so, 1.0, _learner(cfg, eta, cfg.epochs_12, None)),
                  cfg.trials_12, cfg.seed, 100 + k)
           for k, eta in enumerate(cfg.etas_12)]
    for i, a0 in enumerate(cfg.alpha0_13):
        for k, eta in enumerate(cfg.etas_13):
            # same stream for every eta: each trace is trained on the same data
            learner = _learner(cfg, eta, cfg.epochs_13, cfg.record_every_13)
            out.append(Series(f"13_a{i}_eta{k}", _specs(cfg.N0_13, np.array([a0]), so, 1.0, learner),
                              cfg.trials_13, cfg.seed, 200 + i))
    out += [Series(f"14_beta{k}", _specs(cfg.N0_14, a14, so, b, _learner(cfg, cfg.eta_14, cfg.epochs_14, None)),
                   cfg.trials_14, cfg.seed, 300 + k)
            for k, b in enumerate(cfg.betas_14)]
    return out


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    so = cfg.sigma_out
    out: dict[str, np.ndarray] = {}
    a12 = linspace(cfg.alpha0_grid_12)
    out["alpha0_12"] = a12
    for k in range(len(cfg.etas_12)):
        out.update(series_arrays(f"12_eta{k}", sims[f"12_eta{k}"].sweep, ("R_pruned",), cfg.rule))
    out["12_maxmargin"] = R_curve_sigma_in_1(a12, 1.0, so)

    for i, a0 in enumerate(cfg.alpha0_13):
        for k, eta in enumerate(cfg.etas_13):
            trials = sims[f"13_a{i}_eta{k}"].raw[0]
            out[f"13_t_a{i}_eta{k}"] = trials[0].trace_epochs * eta
            out[f"13_R_a{i}_eta{k}"] = np.mean([r.trace_R for r in trials], axis=0)
            out[f"13_kappa_a{i}_eta{k}"] = np.mean([r.trace_kappa for r in trials], axis=0)
        try:
            sp = saddle_sigma_in_1(a0, so)
            out[f"13_maxmargin_a{i}"] = np.array([sp.m, sp.kappa])
        except AboveCapacity:
            out[f"13_maxmargin_a{i}"] = np.array([np.nan, np.nan])     # no max-margin solution

    a14 = linspace(cfg.alpha0_grid_14)
    out["alpha0_14"] = a14
    for k in range(len(cfg.betas_14)):
        out.update(series_arrays(f"14_beta{k}", sims[f"14_beta{k}"].sweep, ("R_pruned",), cfg.rule))
    out["14_maxmargin"] = R_curve_sigma_in_1(a14, max(cfg.betas_14), so)
    out["14_capacity"] = np.array([capacity_alpha0(b, so) for b in cfg.betas_14])
    return out


def _fig12(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\alpha_0$", r"$R$",
               rf"Fig. 12  β=1, $\sigma_{{out}}={cfg.sigma_out:g}$, $N={cfg.N0_12}$, "
               f"{cfg.epochs_12} epochs, {cfg.trials_12} trials")
    for k, eta in enumerate(cfg.etas_12):
        overlay_published(ax, "fig12", rf"\eta={eta:g}", legend="published" if k == 0 else None)
        errorbar(ax, res["alpha0_12"], res, f"12_eta{k}", "R_pruned", cfg.n_sem, label=rf"$\eta={eta:g}$")
    overlay_published(ax, "fig12", "max margin")
    ax.plot(res["alpha0_12"], res["12_maxmargin"], "k", label="max margin (replica)")


def _fig13(ax: Axes, res: dict[str, np.ndarray], cfg: Config, q: str, i: int) -> None:
    panel, ylabel, j = ("fig13a", "R", 0) if q == "R" else ("fig13b", r"$\kappa$", 1)
    label_axes(ax, r"$\eta t$", ylabel, rf"Fig. {panel[3:]} ($\alpha_0={cfg.alpha0_13[i]:g}$)  β=1, "
               rf"$N={cfg.N0_13}$, {cfg.trials_13} trial(s)")
    for k, eta in enumerate(cfg.etas_13):
        pub = reference_series(panel, rf"\eta={eta:g}", i)
        ax.plot(pub["x"] * PUBLISHED_13_X_SCALE[eta], pub["y"], **PUBLISHED_STYLE,
                label="published (time axis corrected)" if k == 0 else None)
        ax.plot(res[f"13_t_a{i}_eta{k}"], res[f"13_{q}_a{i}_eta{k}"], label=rf"$\eta={eta:g}$")
    mm = res[f"13_maxmargin_a{i}"][j]
    if np.isfinite(mm):
        ax.axhline(mm, color="k", lw=1, label="max margin (replica)")
    ax.set_xscale("log")


def _fig14(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\alpha_0$", r"$R$ (pruned)",
               rf"Fig. 14  $\eta={cfg.eta_14:g}$, $\sigma_{{out}}={cfg.sigma_out:g}$, "
               rf"$N_0={cfg.N0_14}$, {cfg.trials_14} trials")
    for k, b in enumerate(cfg.betas_14):
        overlay_published(ax, "fig14", rf"\beta={b:g}", legend="published" if k == 0 else None)
        errorbar(ax, res["alpha0_14"], res, f"14_beta{k}", "R_pruned", cfg.n_sem, label=rf"$\beta={b:g}$")
        ac = res["14_capacity"][k]
        ax.plot([ac], [np.interp(ac, res["alpha0_14"], res[f"14_beta{k}/R_pruned"])], "o", mfc="none", color="k")
    overlay_published(ax, "fig14", "max margin")
    ax.plot(res["alpha0_14"], res["14_maxmargin"], "k", label=rf"max margin β={max(cfg.betas_14):g}")


def panels(cfg: Config) -> dict[str, Panel]:
    out: dict[str, Panel] = {"fig12": partial(_fig12, cfg=cfg)}
    for q, letter in (("R", "a"), ("kappa", "b")):
        for i, a0 in enumerate(cfg.alpha0_13):
            out[f"fig13{letter}_alpha0_{a0:g}"] = partial(_fig13, cfg=cfg, q=q, i=i)
    out["fig14"] = partial(_fig14, cfg=cfg)
    return out
