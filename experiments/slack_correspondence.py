"""Sec. III C (Eqs. 25-30): stochastic expansion -> L2-slack SVM as beta -> inf.

On the same training set, compare the pruned max-margin student of the
stochastically expanded network (N_plus units of variance sigma_in^2) with the
unexpanded L2-slack SVM (Eq. 25). With rows Phi = [x, u]/sqrt(N), the expanded
units add sigma_in^2 (1 - 1/beta) to the diagonal of the pattern Gram matrix,
plus off-diagonal terms of order 1/sqrt(N_plus); the slack SVM on x/sqrt(N)
with C = 1/sigma_in^2 adds exactly 1/C to the diagonal (Eq. 30). The cosine
between the two weight vectors should approach 1 as beta grows.
No published figure; the panel is a check of the stated correspondence. Each trial
pairs two solvers on one training set, so it runs as a whole (``run_check``, no
shards, CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes
from joblib import Parallel, delayed

from nn_expansion.expansion import design_matrix, stochastic_units
from nn_expansion.simulate import overlap
from nn_expansion.svm import l2_slack_svm, max_margin
from nn_expansion.teacher import draw_inputs, draw_teacher
from nn_expansion.theory import eg_from_overlap

from .common import Panel, label_axes

NAME = "slack"


@dataclass(frozen=True)
class Config:
    N0: int
    alpha0: float
    sigma_out: float
    sigma_in: float
    betas: list[float]
    trials: int
    seed: int


def _trial(cfg: Config, beta: float, j: int, t: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, j, t]))
    teacher = draw_teacher(cfg.N0, 1.0, cfg.sigma_out, rng)
    P = int(round(cfg.alpha0 * cfg.N0))
    N = int(round(beta * cfg.N0))
    X = draw_inputs(P, cfg.N0, rng)
    y = teacher.labels(X, rng)
    W = max_margin(design_matrix(X, stochastic_units(P, N - cfg.N0, cfg.sigma_in, rng)), y).W
    w_exp = W[:cfg.N0]
    w_slack = l2_slack_svm(X / np.sqrt(N), y, 1.0 / cfg.sigma_in ** 2).w
    eg = lambda w: float(eg_from_overlap(overlap(w, teacher.w0), cfg.sigma_out))   # noqa: E731
    return overlap(w_exp, w_slack), eg(w_exp), eg(w_slack)


def run_check(cfg: Config, n_jobs: int) -> dict[str, np.ndarray]:
    jobs = [(j, b, t) for j, b in enumerate(cfg.betas) for t in range(cfg.trials)]
    res = Parallel(n_jobs=n_jobs)(delayed(_trial)(cfg, b, j, t) for j, b, t in jobs)
    arr = np.array(res).reshape(len(cfg.betas), cfg.trials, 3)
    return {"betas": np.array(cfg.betas), "cos": arr[..., 0], "eg_expanded": arr[..., 1],
            "eg_slack": arr[..., 2]}


def _slack(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\beta$", "value",
               rf"Eqs. 25-30  $\alpha_0={cfg.alpha0:g}$, $\sigma_{{in}}={cfg.sigma_in:g}$, "
               rf"$\sigma_{{out}}={cfg.sigma_out:g}$, $N_0={cfg.N0}$, {cfg.trials} trials")
    b = res["betas"]
    sem = res["cos"].std(axis=1, ddof=1) / np.sqrt(res["cos"].shape[1])
    ax.errorbar(b, 1 - res["cos"].mean(1), yerr=sem, fmt="o-", ms=4, label=r"$1-\cos(w_{exp}, w_{slack})$")
    ax.plot(b, np.abs(res["eg_expanded"] - res["eg_slack"]).mean(1), "s-", ms=4,
            label=r"$|E_g^{exp} - E_g^{slack}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")


def panels(cfg: Config) -> dict[str, Panel]:
    return {"slack": partial(_slack, cfg=cfg)}
