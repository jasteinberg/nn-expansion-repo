"""Logistic regression by full-batch gradient descent (Sec. IV B, Eqs. 38-39).

L(W) = sum_mu log(1 + exp(-u_mu)),  u_mu = y_mu W.Phi_mu  (Phi rows carry the
1/sqrt(N) of Eq. 39), and W <- W - eta dL/dW (the update below Eq. 39).

Learning-rate unit: the published runs used the field h = w.x/sqrt(N) but the
update dw = sum_mu (s_mu - sigmoid(h_mu)) x_mu without the 1/sqrt(N), i.e. a step
eta sqrt(N) (-dL/dW). The published Figs. 12-14 therefore correspond to
eta_paper * sqrt(N) here (``effective_eta`` with unit "published").
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .backend import Device

ETA_UNITS = ("paper", "published")


def effective_eta(eta: float, unit: str, N: int) -> float:
    """Step size of W <- W - eta_eff dL/dW for a learning rate quoted in ``unit``."""
    if unit == "paper":
        return eta
    if unit == "published":
        return eta * np.sqrt(N)
    raise ValueError(f"eta unit must be one of {ETA_UNITS}, got {unit!r}")


def logistic_gd(Phi: np.ndarray, y: np.ndarray, eta: float, epochs: int, W_init: np.ndarray,
                stop_tol: float, record_every: int | None = None,
                record: Callable[[np.ndarray], tuple[float, ...]] | None = None,
                device: Device = Device.cpu()) -> tuple[np.ndarray, np.ndarray, list[tuple[float, ...]]]:
    """Run up to ``epochs`` steps of full-batch GD on the logistic loss.

    Stops early once a step has norm < ``stop_tol`` (1e-6 in the published runs);
    recorded values after the stop repeat the last state, as in the published traces. Returns (W_final, recorded_epochs, recorded_values);
    ``record(W)`` is evaluated after epoch 1 and every ``record_every`` epochs
    (t = 1, 1 + k, 1 + 2k, ...), the sampling of the published Fig. 13 traces.
    """
    if (record_every is None) != (record is None):
        raise ValueError("record_every and record must be given together")
    if eta <= 0 or epochs < 1 or stop_tol < 0:
        raise ValueError(f"need eta > 0, epochs >= 1, stop_tol >= 0; got {eta}, {epochs}, {stop_tol}")
    xp = device.ops()
    G = xp.asarray(np.asarray(y, float)[:, None] * np.asarray(Phi, float))
    W = xp.asarray(W_init)
    if tuple(W.shape) != (G.shape[1],):
        raise ValueError(f"W_init must have shape ({G.shape[1]},), got {tuple(W.shape)}")
    ts = list(range(1, epochs + 1, record_every)) if record_every is not None else []
    vals: list[tuple[float, ...]] = []
    to_np = xp.to_numpy
    for t in range(1, epochs + 1):
        step = eta * (xp.expit(-(G @ W)) @ G)          # -eta grad L = eta sum sigmoid(-u) G
        W = W + step
        if record is not None and (t - 1) % record_every == 0:   # type: ignore[operator]
            vals.append(record(to_np(W)))
        if xp.norm(step) < stop_tol:
            break
    if record is not None:
        vals.extend([vals[-1]] * (len(ts) - len(vals)))
    W_out = to_np(W)
    if not np.all(np.isfinite(W_out)):
        raise FloatingPointError(f"logistic GD diverged (eta_eff={eta}): non-finite weights")
    return W_out, np.asarray(ts), vals
