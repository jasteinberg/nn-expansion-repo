"""Max-margin learning without bias (Eqs. 8-9) and the L2-slack SVM (Eq. 25).

Hard margin
-----------
With G_mu = y_mu Phi_mu (rows), the max-margin weights solve the QP

    min_W (1/2)|W|^2   s.t.  G W >= 1,                                   (8)-(9)

whose Lagrange dual is  max_{lam >= 0} 1.lam - (1/2)|G^T lam|^2,  W = G^T lam.
At the optimum 1.lam = |W|^2 = 1/gamma^2, where gamma = min_mu G_mu.W / |W| is
the geometric margin. Normalising lam_hat = lam / 1.lam puts lam_hat on the
simplex, and gamma = min_{lam_hat in simplex} |G^T lam_hat| (the minimum-norm
point of conv{G_mu}). The data are separable without bias iff 0 is not in
conv{G_mu} (Gordan's theorem), so |G^T lam_hat| ~ 0 is a Farkas certificate of
non-separability -- the regime above capacity, where the paper records no
solution.

We solve the QP with a dense primal-dual interior-point method (Mehrotra
predictor-corrector). Each Newton step reduces to the N x N system

    (I + G^T D G) dW = rhs,   D = lam / s,

so a step costs O(P N^2), which is far cheaper than a general sparse solver on
this dense problem.

L2 slack
--------
min |w|^2 + C sum xi^2 s.t. y_mu w.x_mu >= 1 - xi_mu (Eq. 25) is the hard margin
on features augmented by one private unit per pattern, [Phi_mu, y_mu e_mu/sqrt(C)]:
the weight v_mu on that unit gives xi_mu = v_mu / sqrt(C). This is exactly the
paper's argument (Eqs. 26-30) that orthogonal expanded units act as slack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import Device, Ops


class NotSeparable(RuntimeError):
    """No W with y_mu W.Phi_mu >= 1 for all mu (the training set is above capacity)."""


class SolverFailure(RuntimeError):
    """The interior-point method neither converged nor certified infeasibility."""


@dataclass(frozen=True)
class MaxMarginResult:
    W: np.ndarray          # (N,) weights; min_mu G_mu.W = 1 at the optimum
    lam: np.ndarray        # (P,) dual variables (support vectors have lam > 0)
    iterations: int
    rel_gap: float         # (primal - dual) / (1 + primal) at termination


def max_margin(Phi: np.ndarray, y: np.ndarray, *, tol: float = 1e-8,
               infeas_tol: float = 1e-8, max_iter: int = 200,
               device: Device = Device.cpu()) -> MaxMarginResult:
    """Hard-margin SVM without bias, Eqs. (8)-(9). Raises NotSeparable above capacity.

    Phi: (P, N) design matrix (rows are patterns); y: (P,) labels in {-1,+1}.
    tol: relative tolerance on primal/dual residuals and duality gap.
    infeas_tol: |G^T lam_hat| / max_mu |G_mu| below which the Farkas certificate
    is accepted.
    device: where the Newton systems are solved (see nn_expansion.backend).
    """
    Phi = np.asarray(Phi, float)
    y = np.asarray(y, float)
    if Phi.ndim != 2 or y.shape != (Phi.shape[0],):
        raise ValueError(f"shapes: Phi {Phi.shape}, y {y.shape}")
    if not np.all(np.abs(y) == 1.0):
        raise ValueError("labels must be +-1")
    xp = device.ops()
    P, N = Phi.shape
    G = xp.asarray(y[:, None] * Phi)
    row_scale = float(np.max(np.linalg.norm(y[:, None] * Phi, axis=1)))
    if row_scale == 0.0:
        raise NotSeparable("all patterns are zero")
    absGT = xp.abs(G).T

    W = xp.zeros(N)
    s = xp.ones(P)
    lam = xp.ones(P)
    I_N = xp.eye(N)
    for it in range(1, max_iter + 1):
        GW = G @ W
        GTlam = G.T @ lam
        r_d = W - GTlam
        r_p = GW - s - 1.0
        mu = xp.item(s @ lam) / P

        primal = 0.5 * xp.item(W @ W)
        dual = xp.item(lam.sum()) - 0.5 * xp.item(GTlam @ GTlam)
        rel_gap = abs(primal - dual) / (1.0 + primal)
        # r_d = W - G^T lam is a cancellation of terms of size |G|^T lam, so it
        # is measured on that scale (its floating-point floor).
        dual_scale = 1.0 + max(xp.max(xp.abs(W)), xp.max(absGT @ lam))
        rp_max = xp.max(xp.abs(r_p))
        if (rp_max <= tol * (1.0 + xp.max(xp.abs(GW)))
                and xp.max(xp.abs(r_d)) <= tol * dual_scale
                and rel_gap <= tol and mu <= tol * (1.0 + primal)):
            res = MaxMarginResult(W=xp.to_numpy(W), lam=xp.to_numpy(lam), iterations=it,
                                  rel_gap=rel_gap)
            _check_kkt(y[:, None] * Phi, res, tol)
            return res

        # Farkas certificate: lam_hat on the simplex with G^T lam_hat ~ 0.
        lam_hat_norm = xp.norm(GTlam) / xp.item(lam.sum())
        if lam_hat_norm <= infeas_tol * row_scale and rp_max > tol:
            raise NotSeparable(
                f"Farkas certificate |G^T lam_hat| = {lam_hat_norm:.2e} (iteration {it})")

        D = lam / s
        L = _regularised_cholesky(xp, I_N + G.T @ (D[:, None] * G))

        def newton(r_c: Any) -> tuple[Any, Any, Any]:
            rhs = -r_d - G.T @ (D * r_p) + G.T @ (r_c / s)
            dW = xp.cho_solve(L, rhs)
            dlam = D * (-r_p - G @ dW) + r_c / s
            ds = (r_c - s * dlam) / lam
            return dW, ds, dlam

        dW_a, ds_a, dlam_a = newton(-s * lam)                       # predictor
        a_aff = min(xp.max_step(s, ds_a), xp.max_step(lam, dlam_a))
        mu_aff = xp.item((s + a_aff * ds_a) @ (lam + a_aff * dlam_a)) / P
        sigma = (mu_aff / mu) ** 3
        dW, ds, dlam = newton(-s * lam - ds_a * dlam_a + sigma * mu)   # corrector
        step = min(0.99 * min(xp.max_step(s, ds), xp.max_step(lam, dlam)), 1.0)
        W = W + step * dW
        s = s + step * ds
        lam = lam + step * dlam

    raise SolverFailure(f"no convergence in {max_iter} iterations "
                        f"(rel_gap={rel_gap:.2e}, |G^T lam_hat|={lam_hat_norm:.2e})")


def _check_kkt(G: np.ndarray, res: MaxMarginResult, tol: float) -> None:
    """Certify the returned point: lam >= 0, G W >= 1, and zero duality gap (sum lam = |W|^2).

    For this convex QP these conditions are sufficient for optimality, so a
    result that passes is the max-margin solution whatever the solver did.
    """
    margins = G @ res.W
    WW = float(res.W @ res.W)
    if (np.min(res.lam) < 0.0 or np.min(margins) < 1.0 - 1e3 * tol
            or abs(float(res.lam.sum()) - WW) > 1e3 * tol * (1.0 + WW)):
        raise SolverFailure(f"KKT check failed: min lam {np.min(res.lam):.2e}, min margin "
                            f"{np.min(margins):.8f}, sum lam {res.lam.sum():.6e} vs |W|^2 {WW:.6e}")


def _regularised_cholesky(xp: Ops, M: Any) -> Any:
    """Cholesky factor of M, adding delta*I when round-off has destroyed definiteness.

    Near the optimum D = lam/s spans many decades, and M = I + G^T D G loses
    positive-definiteness in floating point although it is PD exactly. Adding a
    small multiple of the identity only perturbs the Newton direction; the
    stopping test is on the true residuals, so the accepted solution is not
    affected (dynamic regularisation, as in standard IPM codes).
    """
    n = M.shape[0]
    scale = xp.item(M.diagonal().sum()) / n
    I_n = xp.eye(n)
    for k in [None, *range(-14, -3)]:
        L = xp.cholesky(M if k is None else M + (10.0 ** k) * scale * I_n)
        if L is not None:
            return L
    raise SolverFailure("Newton matrix not factorisable even with regularisation")


def geometric_margin(Phi: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    """min_mu y_mu W.Phi_mu / |W| (negative when W misclassifies a pattern)."""
    return float(np.min(y * (Phi @ W)) / np.linalg.norm(W))


@dataclass(frozen=True)
class SlackSVMResult:
    w: np.ndarray       # (N,) weights on the original features
    xi: np.ndarray      # (P,) slack variables, y_mu w.Phi_mu >= 1 - xi_mu


def l2_slack_svm(Phi: np.ndarray, y: np.ndarray, C: float, *, tol: float = 1e-8) -> SlackSVMResult:
    """L2-slack SVM of Eq. (25) via the augmented hard margin (see module doc).

    The augmented problem has N + P features, so this is meant for moderate P.
    """
    if C <= 0:
        raise ValueError(f"C must be > 0, got {C}")
    Phi = np.asarray(Phi, float)
    y = np.asarray(y, float)
    P, N = Phi.shape
    aug = np.concatenate([Phi, np.diag(y) / np.sqrt(C)], axis=1)
    res = max_margin(aug, y, tol=tol)
    return SlackSVMResult(w=res.W[:N], xi=res.W[N:] / np.sqrt(C))
