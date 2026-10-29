"""Replica mean-field theory of the stochastically expanded perceptron (Sec. III, App. B-F).

Notation: alpha0 = P/N0, beta = N/N0, alpha = alpha0/beta = P/N, s = 1 + sigma_out^2,
m_bar = m/sqrt(s), Dx the standard Gaussian measure, phi/Phi its density/CDF,
H(z) = 1 - Phi(z).

Generalization error (Eq. 23 / D13), for overlap R between the teacher and the
student's N0-block,

    Eg(R) = (1/pi) [pi/2 - arctan( R / sqrt(s - R^2) )].

Pruned students use R = m / sqrt(1 - r_tilde) (Eq. 24). A student that keeps its
stochastic units sees fresh noise at test time; its field has variance
(1 - r_tilde) + sigma_in^2 r_tilde = r_Q, so its effective overlap is m/sqrt(r_Q),
which is m at sigma_in = 1 (the "replace R by m" rule of Sec. III A).

sigma_in = 1 (App. B, Eqs. B13-B18)
-----------------------------------
r_tilde = (beta-1)/beta (1-m^2), lambda = beta, r_Q = 1, and (m, kappa) solve

    1 - m^2 = 2 alpha int_{-inf}^{kappa} Dx H(-c x) (kappa - x)^2,       (B13)
    c       = alpha/(sqrt(2 pi) s) int_{-k*}^{inf} Dx (k* + x)^2 x,      (B14)

with c = m_bar/sqrt(1 - m_bar^2) and k* = kappa/sqrt(1 - m_bar^2). The integral in
(B14) is 2[phi(k*) + k* Phi(k*)] in closed form, so (B14) gives c, hence m and
kappa, as explicit functions of k*; (B13) is then one equation in k* >= 0,
solved by bracketing. At k* = 0 these are the capacity equations (C4-C5), so
the bracket exists iff alpha < alpha_c.

General sigma_in (App. B, Eqs. B5-B11)
--------------------------------------
Three equations in (kappa, m, r_tilde), with r_Q = 1 - (1 - sigma_in^2) r_tilde and
lambda from (B11). They are written term by term as in Eqs. (B5)-(B11)
(err_func_new1..3, including its sign fix), and reduce to the sigma_in = 1 case
above, which the tests check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.special import ndtr

SQRT2PI = np.sqrt(2.0 * np.pi)


class AboveCapacity(ValueError):
    """The requested alpha is at or above capacity: no max-margin solution exists."""


def _phi(x: float) -> float:
    return float(np.exp(-0.5 * x * x) / SQRT2PI)


def _H(z: float | np.ndarray) -> float | np.ndarray:
    """Gaussian tail H(z) = int_z^inf Dy."""
    return ndtr(-np.asarray(z, float))


# ------------------------------------------------------------- closed forms ----

def eg_from_overlap(R: float | np.ndarray, sigma_out: float) -> float | np.ndarray:
    """Generalization error against the noisy linear teacher, Eq. (23)/(D13)."""
    R = np.asarray(R, float)
    if np.any(np.abs(R) > 1.0 + 1e-12):
        raise ValueError(f"overlap must satisfy |R| <= 1, got max {np.max(np.abs(R))}")
    R = np.clip(R, -1.0, 1.0)              # remove only the 1e-12 round-off allowed above
    return (np.pi / 2.0 - np.arctan(R / np.sqrt(1.0 + sigma_out ** 2 - R ** 2))) / np.pi


def rtilde_sigma_in_1(m: float | np.ndarray, beta: float) -> float | np.ndarray:
    """Expanded-weight norm fraction at sigma_in = 1, Eq. (B15)."""
    return (beta - 1.0) / beta * (1.0 - np.asarray(m, float) ** 2)


def overlap_pruned_sigma_in_1(m: float | np.ndarray, beta: float) -> float | np.ndarray:
    """R = m / sqrt(1 - r_tilde) at sigma_in = 1, Eq. (B18)."""
    return np.asarray(m, float) / np.sqrt(1.0 - rtilde_sigma_in_1(m, beta))


# ----------------------------------------------------------------- capacity ----

def _capacity_residual(m_bar: float, sigma_out: float) -> float:
    """Eq. (C4) with alpha eliminated through Eq. (C5)."""
    s = 1.0 + sigma_out ** 2
    root = np.sqrt(1.0 - m_bar ** 2)
    alpha = np.pi * s * m_bar / root
    return (1.0 - s * m_bar ** 2) - (alpha / np.pi) * (np.arctan(root / m_bar) - m_bar * root)


def capacity_m_bar(sigma_out: float) -> float:
    """m_bar at capacity (kappa = 0), the root of Eqs. (C4)-(C5)."""
    hi = min(1.0 - 1e-12, 1.0 / np.sqrt(1.0 + sigma_out ** 2) - 1e-12)   # m <= 1
    return float(brentq(_capacity_residual, 1e-12, hi, args=(sigma_out,), xtol=1e-15))


def capacity_alpha(sigma_out: float) -> float:
    """alpha_c = (P/N)_c at sigma_in = 1; independent of beta (Eq. C4-C5)."""
    m_bar = capacity_m_bar(sigma_out)
    return float(np.pi * (1.0 + sigma_out ** 2) * m_bar / np.sqrt(1.0 - m_bar ** 2))


def capacity_alpha0(beta: float, sigma_out: float) -> float:
    """alpha0_c(beta) = beta alpha_c(1), Eq. (C6) / (22)."""
    return beta * capacity_alpha(sigma_out)


def eg_at_capacity(beta: float, sigma_out: float) -> float:
    """Pruned Eg at alpha0 -> alpha0_c(beta), the infimum over alpha0 (Figs. 6c,d).

    Eg decreases monotonically in alpha0 up to capacity (Sec. III A), so the
    minimum over alpha0 is the kappa -> 0 limit.
    """
    m = capacity_m_bar(sigma_out) * np.sqrt(1.0 + sigma_out ** 2)
    return float(eg_from_overlap(overlap_pruned_sigma_in_1(m, beta), sigma_out))


# --------------------------------------------------------------- large beta ----

def large_beta_R2(alpha0: float | np.ndarray, sigma_out: float) -> float | np.ndarray:
    """beta -> inf overlap, Eq. (E8): R^2 = X/(1+X), X = (2 alpha0/pi)/(1 + sigma_out^2).

    The printed Eq. (E8) shows sqrt(1 + sigma_out^2); the exact saddle at large
    beta and the published beta = inf curves both give 1/(1 + sigma_out^2) (the
    tests check the limit).
    """
    X = (2.0 * np.asarray(alpha0, float) / np.pi) / (1.0 + sigma_out ** 2)
    return X / (1.0 + X)


def large_beta_eg(alpha0: float | np.ndarray, sigma_out: float) -> float | np.ndarray:
    """beta -> inf pruned generalization error (the beta = inf curves of Fig. 6)."""
    return eg_from_overlap(np.sqrt(large_beta_R2(alpha0, sigma_out)), sigma_out)


# ------------------------------------------------- sigma_in = 1 saddle point ----

@dataclass(frozen=True)
class SaddleSigmaIn1:
    m: float
    kappa: float


def _J(k: float) -> float:
    """int_{-k}^{inf} Dx (k + x)^2 x = 2 [phi(k) + k Phi(k)]."""
    return 2.0 * (_phi(k) + k * float(ndtr(k)))


def _I_b13(c: float, kappa: float) -> float:
    """int_{-inf}^{kappa} Dx H(-c x) (kappa - x)^2, the integral of Eq. (B13)."""
    val, _ = quad(lambda x: _phi(x) * float(_H(-c * x)) * (kappa - x) ** 2,
                  -np.inf, kappa, epsabs=1e-13, epsrel=1e-12, limit=200)
    return float(val)


def _from_kstar(t: float, alpha: float, s: float) -> tuple[float, float, float]:
    """(c, m, kappa) as functions of k* = t through Eq. (B14)."""
    c = alpha * _J(t) / (SQRT2PI * s)
    m = c / np.sqrt(1.0 + c * c) * np.sqrt(s)
    kappa = t / np.sqrt(1.0 + c * c)
    return c, m, kappa


def _b13_residual(t: float, alpha: float, s: float) -> float:
    c, m, kappa = _from_kstar(t, alpha, s)
    return 1.0 - m * m - 2.0 * alpha * _I_b13(c, kappa)


def saddle_sigma_in_1(alpha: float, sigma_out: float) -> SaddleSigmaIn1:
    """Max-margin order parameters (m, kappa) at sigma_in = 1, alpha = alpha0/beta.

    Raises AboveCapacity for alpha >= alpha_c(sigma_out).
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    s = 1.0 + sigma_out ** 2
    if _b13_residual(0.0, alpha, s) <= 0.0:
        raise AboveCapacity(f"alpha={alpha} >= alpha_c={capacity_alpha(sigma_out)}")
    hi = 1.0
    while _b13_residual(hi, alpha, s) > 0.0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("could not bracket the B13 root")
    t = brentq(_b13_residual, 0.0, hi, args=(alpha, s), xtol=1e-14, rtol=1e-13)
    _, m, kappa = _from_kstar(t, alpha, s)
    res = _b13_residual(t, alpha, s)
    if abs(res) > 1e-9 or not (0.0 < m <= 1.0) or kappa < 0.0:
        raise RuntimeError(f"sigma_in=1 saddle out of range at alpha={alpha}: residual {res:.1e}, "
                           f"m={m}, kappa={kappa}")
    return SaddleSigmaIn1(m=float(m), kappa=float(kappa))


def m_curve_sigma_in_1(alpha0: np.ndarray, beta: float, sigma_out: float) -> np.ndarray:
    """Replica overlap m(alpha0) at fixed beta, sigma_in = 1.

    Entries with alpha0 <= 0 or alpha0 >= alpha0_c(beta) lie outside the domain of
    the max-margin theory and are returned as NaN; every other entry is solved.
    """
    alpha0 = np.asarray(alpha0, float)
    out = np.full(alpha0.shape, np.nan)
    a_c = capacity_alpha0(beta, sigma_out)
    for i, a0 in np.ndenumerate(alpha0):
        if 0.0 < a0 < a_c:
            out[i] = saddle_sigma_in_1(a0 / beta, sigma_out).m
    return out


def R_curve_sigma_in_1(alpha0: np.ndarray, beta: float, sigma_out: float) -> np.ndarray:
    """Replica pruned overlap R(alpha0), sigma_in = 1 (Eq. B18); NaN outside (0, alpha0_c)."""
    return np.asarray(overlap_pruned_sigma_in_1(m_curve_sigma_in_1(alpha0, beta, sigma_out), beta))


def eg_curve_sigma_in_1(alpha0: np.ndarray, beta: float, sigma_out: float,
                        pruned: bool) -> np.ndarray:
    """Replica Eg(alpha0) at fixed beta, sigma_in = 1 (Fig. 6a,b); NaN outside (0, alpha0_c).

    pruned: Eg(R) with R from Eq. (B18); otherwise Eg(m) (stochastic units kept).
    """
    m = m_curve_sigma_in_1(alpha0, beta, sigma_out)
    return np.asarray(eg_from_overlap(overlap_pruned_sigma_in_1(m, beta) if pruned else m, sigma_out))


# ---------------------------------------------------- general sigma_in saddle ----

@dataclass(frozen=True)
class SaddleGeneral:
    kappa: float
    m: float
    rtilde: float
    sigma_in: float

    @property
    def rQ(self) -> float:
        return 1.0 - (1.0 - self.sigma_in ** 2) * self.rtilde

    @property
    def R_pruned(self) -> float:
        """Eq. (24)/(B12)."""
        return self.m / np.sqrt(1.0 - self.rtilde)

    @property
    def R_unpruned(self) -> float:
        """Effective overlap when the stochastic units are kept at test time."""
        return self.m / np.sqrt(self.rQ)


def _general_residuals(v: np.ndarray, alpha0: float, beta: float, sigma_out: float,
                       sigma_in: float) -> np.ndarray:
    """Residuals of the kappa, m and r_tilde equations, Eqs. (B5)-(B9), (B11)."""
    kappa, m, rt = (float(x) for x in v)
    aa = sigma_in ** 2
    s = 1.0 + sigma_out ** 2
    alpha = alpha0 / beta
    rQ = 1.0 - (1.0 - aa) * rt
    mb = m / np.sqrt(s)
    q = rQ - mb * mb                                   # r_Q - m_bar^2
    lam = 1.0 + np.sqrt(aa * (beta - 1.0) * rt / (1.0 - rt - m * m))     # (B11)
    sq = np.sqrt(rQ)
    u = kappa / sq                                     # theta(kappa - sqrt(rQ) x) -> x < u

    def Hc(x: float) -> float:
        return float(_H(-mb * x / np.sqrt(q)))

    I1, _ = quad(lambda x: _phi(x) * Hc(x) * (kappa - sq * x) ** 2, -np.inf, u,
                 epsabs=1e-13, epsrel=1e-12, limit=200)
    I3, _ = quad(lambda x: _phi(x) * Hc(x) * x * (kappa - sq * x), -np.inf, u,
                 epsabs=1e-13, epsrel=1e-12, limit=200)
    kbar = kappa / np.sqrt(q)

    e1 = (lam / beta) * (1.0 - rt - m * m) + aa * rt * (beta - 1.0) / beta * lam / (lam - 1.0) \
        - 2.0 * alpha * I1
    e2 = lam * mb / np.sqrt(q) - alpha * beta / (SQRT2PI * s) * _J(kbar)
    e3 = (1.0 / beta) * (lam / (lam - 1.0) * (beta - 1.0) * aa - lam) \
        + 2.0 * alpha * (aa - 1.0) / sq * I3 \
        - (aa - 1.0) * s * lam * mb * mb / (beta * rQ)
    return np.array([e1, e2, e3])


def saddle_general(alpha0: float, beta: float, sigma_out: float, sigma_in: float,
                   guess: SaddleGeneral | SaddleSigmaIn1, *, tol: float = 1e-10) -> SaddleGeneral:
    """Solve the general-sigma_in saddle from an initial guess (use continuation).

    ``guess`` is a nearby solution, e.g. the sigma_in = 1 saddle at the same
    (alpha0, beta) or the previous point along a sigma_in or alpha0 sweep.
    Raises RuntimeError if the residual does not fall below ``tol``.
    """
    if beta <= 1.0:
        raise ValueError("the general-sigma_in theory needs beta > 1 (expanded units)")
    if isinstance(guess, SaddleSigmaIn1):
        x0 = np.array([guess.kappa, guess.m, rtilde_sigma_in_1(guess.m, beta)])
    else:
        x0 = np.array([guess.kappa, guess.m, guess.rtilde])

    def fun(v: np.ndarray) -> np.ndarray:
        return _general_residuals(v, alpha0, beta, sigma_out, sigma_in)

    eps = 1e-9
    lo = np.array([0.0, eps, eps])
    hi = np.array([np.inf, 1.0 - eps, 1.0 - eps])
    sol = least_squares(fun, np.clip(x0, lo + eps, hi - eps), bounds=(lo, hi),
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    kappa, m, rt = sol.x
    res = float(np.max(np.abs(fun(sol.x))))
    if res > tol or 1.0 - rt - m * m <= 0.0:
        raise RuntimeError(f"general saddle not solved: residual {res:.2e} at "
                           f"alpha0={alpha0}, beta={beta}, sigma_in={sigma_in}")
    return SaddleGeneral(kappa=float(kappa), m=float(m), rtilde=float(rt),
                         sigma_in=float(sigma_in))


def saddle_sigma_in_sweep(alpha0: float, beta: float, sigma_out: float,
                          sigma_in_values: np.ndarray, *, n_sub: int = 4) -> list[SaddleGeneral]:
    """General saddle along a sigma_in sweep, continued from the exact sigma_in = 1 point.

    The sweep is traversed outward from sigma_in = 1 in both directions with
    ``n_sub`` sub-steps between requested values, so each solve starts next to
    the solution.
    """
    vals = np.asarray(sigma_in_values, float)
    start = saddle_sigma_in_1(alpha0 / beta, sigma_out)
    out: dict[float, SaddleGeneral] = {}
    below = np.sort(vals[vals <= 1.0])[::-1]          # 1 -> down
    above = np.sort(vals[vals > 1.0])                 # 1 -> up
    for targets in (below, above):
        guess: SaddleGeneral | SaddleSigmaIn1 = start
        prev = 1.0
        for target in targets:
            for si in np.linspace(prev, target, n_sub + 1)[1:]:
                guess = saddle_general(alpha0, beta, sigma_out, float(si), guess)
            out[float(target)] = guess            # type: ignore[assignment]
            prev = float(target)
    return [out[float(v)] for v in vals]


def saddle_alpha0_sweep(alpha0_values: np.ndarray, beta: float, sigma_out: float,
                        sigma_in: float, *, n_sub: int = 4) -> list[SaddleGeneral]:
    """General saddle along an alpha0 sweep at fixed sigma_in (Fig. 7a theory).

    Each alpha0 is reached by continuation in sigma_in from its exact sigma_in = 1
    solution, so the points are solved independently of each other.
    """
    return [saddle_sigma_in_sweep(float(a0), beta, sigma_out, np.array([sigma_in]),
                                  n_sub=n_sub)[0] for a0 in np.asarray(alpha0_values, float)]


def optimal_sigma_in(alpha0: float, beta: float, sigma_out: float,
                     bounds: tuple[float, float]) -> tuple[float, float]:
    """sigma_in minimising the pruned Eg at fixed (alpha0, beta) (App. F, Fig. 7b).

    Returns (sigma_in*, Eg*). Each evaluation continues from sigma_in = 1.
    """
    def eg(si: float) -> float:
        sp = saddle_sigma_in_sweep(alpha0, beta, sigma_out, np.array([si]))[0]
        return float(eg_from_overlap(sp.R_pruned, sigma_out))

    res = minimize_scalar(eg, bounds=bounds, method="bounded", options={"xatol": 1e-6})
    if not res.success:
        raise RuntimeError(f"optimal sigma_in search failed: {res.message}")
    return float(res.x), float(res.fun)
