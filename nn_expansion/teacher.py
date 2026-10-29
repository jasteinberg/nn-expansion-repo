"""Teacher rules (Eqs. 1, 31) and the noise floor (Eq. 4).

One teacher family covers every figure. The teacher field is

    h0(x) = a (1/sqrt(N0)) w0.x + (1-a) (1/N0) x^T W0 x + sigma_out eps,

with x_i, eps iid N(0,1), w0 and W0 iid N(0,1) directions with |w0|^2 = N0 and
|W0|_F^2 = N0^2 (see ``draw_teacher``), and labels y0 = sgn(h0).

* a = 1: the noisy linear teacher of Eq. (1); the signal term has unit variance,
  so sigma_out is the label noise relative to the signal.
* a in (0,1), sigma_out = 0: the quadratic teacher of Eq. (31) (Figs. 9-10). For a
  linear student it acts like a noisy linear teacher with sigma_out = (1-a)/a
  (Eq. 32), because the quadratic term is unit-variance noise uncorrelated with
  the linear one, rescaled by a.
* a = 0, sigma_out > 0: the noisy quadratic-kernel teacher of Fig. 11.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Teacher:
    """Mixed linear/quadratic teacher with Gaussian label noise (Eqs. 1, 31)."""

    w0: np.ndarray            # (N0,) linear weights
    W0: np.ndarray | None     # (N0, N0) quadratic weights; None when a == 1
    a: float                  # linear fraction
    sigma_out: float          # label-noise std

    @property
    def N0(self) -> int:
        return self.w0.shape[0]

    @property
    def is_noisy_linear(self) -> bool:
        return self.a == 1.0

    def clean_field(self, X: np.ndarray) -> np.ndarray:
        """Noise-free field for inputs X (P, N0) -> (P,)."""
        if X.ndim != 2 or X.shape[1] != self.N0:
            raise ValueError(f"X must be (P, {self.N0}), got {X.shape}")
        h = self.a * (X @ self.w0) / np.sqrt(self.N0)
        if self.a != 1.0:
            if self.W0 is None:
                raise ValueError("a quadratic teacher (a < 1) needs W0")
            h = h + (1.0 - self.a) * np.einsum("pi,ij,pj->p", X, self.W0, X) / self.N0
        return h

    def labels(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """y0 = sgn(h0) in {-1,+1} with fresh label noise, (P,)."""
        h = self.clean_field(X)
        if self.sigma_out > 0:
            h = h + self.sigma_out * rng.standard_normal(h.shape[0])
        return _sign(h)


def draw_teacher(N0: int, a: float, sigma_out: float, rng: np.random.Generator) -> Teacher:
    """Draw teacher weights with iid N(0,1) directions and fixed norms.

    |w0|^2 = N0 and |W0|_F^2 = N0^2 (Eq. D7, as in the published runs). Then the linear signal has unit variance for every
    teacher, so sigma_out is the noise-to-signal ratio exactly and Eq. (23) with
    the nominal sigma_out is exact; the quadratic form x^T W0 x / N0 has
    variance 1 + O(1/N0) and a teacher-specific mean tr(W0)/N0 = O(1/sqrt(N0)).
    """
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"a must lie in [0,1], got {a}")
    if sigma_out < 0:
        raise ValueError(f"sigma_out must be >= 0, got {sigma_out}")
    w0 = rng.standard_normal(N0)
    w0 *= np.sqrt(N0) / np.linalg.norm(w0)
    W0 = None
    if a < 1.0:
        W0 = rng.standard_normal((N0, N0))
        W0 *= N0 / np.linalg.norm(W0)
    return Teacher(w0=w0, W0=W0, a=float(a), sigma_out=float(sigma_out))


def draw_inputs(P: int, N0: int, rng: np.random.Generator) -> np.ndarray:
    """Training or test inputs x ~ N(0, I_N0), (P, N0)."""
    return rng.standard_normal((P, N0))


def quad_to_noise_sigma(a: float) -> float:
    """Effective label-noise std of the quadratic teacher, Eq. (32): (1-a)/a."""
    return (1.0 - a) / a


def min_generalization_error(sigma_out: float | np.ndarray) -> float | np.ndarray:
    """Noise floor E_min, Eq. (4): (1/pi)(pi/2 - arctan(1/sigma_out))."""
    return (np.pi / 2.0 - np.arctan(1.0 / np.asarray(sigma_out, float))) / np.pi


def _sign(h: np.ndarray) -> np.ndarray:
    """sgn with sgn(0) := +1 (a measure-zero event for continuous fields)."""
    return np.where(h >= 0.0, 1.0, -1.0)
