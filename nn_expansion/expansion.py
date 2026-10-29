"""Input-layer expansions (Eqs. 5-6, 10) and the quadratic feature map (Fig. 11).

A student of width N0 is widened to N = beta N0 = N0 + N_plus.

* Deterministic (Eqs. 5-6, Fig. 1c): a fixed random map J (N_plus x N0),
  J_ji ~ N(0, 1/N0), feeds threshold units

      z_j = A / sqrt(f(1-f)) [ Theta((J x)_j - T) - f ].

  T is set from the training inputs as for the published figures: sort the
  preactivations of each pattern over units, average the order statistics over
  patterns, and take the one at index round((1-f) N_plus). Each pattern then
  activates a fraction ~f of the units, and the units have mean ~0 and
  variance ~A^2, so A plays the role of sigma_in (Sec. III B).

* Stochastic (Eq. 10, Fig. 1d): x_tilde ~ N(0, sigma_in^2), drawn afresh for
  every pattern, independent of x.

The student field is h = (1/sqrt(N)) W.[x, u] (Eqs. 5, 10); ``design_matrix``
returns the rows [x, u]/sqrt(N) so that h = Phi @ W.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeterministicExpansion:
    """Fixed threshold layer of Eqs. (5)-(6)."""

    J: np.ndarray     # (N_plus, N0), entries N(0, 1/N0)
    T: float          # firing threshold
    f: float          # target sparsity (fraction of active units)
    A: float          # activity scale (== sigma_in)

    @property
    def N_plus(self) -> int:
        return self.J.shape[0]

    def preactivation(self, X: np.ndarray) -> np.ndarray:
        """(J x)_j for every pattern, (P, N_plus)."""
        return X @ self.J.T

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Hidden activities z (P, N_plus) of Eq. (6)."""
        active = (self.preactivation(X) > self.T).astype(float)
        Z = self.A / np.sqrt(self.f * (1.0 - self.f)) * (active - self.f)
        if Z.shape != (X.shape[0], self.N_plus):
            raise AssertionError(f"hidden activities have shape {Z.shape}")
        return Z


def fit_deterministic_expansion(X_train: np.ndarray, N_plus: int, f: float, A: float,
                                rng: np.random.Generator) -> DeterministicExpansion:
    """Draw J and set T from the training inputs (rule of the published runs, see module doc)."""
    if not 0.0 < f < 1.0:
        raise ValueError(f"sparsity f must lie in (0,1), got {f}")
    if N_plus < 1:
        raise ValueError(f"N_plus must be >= 1, got {N_plus}")
    N0 = X_train.shape[1]
    J = rng.standard_normal((N_plus, N0)) / np.sqrt(N0)
    H = X_train @ J.T                                     # (P, N_plus)
    mean_order_stats = np.sort(H, axis=1).mean(axis=0)    # ascending, (N_plus,)
    k = int(np.floor((1.0 - f) * N_plus + 0.5))           # round half up, 1-indexed
    if not 1 <= k <= N_plus:
        raise ValueError(f"threshold index {k} outside 1..{N_plus} for f={f}")
    T = float(mean_order_stats[k - 1])
    if not np.isfinite(T):
        raise ValueError(f"non-finite threshold T={T}")
    active = float(np.mean(H > T))
    if not 0.0 < active < 1.0:
        raise ValueError(f"hidden layer {'dead' if active == 0 else 'saturated'} on the training set "
                         f"(f={f}, N_plus={N_plus}, fraction active {active})")
    return DeterministicExpansion(J=J, T=T, f=float(f), A=float(A))


def stochastic_units(P: int, N_plus: int, sigma_in: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Per-pattern random units x_tilde ~ N(0, sigma_in^2), (P, N_plus), Eq. (10)."""
    return sigma_in * rng.standard_normal((P, N_plus))


def design_matrix(X: np.ndarray, U: np.ndarray | None) -> np.ndarray:
    """Rows [x, u] / sqrt(N), (P, N), so that the student field is h = Phi @ W."""
    Z = X if U is None else np.concatenate([X, U], axis=1)
    return Z / np.sqrt(Z.shape[1])


def quadratic_features(Z: np.ndarray) -> np.ndarray:
    """Quadratic-kernel features vec(z z^T) / N for rows z of Z (P, N) -> (P, N^2).

    K(z, z') = (z.z')^2 / N^2, the feature map used for the published Fig. 11
    (Fig. 11). The 1/N scale only conditions the solver; the max-margin
    direction does not depend on it.
    """
    P, N = Z.shape
    return np.einsum("pi,pj->pij", Z, Z).reshape(P, N * N) / N
