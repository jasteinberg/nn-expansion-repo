"""One simulated trial: draw a teacher and a training set, expand, train, measure.

Every simulation figure (2-4, 7-14) is a sweep of ``run_trial`` over a grid of
``TrialSpec``s, so the teacher / expansion / learner / measurement logic exists
once.

Measurements
------------
* R_pruned: cosine between the student's N0-block and w0 (Eq. 24/D8).
* Eg_pruned, Eg_unpruned: generalization error of Eq. (2). When the student is
  linear in x and the teacher is the noisy linear teacher, Eg depends on the
  trained weights only through an overlap and is computed exactly with Eq. (23):
  pruned -> R_pruned; stochastic unpruned -> w.w0 / (|w0| sqrt(|w|^2 + sigma_in^2
  |w_tilde|^2)), since the fresh test units are Gaussian. Otherwise (deterministic
  unpruned, quadratic teacher, kernel student) Eg is a Monte-Carlo average over
  ``n_test`` fresh inputs with fresh label noise, as for the published figures.
* kappa: min_mu y_mu h_mu / |W| with h the field of unit-variance inputs (Eq. 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import Device
from .expansion import (DeterministicExpansion, design_matrix, fit_deterministic_expansion,
                        quadratic_features, stochastic_units)
from .logistic import effective_eta, logistic_gd
from .svm import NotSeparable, geometric_margin, max_margin
from .teacher import Teacher, draw_inputs, draw_teacher
from .theory import eg_from_overlap


# ---------------------------------------------------------------- specs ----

@dataclass(frozen=True)
class NoExpansion:
    pass


@dataclass(frozen=True)
class Stochastic:
    beta: float
    sigma_in: float


@dataclass(frozen=True)
class Deterministic:
    beta: float
    f: float
    A: float


Expansion = NoExpansion | Stochastic | Deterministic


@dataclass(frozen=True)
class MaxMargin:
    pass


@dataclass(frozen=True)
class Logistic:
    eta: float
    eta_unit: str                # "paper" (Eq. 39 update) | "published" (step x sqrt(N)), see logistic.py
    epochs: int
    init_scale: float            # W_init = init_scale * N(0, 1)^N / sqrt(N) (published runs: 0.1)
    stop_tol: float              # stop once |step| < stop_tol (published runs: 1e-6)
    record_every: int | None     # record R(t), kappa(t) every this many epochs


Learner = MaxMargin | Logistic


@dataclass(frozen=True)
class TrialSpec:
    N0: int
    alpha0: float
    teacher_a: float               # 1 = noisy linear; <1 = quadratic part (Eq. 31)
    sigma_out: float
    expansion: Expansion
    learner: Learner
    student: str                   # "linear" | "quadratic_kernel"
    measure_unpruned: bool
    n_test: int                    # Monte-Carlo test-set size (only used when needed)

    def __post_init__(self) -> None:
        if self.student not in ("linear", "quadratic_kernel"):
            raise ValueError(f"unknown student {self.student!r}")
        if self.student == "quadratic_kernel" and isinstance(self.expansion, Deterministic):
            raise ValueError("the kernel student is only defined with stochastic expansion")

    @property
    def beta(self) -> float:
        return 1.0 if isinstance(self.expansion, NoExpansion) else self.expansion.beta

    @property
    def N(self) -> int:
        return int(round(self.beta * self.N0))

    @property
    def P(self) -> int:
        return int(round(self.alpha0 * self.N0))


@dataclass(frozen=True)
class TrialResult:
    separable: bool
    R_pruned: float = np.nan
    eg_pruned: float = np.nan
    eg_unpruned: float = np.nan
    kappa: float = np.nan
    trace_epochs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trace_R: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trace_kappa: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ------------------------------------------------------------- features ----

def _expanded_units(spec: TrialSpec, X: np.ndarray, rng: np.random.Generator,
                    det: DeterministicExpansion | None) -> np.ndarray | None:
    """Expanded units for inputs X (training or test); None without expansion."""
    n_plus = spec.N - spec.N0
    if isinstance(spec.expansion, NoExpansion):
        return None
    if isinstance(spec.expansion, Stochastic):
        return stochastic_units(X.shape[0], n_plus, spec.expansion.sigma_in, rng)
    assert det is not None
    return det.transform(X)


def _features(spec: TrialSpec, X: np.ndarray, U: np.ndarray | None) -> np.ndarray:
    if spec.student == "linear":
        return design_matrix(X, U)
    Z = X if U is None else np.concatenate([X, U], axis=1)
    return quadratic_features(Z)


# ---------------------------------------------------------------- trial ----

def run_trial(spec: TrialSpec, rng: np.random.Generator, device: Device = Device.cpu()) -> TrialResult:
    """Simulate one trial of ``spec`` with randomness from ``rng``.

    ``device`` runs the learning rule (nn_expansion.backend); data are always drawn on the CPU.
    """
    teacher = draw_teacher(spec.N0, spec.teacher_a, spec.sigma_out, rng)
    X = draw_inputs(spec.P, spec.N0, rng)
    y = teacher.labels(X, rng)
    det = None
    if isinstance(spec.expansion, Deterministic):
        det = fit_deterministic_expansion(X, spec.N - spec.N0, spec.expansion.f,
                                          spec.expansion.A, rng)
    U = _expanded_units(spec, X, rng, det)
    Phi = _features(spec, X, U)
    n_features = spec.N if spec.student == "linear" else spec.N ** 2
    if X.shape != (spec.P, spec.N0) or Phi.shape != (spec.P, n_features) or y.shape != (spec.P,):
        raise AssertionError(f"shapes: X {X.shape}, Phi {Phi.shape}, y {y.shape} for P={spec.P}, "
                             f"N0={spec.N0}, features={n_features}")
    if not np.all(np.abs(y) == 1.0):
        raise AssertionError("labels must be +-1")

    empty = np.zeros(0)
    trace_epochs, trace_R, trace_kappa = empty, empty, empty
    if isinstance(spec.learner, MaxMargin):
        try:
            W = max_margin(Phi, y, device=device).W
        except NotSeparable:
            return TrialResult(separable=False)
        if np.min(y * (Phi @ W)) <= 0.0:
            raise AssertionError("max-margin solution misclassifies a training pattern")
    else:
        lg = spec.learner
        rec = None
        if lg.record_every is not None:
            rec = lambda W: (overlap(W[:spec.N0], teacher.w0),          # noqa: E731
                             geometric_margin(Phi, y, W) * np.sqrt(spec.N))
        W_init = lg.init_scale * rng.standard_normal(Phi.shape[1]) / np.sqrt(Phi.shape[1])
        W, ts, vals = logistic_gd(Phi, y, effective_eta(lg.eta, lg.eta_unit, spec.N), lg.epochs,
                                  W_init, lg.stop_tol, lg.record_every, rec, device)
        if rec is not None:
            arr = np.asarray(vals)
            trace_epochs, trace_R, trace_kappa = ts, arr[:, 0], arr[:, 1]

    linear = spec.student == "linear"
    result = TrialResult(
        separable=True,
        R_pruned=overlap(W[:spec.N0], teacher.w0) if linear else np.nan,
        kappa=geometric_margin(Phi, y, W) * np.sqrt(spec.N) if linear else np.nan,
        eg_pruned=_eg(spec, teacher, W, rng, pruned=True, det=det),
        eg_unpruned=_eg(spec, teacher, W, rng, pruned=False, det=det) if spec.measure_unpruned else np.nan,
        trace_epochs=trace_epochs, trace_R=trace_R, trace_kappa=trace_kappa)
    _check_result(result, spec)
    return result


def _check_result(r: TrialResult, spec: TrialSpec) -> None:
    """Measured values exist where they should and lie in their ranges."""
    measured = {"eg_pruned": True, "eg_unpruned": spec.measure_unpruned,
                "R_pruned": spec.student == "linear", "kappa": spec.student == "linear"}
    for name, expected in measured.items():
        v = getattr(r, name)
        if expected != bool(np.isfinite(v)):
            raise AssertionError(f"{name} = {v} but measured={expected}")
    if not 0.0 <= r.eg_pruned <= 1.0 or (spec.measure_unpruned and not 0.0 <= r.eg_unpruned <= 1.0):
        raise AssertionError(f"error rate outside [0, 1]: {r.eg_pruned}, {r.eg_unpruned}")
    if spec.student == "linear" and not -1.0 - 1e-12 <= r.R_pruned <= 1.0 + 1e-12:
        raise AssertionError(f"overlap outside [-1, 1]: {r.R_pruned}")
    if isinstance(spec.learner, MaxMargin) and spec.student == "linear" and r.kappa <= 0.0:
        raise AssertionError(f"max-margin solution with non-positive margin {r.kappa}")


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two weight vectors (R of Eq. 24 for a = student N0-block, b = w0)."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _eg(spec: TrialSpec, teacher: Teacher, W: np.ndarray, rng: np.random.Generator,
        pruned: bool, det: DeterministicExpansion | None) -> float:
    """Generalization error of Eq. (2), exact where possible (see module doc)."""
    N0 = spec.N0
    w, w_tilde = W[:N0], W[N0:]
    exact_ok = spec.student == "linear" and teacher.is_noisy_linear
    if exact_ok and (pruned or isinstance(spec.expansion, NoExpansion)):
        return float(eg_from_overlap(overlap(w, teacher.w0), spec.sigma_out))
    if exact_ok and isinstance(spec.expansion, Stochastic):
        si = spec.expansion.sigma_in
        R_eff = float(w @ teacher.w0) / (np.linalg.norm(teacher.w0)
                                         * np.sqrt(w @ w + si ** 2 * (w_tilde @ w_tilde)))
        return float(eg_from_overlap(R_eff, spec.sigma_out))
    # Monte Carlo against fresh inputs and fresh label noise
    Xt = draw_inputs(spec.n_test, N0, rng)
    yt = teacher.labels(Xt, rng)
    if pruned or isinstance(spec.expansion, NoExpansion):
        Ut = None if isinstance(spec.expansion, NoExpansion) else np.zeros((spec.n_test, spec.N - N0))
    else:
        Ut = _expanded_units(spec, Xt, rng, det)
    pred = np.where(_features(spec, Xt, Ut) @ W >= 0.0, 1.0, -1.0)
    return float(np.mean(pred != yt))
