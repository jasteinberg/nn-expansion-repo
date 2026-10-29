"""Where the dense linear algebra of the learning rules runs: NumPy (default) or torch.

The max-margin interior-point method and logistic gradient descent are written
once against the small ``Ops`` interface below. NumPy on the CPU is the default
and needs nothing beyond the base requirements. torch is optional and used in
float64 only:

* "cpu"        NumPy / SciPy (default)
* "torch-cpu"  torch on the CPU (exercises the torch code path without a GPU)
* "cuda", "cuda:k"  torch on an NVIDIA GPU

Apple "mps" is refused: it has no float64, and the interior-point Newton system
is too ill-conditioned near the optimum for float32. A device that is requested
but unavailable raises; there is no fallback to another device.

Data are generated with NumPy's RNG on the CPU for every device, so a trial sees
the same training set on every device and results agree to round-off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import expit


@dataclass(frozen=True)
class Device:
    name: str

    @staticmethod
    def cpu() -> "Device":
        return Device("cpu")

    @staticmethod
    def parse(name: str) -> "Device":
        """Validate a device name and check that it is usable here."""
        if name == "cpu":
            return Device(name)
        if name == "mps":
            raise ValueError("device 'mps' has no float64; use 'cpu' (or 'cuda' on an NVIDIA GPU)")
        if name != "torch-cpu" and not (name == "cuda" or name.startswith("cuda:")):
            raise ValueError(f"unknown device {name!r} (cpu, torch-cpu, cuda, cuda:k)")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"device {name!r} needs PyTorch (pip install torch)") from exc
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {name!r} requested but CUDA is not available")
        return Device(name)

    def ops(self) -> Ops:
        return NumpyOps() if self.name == "cpu" else TorchOps(self.name)


class Ops(Protocol):
    """Array operations used by the learning rules (float64 throughout)."""

    def asarray(self, x: np.ndarray) -> Any: ...
    def to_numpy(self, x: Any) -> np.ndarray: ...
    def item(self, x: Any) -> float: ...
    def eye(self, n: int) -> Any: ...
    def ones(self, n: int) -> Any: ...
    def zeros(self, n: int) -> Any: ...
    def abs(self, x: Any) -> Any: ...
    def max(self, x: Any) -> float: ...
    def min(self, x: Any) -> float: ...
    def norm(self, x: Any) -> float: ...
    def expit(self, x: Any) -> Any: ...
    def max_step(self, v: Any, dv: Any) -> float: ...
    def cholesky(self, M: Any) -> Any | None: ...
    def cho_solve(self, L: Any, b: Any) -> Any: ...


class NumpyOps:
    def asarray(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)

    def item(self, x: Any) -> float:
        return float(x)

    def eye(self, n: int) -> np.ndarray:
        return np.eye(n)

    def ones(self, n: int) -> np.ndarray:
        return np.ones(n)

    def zeros(self, n: int) -> np.ndarray:
        return np.zeros(n)

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def max(self, x: np.ndarray) -> float:
        return float(np.max(x))

    def min(self, x: np.ndarray) -> float:
        return float(np.min(x))

    def norm(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(x))

    def expit(self, x: np.ndarray) -> np.ndarray:
        return expit(x)

    def max_step(self, v: np.ndarray, dv: np.ndarray) -> float:
        neg = dv < 0
        return float(np.min(-v[neg] / dv[neg])) if np.any(neg) else np.inf

    def cholesky(self, M: np.ndarray) -> np.ndarray | None:
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return None

    def cho_solve(self, L: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cho_solve((L, True), b, check_finite=False)


class TorchOps:
    def __init__(self, name: str) -> None:
        import torch
        self.torch = torch
        self.dev = torch.device("cpu" if name == "torch-cpu" else name)

    def asarray(self, x: np.ndarray) -> Any:
        return self.torch.as_tensor(np.asarray(x, dtype=np.float64), device=self.dev)

    def to_numpy(self, x: Any) -> np.ndarray:
        return x.detach().cpu().numpy()

    def item(self, x: Any) -> float:
        return float(x)

    def eye(self, n: int) -> Any:
        return self.torch.eye(n, dtype=self.torch.float64, device=self.dev)

    def ones(self, n: int) -> Any:
        return self.torch.ones(n, dtype=self.torch.float64, device=self.dev)

    def zeros(self, n: int) -> Any:
        return self.torch.zeros(n, dtype=self.torch.float64, device=self.dev)

    def abs(self, x: Any) -> Any:
        return self.torch.abs(x)

    def max(self, x: Any) -> float:
        return float(self.torch.max(x))

    def min(self, x: Any) -> float:
        return float(self.torch.min(x))

    def norm(self, x: Any) -> float:
        return float(self.torch.linalg.vector_norm(x))

    def expit(self, x: Any) -> Any:
        return self.torch.sigmoid(x)

    def max_step(self, v: Any, dv: Any) -> float:
        neg = dv < 0
        return float(self.torch.min(-v[neg] / dv[neg])) if bool(neg.any()) else np.inf

    def cholesky(self, M: Any) -> Any | None:
        L, info = self.torch.linalg.cholesky_ex(M)
        return L if int(info) == 0 else None

    def cho_solve(self, L: Any, b: Any) -> Any:
        return self.torch.cholesky_solve(b[:, None], L).squeeze(1)
