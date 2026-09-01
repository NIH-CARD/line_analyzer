"""Array-namespace resolution, so one kernel runs on CPU or GPU unchanged.

The pipeline's inner kernel is memory-bound at ~1.3 FLOP/byte (see `bench.py`),
so nothing here tries to be clever about arithmetic. What matters is:

  * keep the data RESIDENT on the device across a whole permutation sweep, so
    the PCIe transfer is paid once rather than per design;
  * move as few bytes as possible, which is what the float32 option buys;
  * chunk over proteins when the working set exceeds device memory, rather than
    failing.

Backends: `numpy` (always), `cupy`, `torch` (CPU or CUDA). The three expose
different spellings for the same operations, so this module normalises them to
the handful the kernels need. Everything degrades to numpy when no GPU is
present, and `tests.py` checks the accelerated path against the reference loop
on whatever backend is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Backend:
    name: str
    label: str
    xp: Any                       # array namespace
    asarray: Callable             # host -> device
    to_numpy: Callable            # device -> host
    sync: Callable                # block until queued work completes
    free_bytes: Callable          # device memory available, or None on CPU
    is_gpu: bool

    def __repr__(self):
        return f"<Backend {self.label}>"


def _numpy_backend() -> Backend:
    return Backend(
        name="numpy", label="numpy (CPU)", xp=np,
        asarray=lambda a, dtype=None: np.asarray(a, dtype=dtype),
        to_numpy=np.asarray, sync=lambda: None,
        free_bytes=lambda: None, is_gpu=False)


def _cupy_backend() -> Backend:
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]

    def free():
        return int(cp.cuda.Device().mem_info[0])

    return Backend(
        name="cupy", label=f"cupy ({name})", xp=cp,
        asarray=lambda a, dtype=None: cp.asarray(a, dtype=dtype),
        to_numpy=cp.asnumpy,
        sync=cp.cuda.Stream.null.synchronize,
        free_bytes=free, is_gpu=True)


def _torch_backend(device: str = "cuda") -> Backend:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    is_gpu = device.startswith("cuda")
    label = f"torch {torch.__version__} on {device}"
    if is_gpu:
        label += f" ({torch.cuda.get_device_name(0)})"

    _DT = {np.dtype("float64"): torch.float64, np.dtype("float32"): torch.float32}

    class _NS:
        """Just enough of the numpy surface for the kernels in accel.py."""
        @staticmethod
        def sqrt(a):
            return torch.sqrt(a)

        @staticmethod
        def einsum(spec, *ops):
            return torch.einsum(spec, *ops)

        @staticmethod
        def stack(seq, axis=0):
            return torch.stack(list(seq), dim=axis)

        @staticmethod
        def empty(shape, dtype=None):
            return torch.empty(shape, device=device,
                               dtype=_DT.get(np.dtype(dtype or "float64")))

        @staticmethod
        def zeros(shape, dtype=None):
            return torch.zeros(shape, device=device,
                               dtype=_DT.get(np.dtype(dtype or "float64")))

        class linalg:
            @staticmethod
            def solve(a, b):
                return torch.linalg.solve(a, b)

    def conv(a, dtype=None):
        return torch.as_tensor(np.asarray(a), device=device,
                               dtype=_DT.get(np.dtype(dtype or "float64")))

    def free():
        if not is_gpu:
            return None
        f, _ = torch.cuda.mem_get_info()
        return int(f)

    return Backend(
        name="torch", label=label, xp=_NS, asarray=conv,
        to_numpy=lambda a: a.detach().cpu().numpy(),
        sync=(torch.cuda.synchronize if is_gpu else (lambda: None)),
        free_bytes=free, is_gpu=is_gpu)


def resolve(name: str = "auto", device: str = "cuda") -> Backend:
    """Pick a backend. 'auto' takes the fastest available without complaining."""
    if name == "auto":
        for candidate in ("cupy", "torch"):
            try:
                b = _cupy_backend() if candidate == "cupy" else _torch_backend(device)
                if b.is_gpu:
                    return b
            except Exception:
                continue
        return _numpy_backend()
    if name == "numpy":
        return _numpy_backend()
    if name == "cupy":
        return _cupy_backend()
    if name == "torch":
        return _torch_backend(device)
    raise ValueError(f"unknown backend {name!r}")


def plan_chunks(n_proteins: int, n_clones: int, n_params: int, backend: Backend,
                dtype=np.float64, n_designs: int = 1, safety: float = 0.6) -> int:
    """Proteins per chunk that will fit in device memory.

    Working set per protein: means + weights + residuals + the p x p normal
    matrix and its right-hand side, plus slack for temporaries that einsum
    materialises. On CPU there is no hard limit, so the whole thing goes in one
    chunk unless a caller asks otherwise.
    """
    if not backend.is_gpu:
        return n_proteins
    free = backend.free_bytes()
    if not free:
        return n_proteins
    w = np.dtype(dtype).itemsize
    per_protein = w * (3 * n_clones + n_clones * n_params
                       + n_params * n_params + 2 * n_params
                       + max(n_designs, 1))
    per_protein = int(per_protein * 2.0)          # temporaries
    budget = int(free * safety)
    chunk = max(1024, budget // max(per_protein, 1))
    return int(min(chunk, n_proteins))
