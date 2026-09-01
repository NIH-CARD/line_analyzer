"""Scaling benchmark for the clone-level fit: CPU now, GPU if you have one.

    python -m proteomics_revertant.bench                     # CPU sweep
    python -m proteomics_revertant.bench --backend cupy      # on a CUDA box
    python -m proteomics_revertant.bench --backend torch --device cuda

The kernel benchmarked here is the one `fastfit.fit_batched` actually runs: a
weighted least squares of `means` (proteins x clones) on a design (clones x
parameters), plus the residual variance and the contrast standard errors. That
is the whole per-protein computation; everything else in the pipeline is either
a scalar reduction over proteins or file I/O.

Three shapes are timed:

  loop      one `np.linalg.lstsq` per protein -- what `consensus_icc` still does
  batched   the same arithmetic as stacked einsum + solve -- what `fastfit` does
  sweep     P design matrices against one resident copy of the data, which is
            the permutation test's shape and the only part with enough
            parallel work to interest a GPU

For each configuration it reports achieved GFLOP/s, bytes moved, and arithmetic
intensity in FLOP/byte. Arithmetic intensity is the number that decides whether
a GPU can help: below roughly 10 FLOP/byte you are memory-bound, and a
memory-bound kernel only wins on a GPU if the data is already resident there.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


# ---------------------------------------------------------------------------
# work counting
# ---------------------------------------------------------------------------


def flop_count(n, c, p, n_designs=1):
    """FLOPs for one batched weighted fit of n proteins, c clones, p parameters."""
    per_protein = (
        2 * c * p            # Xw = X * w
        + 2 * c * p * p      # X'WX
        + 2 * c * p          # X'Wy
        + (2 * p ** 3) // 3  # solve
        + 2 * c * p          # fitted values
        + 3 * c              # residuals, weighted SS
    )
    return n * per_protein * n_designs


def byte_count(n, c, p, dtype=np.float64, n_designs=1):
    """Bytes touched: means + weights read per design, coefficients written."""
    w = np.dtype(dtype).itemsize
    return n_designs * (2 * n * c * w + n * p * w)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def fit_loop(xp, means, W, X):
    n = means.shape[0]
    out = xp.empty((n, X.shape[1]))
    for g in range(n):
        sw = xp.sqrt(W[g])
        out[g] = xp.linalg.lstsq(X * sw[:, None], means[g] * sw, rcond=None)[0]
    return out


def fit_batched(xp, means, W, X):
    """beta, residual variance -- identical maths to fastfit.fit_batched."""
    Xw = X[None, :, :] * W[:, :, None]                  # n x c x p
    XtWX = xp.einsum("ncp,cq->npq", Xw, X)              # n x p x p
    XtWy = xp.einsum("ncp,nc->np", Xw, means)           # n x p
    beta = xp.linalg.solve(XtWX, XtWy[..., None])[..., 0]
    resid = means - beta @ X.T
    s2 = xp.einsum("nc,nc->n", resid * W, resid) / (X.shape[0] - X.shape[1])
    return beta, s2


def sweep(xp, means, W, designs):
    """P designs against one resident copy of the data: the permutation shape."""
    out = []
    for X in designs:
        out.append(fit_batched(xp, means, W, X)[1])
    return xp.stack(out) if hasattr(xp, "stack") else np.stack(out)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def get_backend(name, device="cuda"):
    if name == "numpy":
        return np, "numpy (CPU)", lambda: None, lambda a: a
    if name == "cupy":
        import cupy as cp
        return cp, f"cupy ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})", \
            cp.cuda.Stream.null.synchronize, cp.asarray
    if name == "torch":
        import torch

        class _TorchNS:
            @staticmethod
            def sqrt(a): return torch.sqrt(a)
            @staticmethod
            def empty(shape): return torch.empty(shape, device=device, dtype=torch.float64)
            @staticmethod
            def einsum(*a): return torch.einsum(*a)
            @staticmethod
            def stack(a): return torch.stack(a)
            linalg = torch.linalg

        label = (f"torch {torch.__version__} on {device}"
                 + (f" ({torch.cuda.get_device_name(0)})"
                    if device == "cuda" and torch.cuda.is_available() else ""))
        sync = (torch.cuda.synchronize if device == "cuda" else (lambda: None))
        return _TorchNS, label, sync, \
            (lambda a: torch.as_tensor(a, device=device, dtype=torch.float64))
    raise SystemExit(f"unknown backend {name}")


def timed(fn, sync, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    sync()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best


def run(backend="numpy", device="cuda", loop_limit=20000, n_designs=30, seed=0):
    xp, label, sync, to_dev = get_backend(backend, device)
    rng = np.random.default_rng(seed)

    print(f"backend: {label}")
    print()
    header = (f"{'proteins':>9} {'clones':>7} {'loop ms':>10} {'batched ms':>11} "
              f"{'speedup':>8} {'GFLOP/s':>9} {'FLOP/byte':>10}")
    print(header)
    print("-" * len(header))

    configs = [(n, c) for n in (1_000, 5_000, 20_000, 100_000)
               for c in (5, 15, 25, 50)]
    rows = []
    for n, c in configs:
        p = 2
        dose = np.repeat(np.arange(3.0), int(np.ceil(c / 3)))[:c]
        X_np = np.column_stack([np.ones(c), dose])
        means_np = rng.normal(23, 1.0, (n, c))
        W_np = rng.uniform(1.0, 3.0, (n, c))
        X, means, W = to_dev(X_np), to_dev(means_np), to_dev(W_np)

        t_loop = np.nan
        if n <= loop_limit and backend == "numpy":
            t_loop = timed(lambda: fit_loop(xp, means, W, X), sync, repeats=1, warmup=0)
        t_batch = timed(lambda: fit_batched(xp, means, W, X), sync)

        f = flop_count(n, c, p)
        b = byte_count(n, c, p)
        rows.append(dict(n=n, c=c, loop=t_loop, batch=t_batch,
                         gflops=f / t_batch / 1e9, intensity=f / b))
        print(f"{n:>9,} {c:>7} "
              f"{1e3 * t_loop:>10.1f} " if np.isfinite(t_loop) else
              f"{n:>9,} {c:>7} {'--':>10} ", end="")
        print(f"{1e3 * t_batch:>11.2f} "
              f"{(t_loop / t_batch if np.isfinite(t_loop) else np.nan):>8.0f} "
              if np.isfinite(t_loop) else f"{1e3 * t_batch:>11.2f} {'--':>8} ", end="")
        print(f"{f / t_batch / 1e9:>9.2f} {f / b:>10.2f}")

    # the permutation-sweep shape
    print()
    print(f"permutation sweep: {n_designs} designs against one resident copy")
    hdr2 = (f"{'proteins':>9} {'clones':>7} {'total ms':>10} {'GFLOP/s':>9} "
            f"{'FLOP/byte':>10} {'device MB':>10}")
    print(hdr2)
    print("-" * len(hdr2))
    for n, c in [(20_000, 5), (20_000, 25), (100_000, 25), (100_000, 50)]:
        p = 2
        dose = np.repeat(np.arange(3.0), int(np.ceil(c / 3)))[:c]
        designs_np = []
        for k in range(n_designs):
            d = rng.permutation(dose)
            designs_np.append(np.column_stack([np.ones(c), d]))
        means, W = to_dev(rng.normal(23, 1.0, (n, c))), to_dev(rng.uniform(1, 3, (n, c)))
        designs = [to_dev(d) for d in designs_np]
        t = timed(lambda: sweep(xp, means, W, designs), sync, repeats=2)
        f = flop_count(n, c, p, n_designs)
        b = byte_count(n, c, p, n_designs=n_designs)
        mb = 2 * n * c * 8 / 1e6
        print(f"{n:>9,} {c:>7} {1e3 * t:>10.1f} {f / t / 1e9:>9.2f} "
              f"{f / b:>10.2f} {mb:>10.1f}")

    print()
    print("Arithmetic intensity below ~10 FLOP/byte is memory-bound. A GPU only")
    print("helps a memory-bound kernel when the data is already resident on the")
    print("device -- a PCIe round trip costs about 25 us per 160 KB each way, and")
    print("a kernel launch about 5-10 us.")
    return rows


def bench_accel(backend="auto", device="cuda", dtype=np.float64, n_designs=30):
    """The real pipeline shape: resident clone summaries, P designs over them."""
    from .accel import sweep_designs
    from .backend import resolve

    b = resolve(backend, device)
    rng = np.random.default_rng(0)
    print(f"\naccel.sweep_designs on {b.label}, dtype={np.dtype(dtype).name}")
    hdr = (f"{'proteins':>9} {'clones':>7} {'designs':>8} {'ms':>9} "
           f"{'GB/s':>7} {'per design ms':>14}")
    print(hdr); print("-" * len(hdr))
    for n, c in [(20_000, 5), (20_000, 25), (100_000, 25), (1_000_000, 25)]:
        dose = np.repeat(np.arange(3.0), int(np.ceil(c / 3)))[:c]
        M = rng.normal(23, 1.0, (n, c)); W = rng.uniform(1, 3, (n, c))
        designs = [np.column_stack([np.ones(c), rng.permutation(dose)])
                   for _ in range(n_designs)]
        sweep_designs(M, W, designs[:2], backend=b, dtype=dtype)   # warm up
        t0 = time.perf_counter()
        sweep_designs(M, W, designs, backend=b, dtype=dtype)
        b.sync()
        t = time.perf_counter() - t0
        gb = n_designs * 3 * n * c * np.dtype(dtype).itemsize / 1e9
        print(f"{n:>9,} {c:>7} {n_designs:>8} {1e3*t:>9.1f} {gb/t:>7.1f} "
              f"{1e3*t/n_designs:>14.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy",
                    choices=["auto", "numpy", "cupy", "torch"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--designs", type=int, default=30)
    ap.add_argument("--loop-limit", type=int, default=20000,
                    help="skip the per-protein loop above this protein count")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--accel-only", action="store_true")
    args = ap.parse_args()
    if not args.accel_only:
        run(args.backend, args.device, args.loop_limit, args.designs)
    bench_accel(args.backend, args.device, np.dtype(args.dtype), args.designs)


if __name__ == "__main__":
    main()
