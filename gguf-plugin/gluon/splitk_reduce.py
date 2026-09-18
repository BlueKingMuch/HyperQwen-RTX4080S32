"""The split-K reduction inside the tile kernels: every CTA stores its fp32
partial [M, 64] into P[pid_k], fences, and one
thread bumps the tile's counter (atom.acq_rel.gpu; Gluon issues a scalar
atomic once per CTA and broadcasts the result); the
CTA that finds SPLITK - 1 is the last of its tile and sums the partials in
the fixed order 0 .. SPLITK - 1 (deterministic), stores the bf16 result and
resets the counter, so one zeroed counter array per device serves every
launch (stream order: a launch's last CTAs reset before the next launch
starts). The result is the fp32 sum cast once; the launcher's torch.sum
path casts every partial to bf16 first, so the two differ in the last bit
(the in-kernel one is the more exact).

    from .splitk_reduce import _splitk_reduce, splitk_counters
    # in the kernel's epilogue, when REDUCE and SPLITK > 1:
    _splitk_reduce(Y, P, C, acc, ym, yn, omask, pid_n, pid_k, stride_ym, stride_pm, stride_pk, SPLITK)

The caller supplies the output view.
"""
from __future__ import annotations

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def _fence_gl(x):
    """membar.gl per thread (the partial's stores ordered before the counter's atomic), as __threadfence()."""
    return gl.inline_asm_elementwise("membar.gl; mov.b32 $0, $1;", "=r,r", [x], dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _splitk_reduce(Y, P, C, acc, ym, yn, omask, pid_n, pid_k, stride_ym, stride_pm, stride_pk, SPLITK: gl.constexpr):
    """The CTA's fp32 partial to P[pid_k]; the last CTA of the tile sums P[0 .. SPLITK - 1] into Y (bf16)."""
    Pp = P + pid_k.to(gl.int64) * stride_pk
    gl.store(Pp + ym[:, None] * stride_pm + yn[None, :], acc, mask=omask)
    fenced = _fence_gl(pid_k)
    gl.barrier()
    old = gl.atomic_add(C + pid_n + (fenced & 0), 1, sem="acq_rel", scope="gpu")
    if old == SPLITK - 1:
        tot = gl.load(P + ym[:, None] * stride_pm + yn[None, :], mask=omask, other=0.0, cache_modifier=".cg")
        for s in gl.static_range(1, SPLITK):
            tot = tot + gl.load(P + s * stride_pk + ym[:, None] * stride_pm + yn[None, :], mask=omask, other=0.0, cache_modifier=".cg")
        gl.store(Y + ym[:, None] * stride_ym + yn[None, :], tot.to(Y.dtype.element_ty), mask=omask)
        gl.atomic_xchg(C + pid_n, 0, sem="relaxed", scope="gpu")


_COUNTERS = {}


def splitk_counters(device: torch.device, n: int = 8192) -> torch.Tensor:
    """One zeroed int32 counter array per device, shared by every launch (each launch leaves it zero)."""
    key = (device.type, device.index)
    if key not in _COUNTERS:
        _COUNTERS[key] = torch.zeros(n, dtype=torch.int32, device=device)
    return _COUNTERS[key]


def splitk_buffers(X: torch.Tensor, n_out: int, splitk: int, reduce: bool, out: torch.Tensor | None = None):
    """(Y, P, C, strides) for a tile launcher: with the in-kernel reduction Y is the bf16 result and P the fp32
    partials (strides: ym, yk = 0, pm, pk); without, Y is the fp32 partial array the launcher sums (or the bf16
    result at split 1) and P aliases it. `out` (a [M, n_out] view of the caller's tensor, any row stride, e.g.
    a merged layer's column slice) takes the place of the allocated result where the kernel writes bf16
    directly (reduce, or split 1); with the sum kernel the launcher sums into it (splitk_result)."""
    C = splitk_counters(X.device)
    M = splitk_rows(X)
    if out is not None:
        assert out.dtype == X.dtype and tuple(out.shape) == (M, n_out) and out.stride(1) == 1
    if reduce and splitk > 1:
        Y = out if out is not None else torch.empty((M, n_out), dtype=X.dtype, device=X.device)
        P = torch.empty((splitk, M, n_out), dtype=torch.float32, device=X.device)
        return Y, P, C, (Y.stride(0), 0, P.stride(1), P.stride(0))
    if splitk == 1 and out is not None:
        return out, out, C, (out.stride(0), 0, 0, 0)
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    return Y, Y, C, (Y.stride(1), Y.stride(0), 0, 0)


def splitk_rows(X: torch.Tensor) -> int:
    return int(X.shape[0])


def splitk_result(Y: torch.Tensor, X: torch.Tensor, splitk: int, reduce: bool, out: torch.Tensor | None = None) -> torch.Tensor:
    if (reduce and splitk > 1) or (splitk == 1 and out is not None):
        return Y
    if splitk == 1:
        return Y[0]
    if out is not None:
        return torch.sum(Y, 0, dtype=X.dtype, out=out)
    return torch.sum(Y, 0, dtype=X.dtype)
