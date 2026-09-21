"""CPU reference models of the Q8_0 / Q3_K / Q5_K / Q6_K formats for the kernel checks: the integer fields of a
block (q, the integer scales and mins, the fp16 d / dmin) parsed with the bit mapping the kernel decodes, so that
(1) the parsed fields reproduce gguf.quants.dequantize bit for bit (the format reference) and (2) the int8
forms' arithmetic - the activations quantised to int8, the integer dots, the scales applied after - can be
modelled in fp64 and the kernel compared against it within fp32 rounding."""
from __future__ import annotations

import numpy as np

QK = 256


def _f16(b: np.ndarray) -> np.ndarray:
    return b.copy().view(np.float16).astype(np.float32)


def _scale_min_k4(sc12: np.ndarray):
    """The 12 scale bytes of Q4_K / Q5_K -> (sc[8], m[8]) 6-bit, as get_scale_min_k4."""
    s = sc12.astype(np.int32)
    sc = np.empty(s.shape[:-1] + (8,), np.int32)
    mn = np.empty_like(sc)
    for j in range(4):
        sc[..., j] = s[..., j] & 63
        mn[..., j] = s[..., j + 4] & 63
        sc[..., j + 4] = (s[..., j + 8] & 0xF) | ((s[..., j] >> 6) << 4)
        mn[..., j + 4] = (s[..., j + 8] >> 4) | ((s[..., j + 4] >> 6) << 4)
    return sc, mn


def parse(raw: np.ndarray, wt: int):
    """raw uint8 [n, nb * block] -> dict(q int32 [n, nb, 256], dg fp32 [n, nb, G] the scale of each group of
    256 // G weights (d * sc in fp32, exact; Q8_0: the group's own d), mn int32 [n, nb, 8] or None, dmin fp32 [n, nb]
    or None). value = dg * q - dmin * mn."""
    n = raw.shape[0]
    if wt == 8:      # Q8_0: 8 blocks of 34 per 256 weights: d, int8 qs[32]
        b = raw.reshape(n, -1, 8, 34)
        d8 = _f16(b[..., :2]).reshape(n, -1, 8)                 # [n, nb, 8]
        q = b[..., 2:].view(np.int8).astype(np.int32).reshape(n, -1, QK)
        return dict(q=q, dg=d8, mn=None, dmin=None, G=8)
    if wt == 11:     # Q3_K
        b = raw.reshape(n, -1, 110)
        hmask = b[..., :32].astype(np.int32)
        qs = b[..., 32:96].astype(np.int32)
        s = b[..., 96:108].astype(np.int32)
        d = _f16(b[..., 108:110]).reshape(n, -1)
        q = np.empty(b.shape[:2] + (QK,), np.int32)
        for w in range(QK):
            nn, j, l = w // 128, (w // 32) % 4, w % 32
            hb = (hmask[..., l] >> (4 * nn + j)) & 1
            q[..., w] = ((qs[..., 32 * nn + l] >> (2 * j)) & 3) - np.where(hb == 1, 0, 4)
        sc = np.empty(b.shape[:2] + (16,), np.int32)
        for i in range(16):
            lo = (s[..., i % 8] >> (4 * (i // 8))) & 0xF
            hi = (s[..., 8 + i % 4] >> (2 * (i // 4))) & 3
            sc[..., i] = (lo | (hi << 4)) - 32
        return dict(q=q, dg=d[..., None] * sc.astype(np.float32), mn=None, dmin=None, G=16)
    if wt == 13:     # Q5_K
        b = raw.reshape(n, -1, 176)
        d = _f16(b[..., 0:2]).reshape(n, -1)
        dmin = _f16(b[..., 2:4]).reshape(n, -1)
        sc, mn = _scale_min_k4(b[..., 4:16])
        qh = b[..., 16:48].astype(np.int32)
        qs = b[..., 48:176].astype(np.int32)
        q = np.empty(b.shape[:2] + (QK,), np.int32)
        for w in range(QK):
            ib, p = w // 32, w % 32
            q[..., w] = ((qs[..., 32 * (ib >> 1) + p] >> (4 * (ib & 1))) & 0xF) | (((qh[..., p] >> ib) & 1) << 4)
        return dict(q=q, dg=d[..., None] * sc.astype(np.float32), mn=mn, dmin=dmin, G=8)
    if wt == 14:     # Q6_K
        b = raw.reshape(n, -1, 210)
        ql = b[..., :128].astype(np.int32)
        qh = b[..., 128:192].astype(np.int32)
        sc = b[..., 192:208].view(np.int8).astype(np.int32)
        d = _f16(b[..., 208:210]).reshape(n, -1)
        q = np.empty(b.shape[:2] + (QK,), np.int32)
        for w in range(QK):
            h, t, p = w // 128, (w // 32) % 4, w % 32
            q[..., w] = (((ql[..., 64 * h + 32 * (t & 1) + p] >> (4 * (t >> 1))) & 0xF) | (((qh[..., 32 * h + p] >> (2 * t)) & 3) << 4)) - 32
        return dict(q=q, dg=d[..., None] * sc.astype(np.float32), mn=None, dmin=None, G=16)
    raise KeyError(wt)


def dequantize(fields: dict) -> np.ndarray:
    """The fp32 weights [n, K] from the parsed fields, in gguf.quants' operation order (d * sc first, then times q, minus dmin * mn)."""
    q, dg, G = fields["q"], fields["dg"], fields["G"]
    n, nb, _ = dg.shape
    per = QK // G
    w = dg[..., None] * q.reshape(n, nb, G, per).astype(np.float32)
    if fields["mn"] is not None:
        w = w - (fields["dmin"][..., None] * fields["mn"].astype(np.float32))[..., None]
    return w.reshape(n, nb * QK)


def random_rows(rng: np.random.Generator, n: int, K: int, wt: int) -> np.ndarray:
    """Random valid rows of the type (every bit pattern of the quantised fields is valid; d / dmin small fp16)."""
    block = {8: 272, 11: 110, 13: 176, 14: 210}[wt]
    nb = K // QK
    raw = rng.integers(0, 256, size=(n, nb, block), dtype=np.uint8)
    d = (rng.uniform(0.25, 1.0, size=(n, nb)) * 2.0 ** -6).astype(np.float16).view(np.uint8).reshape(n, nb, 2)
    if wt == 8:
        raw = raw.reshape(n, nb, 8, 34)
        d8 = (rng.uniform(0.25, 1.0, size=(n, nb, 8)) * 2.0 ** -6).astype(np.float16).view(np.uint8).reshape(n, nb, 8, 2)
        raw[..., :2] = d8
        raw = raw.reshape(n, nb, 272)
    elif wt == 11:
        raw[..., 108:110] = d
    elif wt == 13:
        raw[..., 0:2] = d
        raw[..., 2:4] = (rng.uniform(0.25, 1.0, size=(n, nb)) * 2.0 ** -8).astype(np.float16).view(np.uint8).reshape(n, nb, 2)
    elif wt == 14:
        raw[..., 208:210] = d
    return raw.reshape(n, nb * block)


def int8_model(fields: dict, xq: np.ndarray, sx: np.ndarray, sumx: np.ndarray | None, per_256: bool) -> np.ndarray:
    """The int8 form's arithmetic in fp64: xq int8 [M, K], sx fp32 the activation scale per 32 (per_256 False) or
    per 256, sumx the group sums per 32 (fp32 dequantised for per-32, int32 quantised for per-256; the min term).
    Returns fp64 [M, n]."""
    q, dg, G = fields["q"], fields["dg"], fields["G"]
    n, nb, _ = dg.shape
    M = xq.shape[0]
    per = QK // G
    xq64 = xq.astype(np.float64).reshape(M, nb, G, per)
    q64 = q.astype(np.float64).reshape(n, nb, G, per)
    dots = np.einsum("mkgp,nkgp->mnkg", xq64, q64)                                       # exact integer dots
    scaled = dots * dg.astype(np.float64)[None]                                          # [M, n, nb, G]
    if per_256:
        y = np.einsum("mnk,mk->mn", scaled.sum(-1), sx.astype(np.float64))
    else:
        s32 = sx.astype(np.float64).reshape(M, nb, 8)
        s = np.repeat(s32, G // 8, axis=-1) if G > 8 else s32
        y = np.einsum("mnkg,mkg->mn", scaled, s)
    if fields["mn"] is not None:
        mn = fields["mn"].astype(np.float64)                                             # [n, nb, 8]
        dm = fields["dmin"].astype(np.float64)
        if per_256:
            sm = sumx.astype(np.float64).reshape(M, nb, 8)
            y = y - np.einsum("mkg,nkg,nk,mk->mn", sm, mn, dm, sx.astype(np.float64))
        else:
            sm = sumx.astype(np.float64).reshape(M, nb, 8)
            y = y - np.einsum("mkg,nkg,nk->mn", sm, mn, dm)
    return y
