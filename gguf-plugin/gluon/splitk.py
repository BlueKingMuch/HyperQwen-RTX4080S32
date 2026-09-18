"""The split-K of a Gluon decode launch, chosen against the wave count. A CTA of these kernels (64 output rows,
4 warps, 16.4 KB of shared memory, about 155 registers) is one of three
resident per SM on this card, so one wave is 3 x 80 = 240 CTAs; the profile
of the 0030 image showed the file's largest launch groups at 1.13 and 1.33
waves (the second wave a few CTAs on 240 slots) and the k / v tensors at an
eighth of a wave, at 322-372 and 57-280 GB/s where the same kernels reach
440 at 2.7 waves. A sweep over the file's production shapes (the
third IQ3_S int8 form, M = 4 and 8, splits 1-10, the partial sums included)
measured the split per (output tiles, k-blocks) that this table records;
the shapes of this file are few, so the table covers them and the formula
below is the fallback for any other. The partial sums cost 8 s M N bytes
against the weights' bytes, so the table is for the decode regime (M <= 16),
which is the only regime these launchers serve.
"""
from __future__ import annotations

SMS = 80
RESIDENT_PER_SM = 3

# (output tiles of 64, k-blocks of 256) -> split-K; measured on the third
# IQ3_S int8 form unless marked, the geometry is the same for every kernel
_TABLE = {
    (272, 20): 4,     # [17408x5120] ffn gate / up per shard: 376 -> 481 GB/s (M = 4), 421 -> 463 (M = 8)
    (80, 68): 2,      # [5120x17408] ffn_down: 375 -> 434; 1.33 waves is the worst geometry, 2/3 of one wave beats it
    (96, 20): 5,      # [6144x5120] attn_gate and the like: 457 -> 549 (2.0 waves exactly)
    (80, 24): 3,      # [5120x6144] ssm_out: the 0033 profile measured split 2 at 168-265 GB/s in situ against 292-382 at 3 (0030); the sweep's single clean cell for 2 was not to be trusted
    (192, 20): 3,     # [12288x5120]: 441 -> 460
    (32, 20): 5,      # [2048x5120] k + v merged: 280 -> 379
    (16, 20): 5,      # [1024x5120] k, v: 142 -> 272
    (160, 20): 1,   # [10240x5120] attn_qkv alone
    (256, 20): 5,   # [16384x5120] qkv + gate merged
    (544, 20): 1,   # [34816x5120] gate + up merged
    (3880, 20): 1,    # [248320x5120] the head: 16 waves as it is
}


PARTIALS_BUDGET = 0.15     # the partial sums' bytes (written and read back) as a fraction of the weights' bytes


def split_k_for(num_k_blocks: int, n_out: int, sms: int = SMS, m: int | None = None, block_bytes: int | None = None) -> int:
    """The split-K for a launch of n_out output rows over num_k_blocks
    k-blocks of 256: the measured table, else about 4.5 waves of CTAs with
    at least 4 k-blocks per split, never more than 8. With m (the batch
    rows) and block_bytes (the type's bytes per 256-weight block) the split
    is capped so that the fp32 partial sums, 8 S m N bytes written and read
    back against N K b / 8 bytes of weights, stay within PARTIALS_BUDGET:
    S <= budget K b / (64 m) with b the bits per weight (0034: the table was
    measured at m = 4 and 8; at m = 16 the cap is 2 on K = 5120)."""
    n_tiles = -(-n_out // 64)
    s = _TABLE.get((n_tiles, num_k_blocks))
    if s is None:
        wave = RESIDENT_PER_SM * sms
        if n_tiles >= 4 * wave:
            s = 1
        else:
            s = int(max(1, min(8, num_k_blocks // 4, int(round(4.5 * wave / n_tiles)))))
    if m is not None and block_bytes is not None and m > 0:
        bits = 8 * block_bytes / 256
        cap = int(PARTIALS_BUDGET * (num_k_blocks * 256) * bits / (64 * m))
        s = min(s, max(1, cap))
    return max(1, min(s, num_k_blocks))
