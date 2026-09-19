"""Isolated native FP8 flat-prefill mapping; no kernel launch or image mutation."""
import ast
import hashlib
from pathlib import Path
import textwrap

PARENT_SHA = 'b428d0a7680242917e825e5146bb7b35e3a013eb984f3fc34c923d33eebe65c7'
# PARENT: The tree after fp8-causal r2, which is what PINS below is of. A docker
# image id stood here; on this tree the parent is not an image, it is four files,
# and their sha256 is the whole of the claim.
HELPER_SHA = '8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95'
FLAG = 'VLLM_TRITON_FP8_PREFILL_FLAT'
FROZEN = """_FP8_PREFILL_FLAT_TEXT = __import__("os").environ.get("VLLM_TRITON_FP8_PREFILL_FLAT", "0")
if _FP8_PREFILL_FLAT_TEXT not in ("0", "1"):
    raise ValueError("VLLM_TRITON_FP8_PREFILL_FLAT must be 0 or 1")
_FP8_PREFILL_FLAT = _FP8_PREFILL_FLAT_TEXT == "1"
"""

FLAT_RESOLVE = """    if FP8_FLAT_PHASE >= 0:
        tl.static_assert(FP8_FLAT_PHASE < 3 and not IS_3D and FP8_FULL_CAUSAL)
        # Reserve 32 legacy BQ2 slots per request for phase-zero widths <=63.
        left = 0
        right = num_seqs
        while left < right:
            mid = (left + right) // 2
            base = tl.load(query_start_len_ptr + mid) // 8 + 32 * mid
            if base <= q_block_global_idx:
                left = mid + 1
            else:
                right = mid
        seq_idx = left - 1
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_query_len = (
            tl.load(query_start_len_ptr + seq_idx + 1)
            - cur_batch_in_all_start_index
        )
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        q_block_local_idx = q_block_global_idx - (
            cur_batch_in_all_start_index // 8 + 32 * seq_idx
        )
        if FP8_FLAT_PHASE == 0:
            _flat_token_base = tl.where(
                cur_batch_query_len >= 64,
                q_block_local_idx * 8,
                q_block_local_idx * BLOCK_Q,
            )
            _flat_row_shift = 0
        else:
            if cur_batch_query_len < 64:
                return
            _flat_token_base = q_block_local_idx * 8
            _flat_row_shift = FP8_FLAT_PHASE * BLOCK_M
        _flat_qmin = _flat_token_base + _flat_row_shift // num_queries_per_kv
        _flat_qmax = (
            _flat_token_base + (_flat_row_shift + BLOCK_M - 1) // num_queries_per_kv
        )
        if _flat_qmin >= cur_batch_query_len:
            return
"""

OLD_ROWS = """    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
"""
NEW_ROWS = """    if FP8_FLAT_PHASE >= 0:
        query_pos = _flat_token_base + (_flat_row_shift + offs_m) // num_queries_per_kv
        query_offset_1 = (
            kv_head_idx * num_queries_per_kv
            + (_flat_row_shift + offs_m) % num_queries_per_kv
        )
    else:
        query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
        query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
"""

FLAT_BOUNDS = """    if FP8_FLAT_PHASE >= 0:
        # Phase1 spans four query positions: the native BQ2 helper cannot be reused.
        max_seq_prefix_len = tl.minimum(seq_len, context_len + _flat_qmax + 1)
        loop_lo = 0
        loop_hi = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
"""

OLD_FULL = """        full_end = tl.maximum(loop_lo, tl.minimum(
            loop_hi, (context_len + q_block_local_idx * BLOCK_Q + 1) // TILE_SIZE
        ))
"""
NEW_FULL = """        if FP8_FLAT_PHASE >= 0:
            full_end = tl.maximum(loop_lo, tl.minimum(
                loop_hi, (context_len + _flat_qmin + 1) // TILE_SIZE
            ))
        else:
            full_end = tl.maximum(loop_lo, tl.minimum(
                loop_hi, (context_len + q_block_local_idx * BLOCK_Q + 1) // TILE_SIZE
            ))
"""
HOST = """    _fp8_flat_prefill = (
        _FP8_PREFILL_FLAT and _fp8_full_causal
        and not use_3d and max_seqlen_q >= 64 and q.shape[0] >= 64
    )
    _fp8_flat_phases = (0, 1, 2) if _fp8_flat_prefill else (-1,)
    for _fp8_flat_phase in _fp8_flat_phases:
        _fp8_flat_grid = (
            (q.shape[0] // 8 + 32 * num_seqs, num_kv_heads)
            if _fp8_flat_phase >= 0 else grid
        )
"""


def sha(value):
    return hashlib.sha256(value.encode('utf8') if isinstance(value, str) else value).hexdigest()


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def block(source, node):
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def once(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new, 1)


# There is no default parent. The one file this generator may be run against is
# the one PARENT_SHA pins -- the kernel as fp8-causal r2 leaves it -- and the
# caller reads it from the tree it is about to rewrite.


def changes(parent):
    assert sha(parent) == PARENT_SHA
    fn = functions(parent)
    kernel, host = fn['kernel_unified_attention'], fn['unified_attention']
    resolve = next(n for n in kernel.body if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call)
                   and getattr(n.value.func, 'id', '') == 'resolve_seq_and_query_len')
    early = kernel.body[kernel.body.index(resolve)+1]
    assert isinstance(early, ast.If) and 'q_block_local_idx * BLOCK_Q' in ast.unparse(early.test)
    old_resolve = ''.join(parent.splitlines(keepends=True)[resolve.lineno-1:early.end_lineno])
    bounds = next(n for n in kernel.body if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Call)
                  and getattr(n.value.func, 'id', '') == 'compute_tile_loop_bounds')
    old_bounds = block(parent, bounds)
    launch = next(n for n in host.body if isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Subscript)
                  and getattr(n.value.func.value, 'id', '') == 'kernel_unified_attention')
    old_launch = block(parent, launch)
    new_launch = once(old_launch, 'kernel_unified_attention[grid](', 'kernel_unified_attention[_fp8_flat_grid](')
    new_launch = once(new_launch, '        FP8_FULL_CAUSAL=_fp8_full_causal,\n',
                      '        FP8_FULL_CAUSAL=_fp8_full_causal,\n        FP8_FLAT_PHASE=_fp8_flat_phase,\n')
    return [
        ('_FP8_CAUSAL_FULL = envs.VLLM_TRITON_FP8_CAUSAL_FULL\n',
         '_FP8_CAUSAL_FULL = envs.VLLM_TRITON_FP8_CAUSAL_FULL\n' + FROZEN),
        ('    FP8_FULL_CAUSAL: tl.constexpr = False,\n',
         '    FP8_FULL_CAUSAL: tl.constexpr = False,\n    FP8_FLAT_PHASE: tl.constexpr = -1,\n'),
        (old_resolve, FLAT_RESOLVE + '    else:\n' + textwrap.indent(old_resolve, '    ')),
        (OLD_ROWS, NEW_ROWS),
        (old_bounds, FLAT_BOUNDS + '    else:\n' + textwrap.indent(old_bounds, '    ')),
        (OLD_FULL, NEW_FULL),
        (old_launch, HOST + textwrap.indent(new_launch, '    ')),
    ]


def assert_identity(parent, output):
    restored = output
    for old, new in reversed(changes(parent)):
        restored = once(restored, new, old)
    assert restored == parent
    a, b = functions(parent), functions(output)
    unchanged = []
    for name in a.keys() - {'kernel_unified_attention', 'unified_attention'}:
        assert block(parent, a[name]) == block(output, b[name]), name
        unchanged.append(name)
    # Every load/math/PV/store statement in the original loop nests is unchanged.
    def loops(src):
        k = functions(src)['kernel_unified_attention']
        return [n for n in ast.walk(k) if isinstance(n, ast.For) and getattr(n.target, 'id', '') == 'j']
    assert [ast.dump(n) for n in loops(parent)] == [ast.dump(n) for n in loops(output)]
    # Q load and complete 2D/3D output epilogue are byte-identical; only index values differ.
    pa, pb = functions(parent)['kernel_unified_attention'], functions(output)['kernel_unified_attention']
    for name in ('reduce_segments', '_fp8_full_softmax_step'):
        assert block(parent, a[name]) == block(output, b[name])
    for marker in ('    # Q : (BLOCK_M, HEAD_SIZE_PADDED)\n',):
        p = block(parent, pa); q = block(output, pb)
        startp, startq = p.index(marker), q.index(marker)
        endp, endq = p.index('    block_table_offset', startp), q.index('    block_table_offset', startq)
        assert p[startp:endp] == q[startq:endq]
    return dict(parent_reverse_byte_identical=True, unchanged_helpers=sorted(unchanged),
                all_three_existing_KV_loops_AST_identical=True,
                native_reducer_and_rounded_softmax_byte_identical=True,
                no_gpu_or_numerical_proof_claim=True)


def generated_source(parent):
    output = parent
    for old, new in changes(parent):
        output = once(output, old, new)
    compile(output, '<native-fp8-flat-prefill>', 'exec')
    assert_identity(parent, output)
    return output


if __name__ == '__main__':
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    a = p.parse_args()
    parent = a.parent.read_text(encoding='utf8')
    output = generated_source(parent)
    print(json.dumps(dict(parent_sha=PARENT_SHA, generated_sha=sha(output),
                          **assert_identity(parent, output))))
