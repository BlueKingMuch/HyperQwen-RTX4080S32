"""Exact-source native FP8 causal full/boundary port; no engine/GPU actions.

The physical page expressions, QK/PV arithmetic, row ownership, tile/segment
geometry and reducer remain native. Only full-tile predicates are specialized.
"""
import ast
import hashlib
import textwrap

PARENT_SHA = '9a2dac31ea0ba1259b0c4e824fdafd5928584eb6536d3f80716ded019da29b8b'
HELPER_SHA = '8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95'
BACKEND_SHA = 'c01afe31a641ce6c772b41b6b319d6d5a64810a3fbdf6edc727d006e83a4bdde'
ENV_SHA = '59ae64de51704b3b640971b209cee8f264596d6d2f94d0a680438f1f7f15df1a'
GATE_SHA = '74fed6857dc505ad92e3bad1f06b158691f706f9ef0c47fb525583047255ecb1'
FLAG = 'VLLM_TRITON_FP8_CAUSAL_FULL'
ENV_ENTRY = f'    "{FLAG}": lambda: bool(int(os.environ.get("{FLAG}", "0"))),\n'

SOFTMAX = '''@triton.jit
def _fp8_full_softmax_step(S, M, L):
    # Preserve the rounded score consumed by rowmax: no score*scale-minus-max
    # contraction when the full-tile causal select is removed. No FTZ modifier.
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    delta = tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[S, m_j[:, None]], dtype=tl.float32, is_pure=True, pack=1,
    )
    P = tl.exp(delta)
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


'''

HOST_GUARD = '''    _fp8_full_causal = (
        _FP8_CAUSAL_FULL
        and current_platform.is_device_capability(89)
        and kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
        and q.dtype == torch.float8_e4m3fn
        and k.dtype == torch.float8_e4m3fn
        and v.dtype == torch.float8_e4m3fn
        and out.dtype == torch.bfloat16
        and num_query_heads == 24 and num_kv_heads == 4 and head_size == 256
        and block_size == 880 and BLOCK_M == 16 and BLOCK_Q == 2
        and tile_size == 32 and 1 <= num_seqs <= 4 and max_seqlen_q > 1
        and use_causal and not use_per_seq_causal
        and sliding_window_val == 0 and chunk_lookback < 0
        and not use_mm_prefix and not use_rswa
        and softcap == 0 and not use_alibi_slopes and not use_qq_bias
        and sinks is None and output_scale is None and not use_td
        and not is_batch_invariant
    )

'''

KERNEL_GUARD = '''    if FP8_FULL_CAUSAL:
        tl.static_assert(KV_QUANT_MODE == 1 and Q_IS_FP8)
        tl.static_assert(num_query_heads == 24 and num_queries_per_kv == 6)
        tl.static_assert(HEAD_SIZE == 256 and HEAD_SIZE_PADDED == 256)
        tl.static_assert(BLOCK_SIZE == 880 and TILE_SIZE == 32)
        tl.static_assert(BLOCK_M == 16 and BLOCK_Q == 2)
        tl.static_assert(USE_CAUSAL and not USE_PER_SEQ_CAUSAL)
        tl.static_assert(not USE_MM_PREFIX and not USE_R_SWA)
        tl.static_assert(SLIDING_WINDOW == 0 and CHUNK_LOOKBACK < 0)
        tl.static_assert(not USE_ALIBI_SLOPES and not USE_QQ_BIAS)
        tl.static_assert(not USE_SOFTCAP and not USE_SINKS and not USE_FP8)
        tl.static_assert(not USE_TD and not USE_TD_QO)
        tl.static_assert(query_ptr.dtype.element_ty == tl.float8e4nv)
        tl.static_assert(output_ptr.dtype.element_ty == tl.bfloat16)

'''


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def source_block(source, node):
    return ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def replacements(parent):
    assert sha(parent) == PARENT_SHA, 'Unknown native attention source'
    kernel = functions(parent)['kernel_unified_attention']
    loop = next(n for n in kernel.body if isinstance(n, ast.For) and
                isinstance(n.target, ast.Name) and n.target.id == 'j')
    old = source_block(parent, loop)
    assert old.startswith('    for j in range(loop_lo, loop_hi):\n')
    body = old.split('\n', 1)[1]
    full = body.replace('        tile_mask = seq_offset < max_seq_prefix_len\n',
                        '        tile_mask = tl.full((TILE_SIZE,), True, tl.int1)\n', 1)
    begin = full.index('        query_abs_pos = context_len + query_pos[:, None]\n')
    end = full.index('        # S : (BLOCK_M, TILE_SIZE)\n', begin)
    full = full[:begin] + '        seq_mask = tl.full((BLOCK_M, TILE_SIZE), True, tl.int1)\n\n' + full[end:]
    full = full.replace('M, L, P, alpha = softmax_step(S, M, L)',
                        'M, L, P, alpha = _fp8_full_softmax_step(S, M, L)', 1)
    # Full bound is the EARLIEST actual Q row. Keep both bounds of the existing
    # MQ3D segment. Empty/causally empty segments preserve the native epilogue.
    new = '''    if FP8_FULL_CAUSAL:
        full_end = tl.maximum(loop_lo, tl.minimum(
            loop_hi, (context_len + q_block_local_idx * BLOCK_Q + 1) // TILE_SIZE
        ))
        for j in range(loop_lo, full_end):
''' + textwrap.indent(full, '    ') + '''        for j in range(full_end, loop_hi):
''' + textwrap.indent(body, '    ') + '''    else:
''' + textwrap.indent(old, '    ')
    return [
        ('float8_info = torch.finfo(current_platform.fp8_dtype())\n',
         'float8_info = torch.finfo(current_platform.fp8_dtype())\n'
         f'_FP8_CAUSAL_FULL = envs.{FLAG}\n'),
        ('@triton.jit\ndef kernel_unified_attention(', SOFTMAX + '@triton.jit\ndef kernel_unified_attention('),
        ('    MM_PREFIX_CLAMP_SW: tl.constexpr = False,\n',
         '    MM_PREFIX_CLAMP_SW: tl.constexpr = False,\n    FP8_FULL_CAUSAL: tl.constexpr = False,\n'),
        ('    if USE_TD:\n        tl.static_assert(\n', KERNEL_GUARD + '    if USE_TD:\n        tl.static_assert(\n'),
        (old, new),
        ('    kernel_unified_attention[grid](\n', HOST_GUARD + '    kernel_unified_attention[grid](\n'),
        ('        Q_IS_FP8=(q.dtype == current_platform.fp8_dtype()),\n',
         '        Q_IS_FP8=(q.dtype == current_platform.fp8_dtype()),\n        FP8_FULL_CAUSAL=_fp8_full_causal,\n'),
    ]


def generated_source(parent):
    result = parent
    changes = replacements(parent)
    for before, after in changes:
        assert result.count(before) == 1, before[:90]
        result = result.replace(before, after, 1)
    restored = result
    for before, after in reversed(changes):
        assert restored.count(after) == 1
        restored = restored.replace(after, before, 1)
    assert restored == parent
    compile(result, '<native-fp8-full-causal>', 'exec')
    before, after = functions(parent), functions(result)
    for name in before.keys() - {'kernel_unified_attention', 'unified_attention'}:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    # Boundary and default-OFF loop are actual exact-source copies. No changed
    # shared helpers, K/V addresses, reducer, output masks or scratch geometry.
    return result


def generated_env(parent):
    assert sha(parent) == ENV_SHA, 'Unknown Stable env registry'
    anchor = '    "VLLM_TRITON_FP8_MQ3D": lambda:'
    assert parent.count(anchor) == 1 and FLAG not in parent
    result = parent.replace(anchor, ENV_ENTRY + anchor, 1)
    assert result.replace(ENV_ENTRY, '', 1) == parent
    compile(result, '<native-fp8-full-env>', 'exec')
    return result
