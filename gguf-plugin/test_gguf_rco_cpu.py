#!/usr/bin/env python3
"""The build's gate for the GGUF step, CPU only, no GPU.

1. the out-of-tree GGUF plugin imports and registers: ``gguf`` is a
   quantization method, ``gguf`` a load format served by the plugin's loader,
   the compiled extension ``_C_gguf`` is present (importing it needs the CUDA
   driver, so on a CPU-only build a driver-library ImportError is reported
   and deferred to the server's first start; any other error fails);
2. the plugin's Qwen3.5 adapter maps every tensor name of the published
   IQ3_S file (``iq3s-tensor-names.json``, read from the file's header) to
   the HF name of this recipe's target, with the layer structure of a 64-layer
   hybrid (48 Gated DeltaNet layers, 16 attention layers at index % 4 == 3);
3. the draft bridge of ``0022-gguf-loader-draft-bridge.patch`` is installed
   and classifies references the way the engine needs it: a safetensors
   directory is not GGUF, a ``.gguf`` path is.

    python3 test_gguf_rco_cpu.py [--names iq3s-tensor-names.json]
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import sys
import tempfile


def fail(msg: str) -> None:
    print("FAIL: " + msg)
    sys.exit(1)


def check_plugin() -> None:
    import vllm_gguf_plugin

    vllm_gguf_plugin.register()
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
    from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER

    if "gguf" not in QUANTIZATION_METHODS:
        fail("gguf is not a registered quantization method")
    cfg_cls = get_quantization_config("gguf")
    if cfg_cls.__module__.split(".")[0] != "vllm_gguf_plugin":
        fail(f"gguf quantization config comes from {cfg_cls.__module__}, not the plugin")
    loader_cls = _LOAD_FORMAT_TO_MODEL_LOADER.get("gguf")
    if loader_cls is None or loader_cls.__module__.split(".")[0] != "vllm_gguf_plugin":
        fail(f"gguf load format is served by {loader_cls}, not the plugin")
    pkg_dir = os.path.dirname(vllm_gguf_plugin.__file__)
    ext = glob.glob(os.path.join(pkg_dir, "_C_gguf*.so"))
    if not ext:
        fail(f"compiled extension _C_gguf missing under {pkg_dir}")
    try:
        importlib.import_module("vllm_gguf_plugin._C_gguf")
        ext_state = "imports"
    except ImportError as e:  # the CUDA driver is not present at build time
        text = str(e)
        if "libcuda" in text or "cuda" in text.lower():
            ext_state = f"present, import deferred to the GPU ({text[:80]})"
        else:
            fail(f"extension import failed for another reason: {text[:200]}")
    print(f"plugin: registered (quantization + load format), extension {os.path.basename(ext[0])} {ext_state}")


LAYER_MODULES_GDN = {
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
    "linear_attn.conv1d.weight", "linear_attn.norm.weight", "linear_attn.out_proj.weight",
    "linear_attn.dt_bias", "linear_attn.A_log",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
}
LAYER_MODULES_ATTN = {
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
}
TOP_LEVEL = {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"}


def check_mapping(names_file: str) -> None:
    from vllm_gguf_plugin.weights_adapter.qwen3_5 import (
        _map_tensor_name,
        build_qwen35_text_mapper,
        build_qwen35_vision_mapper,
    )

    doc = json.load(open(names_file, encoding="utf-8"))
    tensors = doc["tensors"]
    text_mapper = build_qwen35_text_mapper(is_multimodal=True, is_moe=False)
    vision_mapper = build_qwen35_vision_mapper()
    mapped: dict[str, str] = {}
    unmapped: list[str] = []
    for t in tensors:
        name = t["name"]
        mapper = vision_mapper if name.startswith(("v.", "mm.")) else text_mapper
        m = _map_tensor_name(mapper, name)
        if m is None:
            unmapped.append(name)
        else:
            mapped[name] = m
    if unmapped:
        fail(f"{len(unmapped)} GGUF tensor names without an HF name: {unmapped[:8]}")
    per_layer: dict[int, set[str]] = {}
    top = set()
    vision = 0
    for hf in mapped.values():
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)$", hf)
        if m:
            per_layer.setdefault(int(m.group(1)), set()).add(m.group(2))
        elif hf.startswith("model.visual."):
            vision += 1
        else:
            top.add(hf)
    if top != TOP_LEVEL:
        fail(f"top-level tensors {sorted(top)} != {sorted(TOP_LEVEL)}")
    if sorted(per_layer) != list(range(64)):
        fail(f"layers present: {sorted(per_layer)[:5]}... expected 0..63")
    for i, mods in per_layer.items():
        want = LAYER_MODULES_ATTN if i % 4 == 3 else LAYER_MODULES_GDN
        if mods != want:
            fail(f"layer {i}: modules {sorted(mods ^ want)} differ from the expected set")
    types = sorted({t["type"] for t in tensors})
    print(f"mapping: {len(mapped)} tensors -> HF names, 64 layers with the hybrid structure, "
          f"{vision} vision tensors, types {types}")


def check_bridge() -> None:
    import vllm_gguf_plugin.loader as loader

    src = open(loader.__file__, encoding="utf-8").read()
    for marker in ("def _is_gguf_model(", "def _default_loader(", "gguf-rco draft bridge"):
        if marker not in src:
            fail(f"draft bridge marker missing in {loader.__file__}: {marker!r}")

    class Ref:  # the two ModelConfig fields the bridge reads
        def __init__(self, model, weights=None):
            self.model, self.model_weights = model, weights

    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "config.json"), "w").write("{}")
        if loader._is_gguf_model(Ref(d)):
            fail("a safetensors directory was classified as GGUF")
        if not loader._is_gguf_model(Ref(os.path.join(d, "model.gguf"))):
            fail("a .gguf path was not classified as GGUF")
        if not loader._is_gguf_model(Ref(d, os.path.join(d, "x.gguf"))):
            fail("model_weights pointing at a .gguf was not classified as GGUF")
    print("bridge: installed, classification of directory / .gguf / model_weights correct")

    # the config half: a plain HF checkpoint parsed through the GGUF config
    # parser (a speculative draft inherits config_format = gguf) keeps its
    # own architectures instead of the transformers causal-LM name
    import vllm_gguf_plugin.config_parser as cp

    if "gguf-rco draft bridge (config)" not in open(cp.__file__, encoding="utf-8").read():
        fail(f"config bridge marker missing in {cp.__file__}")
    with tempfile.TemporaryDirectory() as d:
        json.dump({"model_type": "qwen3", "architectures": ["DFlash2DraftModel"], "hidden_size": 64,
                   "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                   "intermediate_size": 128, "vocab_size": 256, "max_position_embeddings": 64},
                  open(os.path.join(d, "config.json"), "w"))
        config_dict, config = cp.GGUFConfigParser().parse(d, trust_remote_code=False)
        if config_dict.get("architectures") != ["DFlash2DraftModel"] or config.architectures != ["DFlash2DraftModel"]:
            fail(f"config bridge: architectures rewritten to {config.architectures}")
    print("bridge (config): a non-GGUF checkpoint keeps its own architectures")

    # the batched MMVQ (0023): the dispatch cap of 16 rows for every type is
    # in place; the kernel itself is compiled into the extension and checked
    # on the GPU
    import vllm_gguf_plugin.quantization.linear as lin  # noqa: F401 (the import wires the Gluon module too)

    src = open(lin.__file__, encoding="utf-8").read()
    if "gguf-rco step 6a" not in src or "mmvq_safe = 16" not in src:
        fail(f"batched MMVQ dispatch marker missing in {lin.__file__}")
    print("mmvq (0023): the 16-row dispatch cap is installed")

    # the Gluon decode kernels (0024): the module is installed in the package,
    # its dispatch is wired into _fused_mul_mat_gguf, the IQ3_S grid table
    # matches the gguf package's grid; the kernel itself is checked on the GPU
    if "gguf-rco step 6d" not in src or "gluon_mul_mat(" not in src:
        fail(f"Gluon dispatch marker missing in {lin.__file__}")
    if "gguf-rco step 6f" not in src:
        fail(f"prefill dispatch marker (0027) missing in {lin.__file__}")
    if "gguf-rco step 6f" not in src:
        fail(f"prefill dispatch marker (0027) missing in {lin.__file__}")
    if "gguf-rco step 6f" not in src:
        fail(f"prefill dispatch marker (0027) missing in {lin.__file__}")
    from vllm_gguf_plugin.triton.gluon import interface as gi
    from vllm_gguf_plugin.triton.gluon import iq3s as gk

    if not {21, 23, 18, 12, 22, 17, 16, 10} <= set(gi.GLUON_TYPES) or gi.GLUON_MAX_ROWS != 16:
        fail(f"Gluon dispatch table: types {sorted(gi.GLUON_TYPES)}, max rows {gi.GLUON_MAX_ROWS}")
    from vllm_gguf_plugin.triton.gluon import iq2, iq3xxs, iq4xs, q2k, q4k  # noqa: F401  (the modules import)
    from vllm_gguf_plugin.triton.gluon import iq2_int8, iq3s_int8, iq3xxs_int8, iq4xs_int8, q4k_int8  # noqa: F401  (the int8 forms import)
    from vllm_gguf_plugin.triton.gluon import splitk
    from vllm_gguf_plugin.triton.gluon.iq3s import split_k_for as _sk_iq3s

    if _sk_iq3s is not splitk.split_k_for or splitk.split_k_for(20, 17408) != 4 or splitk.split_k_for(68, 5120) != 2 or splitk.split_k_for(20, 1024) != 5 or splitk.split_k_for(20, 248320) != 1:
        fail("the split-K table (0033)")
    if splitk.split_k_for(20, 17408, m=4, block_bytes=110) != 4 or splitk.split_k_for(20, 17408, m=16, block_bytes=110) != 2 or splitk.split_k_for(20, 1024, m=8, block_bytes=110) != 5:
        fail("the split-K cap by the batch rows (0034)")
    from vllm_gguf_plugin.triton.gluon import iq3s_int8c

    if gi._INT8_KERNELS.get(21) is not iq3s_int8c.iq3s_int8c_gemm or iq3s_int8c.TAB_OFF != 24576 or iq3s_int8c.BLOCK_BYTES != 110:
        fail("the fourth IQ3_S form's dispatch (0035)")
    from vllm_gguf_plugin.triton.gluon import iq3s_int8t

    if iq3s_int8t.TAB_OFF != 24576 or iq3s_int8t.TILE_ROW_BYTES != 112 or not callable(getattr(gi, "gluon_mul_mat_iq3s_tiles", None)):
        fail("the tile-major IQ3_S form's dispatch (0036)")
    import torch  # noqa: F811  (the function imports torch further down; the name is local to it)

    raw = torch.arange(70 * 220, dtype=torch.int32).remainder(251).to(torch.uint8).reshape(70, 220)
    tiles = iq3s_int8t.repack_iq3s_tiles(raw, 70)
    if (
        tuple(tiles.shape) != (2, 2, 64, 112)
        or not torch.equal(tiles[0, 1, 5, 0:64], raw[5, 112:176])          # qs of block 1 of row 5
        or not torch.equal(tiles[0, 1, 5, 64:72], raw[5, 176:184])         # qh
        or not torch.equal(tiles[0, 1, 5, 72:104], raw[5, 184:216])        # signs
        or not torch.equal(tiles[0, 1, 5, 104:108], raw[5, 216:220])       # scales
        or not torch.equal(tiles[0, 1, 5, 108:110], raw[5, 110:112])       # d, last
        or int(tiles[0, 1, 5, 110:112].sum()) != 0 or int(tiles[1, :, 6:, :].sum()) != 0   # the padding, the rows past 70
        or not torch.equal(tiles[1, 0, 5, 0:64], raw[69, 2:66])
    ):
        fail("the tile-major repack (0036)")
    from vllm_gguf_plugin.triton.gluon import iq3s_tile_dequant  # noqa: F401  (0037: the prefill's dequantiser on the tiles imports)

    import inspect as _inspect

    if "dequantize_iq3s_tiles" not in _inspect.getsource(gi.gluon_mul_mat_iq3s_tiles):
        fail("the tiles op's prefill branch (0037)")
    from vllm_gguf_plugin.triton.gluon import iq2_int8t, iq3xxs_int8t, iq4xs_int8t, q4k_int8t, tiles  # noqa: F401  (0038: the other six types' tile-major forms import)

    if set(tiles.TILE_TYPES) != {21, 23, 12, 18, 16, 17, 22, 8, 11, 13, 14} or set(gi.GLUON_TILE_TYPES) != set(tiles.TILE_TYPES) or not callable(getattr(gi, "gluon_mul_mat_tiles", None)):
        fail("the tile types' registry (0038)")
    if "dequantize_tiles(tiles, n_out, x.shape[1], weight_type, x.dtype)" not in _inspect.getsource(gi.gluon_mul_mat_tiles):
        fail("the tiles op's prefill branch (0038)")
    for wt, (block, row_bytes, d_last) in tiles.TILE_TYPES.items():
        raw = torch.arange(70 * 3 * block, dtype=torch.int32).remainder(251).to(torch.uint8).reshape(70, 3 * block)
        tt = tiles.repack_tiles(raw, 70, wt)
        if d_last is None:   # the stage-major layouts (kq_tiles): the rows past 70 sit inside every stage
            from vllm_gguf_plugin.triton.gluon import kq_tiles as _kq

            pad = tt.reshape(2, 3, _kq.NST[wt], 64, row_bytes // _kq.NST[wt])[1, :, :, 6:, :]
        else:
            pad = tt[1, :, 6:, :]
        if tuple(tt.shape) != (2, 3, 64, row_bytes) or not torch.equal(tiles.unrepack_tiles_torch(tt, 70, wt), raw) or int(pad.sum()) != 0:
            fail(f"the tile repack / un-repack round trip of type {wt} (0038)")
        if d_last is True and not (torch.equal(tt[0, 1, 5, block - 2:block], raw[5, block:block + 2]) and torch.equal(tt[0, 1, 5, :block - 2], raw[5, block + 2:2 * block]) and int(tt[0, 1, 5, block:].sum()) == 0):
            fail(f"the d-last tile layout of type {wt} (0038)")
        if d_last is False and not torch.equal(tt[0, 1, 5, :block], raw[5, block:2 * block]):
            fail(f"the tile layout of type {wt} (0038)")
        if d_last is None:
            # the K-quant / Q8_0 layouts: stage-major, tile byte j of stage h of row 5 (k-block 1) is block byte tile_map[h * hb + j]
            from vllm_gguf_plugin.triton.gluon import kq_tiles

            nst, hb = kq_tiles.NST[wt], row_bytes // kq_tiles.NST[wt]
            m = kq_tiles.tile_map(wt)
            view = tt.reshape(tt.shape[0], tt.shape[1], nst, 64, hb)
            if (
                len(m) != row_bytes or sorted(set(k for k in m if k >= 0)) != list(range(block)) or kq_tiles.BLOCK_BYTES[wt] != block or kq_tiles.ROW_BYTES[wt] != row_bytes
                or not all(int(view[0, 1, h, 5, j]) == (int(raw[5, block + m[h * hb + j]]) if m[h * hb + j] >= 0 else 0) for h in range(nst) for j in range(hb))
            ):
                fail(f"the stage-major tile layout of type {wt} (kq_tiles)")
    from vllm_gguf_plugin.triton.gluon import splitk_reduce  # noqa: F401  (0039: the in-kernel split-K reduction imports)

    for _g in (iq3s_int8t.iq3s_int8t_gemm, iq4xs_int8t.iq4xs_int8t_gemm, q4k_int8t.q4k_int8t_gemm, iq3xxs_int8t.iq3xxs_int8t_gemm, iq2_int8t.iq2_int8t_gemm):
        if _inspect.signature(_g).parameters["reduce"].default is not False:
            fail(f"the in-kernel split-K reduction must stay optional in {_g.__name__} (0039 measured it at +2.5 % on C1; 0040)")
    if not callable(getattr(splitk_reduce, "splitk_counters", None)) or "atomic_add" not in _inspect.getsource(splitk_reduce):
        fail("the split-K reduction helper (0039)")
    if not callable(getattr(gi, "gluon_mul_mat_tiles_multi", None)) or "quantize_activations(x.contiguous(), with_sums=True)" not in _inspect.getsource(gi.gluon_mul_mat_tiles_multi):
        fail("the merged layer's one-op path (0040)")
    if "out" not in _inspect.signature(tiles.tiles_gemm).parameters or any("out" not in _inspect.signature(_g).parameters for _g in (iq3s_int8t.iq3s_int8t_gemm, iq4xs_int8t.iq4xs_int8t_gemm, q4k_int8t.q4k_int8t_gemm, iq3xxs_int8t.iq3xxs_int8t_gemm, iq2_int8t.iq2_int8t_gemm)):
        fail("the launchers' output view (0040)")
    from vllm_gguf_plugin.triton.gluon import iq3s_int8te  # noqa: F401  (0041: the IQ3_S decode on the tile sixth form imports)

    if "iq3s_int8te_gemm" not in _inspect.getsource(tiles._GEMM[21]) or not callable(getattr(iq3s_int8te, "quantize_activations_256", None)) or iq3s_int8te.TAB_OFF != 24576:
        fail("the IQ3_S decode on the tile sixth form (0041)")
    from vllm_gguf_plugin.triton.gluon import tiles_grouped  # noqa: F401  (0042: the grouped kernel imports)

    if (
        not callable(getattr(gi, "gluon_mul_mat_tiles_grouped", None))
        or "quantize_activations_256(x.contiguous(), with_sums=True)" not in _inspect.getsource(gi.gluon_mul_mat_tiles_grouped)
        or tiles_grouped.STAGE != 2304 or tiles_grouped.TAB_W != 1792 or tiles_grouped.GROUPED_SPLITS.get((544, 20)) != 5
    ):
        fail("the grouped kernel's op (0042)")
    # the loader's step on CPU: two shards (IQ3_S 70 rows, Q4_K 64 rows) over three k-blocks packed with their tables
    raw21 = torch.arange(70 * 3 * 110, dtype=torch.int32).remainder(251).to(torch.uint8).reshape(70, 3 * 110)
    raw12 = torch.arange(64 * 3 * 144, dtype=torch.int32).remainder(241).to(torch.uint8).reshape(64, 3 * 144)
    t21, t12 = tiles.repack_tiles(raw21, 70, 21), tiles.repack_tiles(raw12, 64, 12)
    packed, views, descs, splits, nb, block_bytes = tiles_grouped.prepare_grouped_layer([t21, t12], [70, 64], [21, 12])
    if packed.numel() != t21.numel() + t12.numel() or nb != 3 or not torch.equal(views[0], t21) or not torch.equal(views[1], t12) or not views[1].is_contiguous():
        fail("the packed tiles and their views (0042)")
    if splits != sorted({tiles_grouped.grouped_split_for(3, 3, m=m, block_bytes=block_bytes) for m in range(1, 17)}) or len(descs) != len(splits):
        fail("the descriptor tables per split (0042)")
    d = descs[splits.index(1)]
    if tuple(d.shape) != (3, 16) or d[:, 1].tolist() != [21, 21, 12] or d[:, 3].tolist() != [0, 64, 70] or d[:, 4].tolist() != [64, 6, 64] or d[2, 0].item() != t21.numel() // 4 or d[1, 0].item() != 3 * 64 * 28:
        fail("the descriptor rows (0042)")
    if "gguf-rco step 6h (0042)" not in open(lin.__file__, encoding="utf-8").read() or not callable(getattr(lin, "fused_mul_mat_gguf_tiles_grouped", None)):
        fail("the grouped op in the plugin (0042)")
    if (
        tiles_grouped.REGIONS[1]["aoff"] != 6144 or tiles_grouped.REGIONS[0]["stage"] != 2304
        or "amode" not in _inspect.signature(tiles_grouped.grouped_gemm).parameters
        or "0043: the launcher's variant" not in _inspect.getsource(gi.gluon_mul_mat_tiles_grouped)
    ):
        fail("the grouped kernel's latency hiding (0043)")
    # 0044: the balanced split partition - every split index owns a non-empty k-range and the ranges tile nb. The
    # ceil partition of 0042 skipped the indices with s * ceil(nb / splitk) >= nb (20 k-blocks over 8 or 6 splits:
    # the 6144- and 8192-wide projections at their loader splits), so those partial slices were never written and
    # the fp32 sum added torch.empty's contents to the output.
    meta44 = {"nb": 20, "shards": [(0, 21, 28, 6144, 96, 0)]}
    for sk in (3, 5, 6, 7, 8):
        d44 = tiles_grouped.descriptors(meta44, sk, "cpu")
        first = d44[d44[:, 3] == 0]
        if (
            tuple(d44.shape) != (96 * sk, 16) or first.shape[0] != sk
            or first[:, 5].tolist() != [(s * 20) // sk for s in range(sk)] or first[:, 6].tolist() != [((s + 1) * 20) // sk for s in range(sk)]
            or first[:, 7].tolist() != list(range(sk)) or int((first[:, 6] <= first[:, 5]).sum()) != 0
        ):
            fail(f"the split partition at {sk} splits (0044)")

    # the K-quant / Q8_0 tile types (kq_tiles.py): grouped-kernel only - the stage rows, the NEW mask, the wide regions, the dispatch
    from vllm_gguf_plugin.triton.gluon import kq_tiles, kq_ref  # noqa: F401  (the layouts and the CPU reference models import)

    _txtkq = open(tiles_grouped.__file__, encoding="utf-8").read()
    if (
        kq_tiles.NST != {8: 2, 11: 1, 13: 1, 14: 2} or kq_tiles.NEW_BIT != {8: 1, 11: 2, 13: 4, 14: 8}
        or {wt: tiles_grouped.STAGE_ROW_WORDS[wt] for wt in (8, 11, 13, 14, 12, 21)} != {8: 34, 11: 28, 13: 44, 14: 27, 12: 36, 21: 28}
        or tiles_grouped.STAGE_WIDE != 2816 or tiles_grouped.REGIONS[5]["stage"] != 2816 or tiles_grouped.REGIONS[6]["aoff"] != 6144 or tiles_grouped.REGIONS[7]["stage"] != 2816
        or any(f"if NEW & {b}:" not in _txtkq or f"_run({wt}, {tiles_grouped.ROW_WORDS[wt]}, {kq_tiles.NST[wt]}, E," not in _txtkq for wt, b in kq_tiles.NEW_BIT.items())
        or "for h in gl.static_range(NST):" not in _txtkq or "region = {0: 5, 1: 6, 3: 7}[region]" not in _inspect.getsource(tiles_grouped.grouped_gemm)
        or "STAGE_ROW_WORDS[wt] for wt in weight_types" not in _inspect.getsource(gi.gluon_mul_mat_tiles_grouped)
    ):
        fail("the K-quant / Q8_0 tile types (kq_tiles)")
    # a packed layer with a Q6_K shard: the descriptor's row words are the k-block's (54), the region choice reads the stage row (27)
    raw14 = torch.arange(70 * 3 * 210, dtype=torch.int32).remainder(251).to(torch.uint8).reshape(70, 3 * 210)
    t14 = tiles.repack_tiles(raw14, 70, 14)
    p14, v14, d14, s14, nb14, bb14 = tiles_grouped.prepare_grouped_layer([t21, t14], [70, 70], [21, 14])
    d14_1 = d14[s14.index(1)]
    if tuple(t14.shape) != (2, 3, 64, 216) or d14_1[:, 1].tolist() != [21, 21, 14, 14] or d14_1[:, 2].tolist() != [28, 28, 54, 54] or d14_1[:, 8].tolist() != [64 * 28, 64 * 28, 64 * 54, 64 * 54] or bb14 != 300 or not torch.equal(v14[1], t14):
        fail("the packed layer with a Q6_K shard (kq_tiles)")
    # the CPU reference models reproduce gguf-py's dequantiser on random blocks of every type
    import numpy as np
    from gguf.quants import dequantize as _gg_deq
    from gguf.constants import GGMLQuantizationType as _GT

    _rng = np.random.default_rng(3)
    for wt in (8, 11, 13, 14):
        _raw = kq_ref.random_rows(_rng, 6, 512, wt)
        if not np.array_equal(kq_ref.dequantize(kq_ref.parse(_raw, wt)), _gg_deq(_raw, _GT(wt)).reshape(6, 512)):
            fail(f"the CPU reference model of type {wt} against gguf.quants (kq_ref)")
        if not torch.equal(tiles.unrepack_tiles_torch(tiles.repack_tiles(torch.from_numpy(_raw), 6, wt), 6, wt), torch.from_numpy(_raw)):
            fail(f"the repack round trip of type {wt} on random rows (kq_tiles)")

    if (
        not {21, 23, 18, 12, 22, 17, 16} <= set(gi.GLUON_INT8_TYPES)
        or (iq3s_int8.BLOCK_BYTES, iq4xs_int8.BLOCK_BYTES, iq3xxs_int8.BLOCK_BYTES, q4k_int8.BLOCK_BYTES) != (110, 136, 98, 144)
        or iq2_int8.BLOCK_BYTES != {16: 66, 17: 74, 22: 82}
    ):
        fail(f"the int8 dispatch table: types {sorted(gi.GLUON_INT8_TYPES)}")

    if iq2.BLOCK_BYTES != {16: 66, 17: 74, 22: 82} or q2k.BLOCK_BYTES != 84:
        fail("IQ2 / Q2_K block sizes")

    if iq4xs.KVALUES != (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113):
        fail("IQ4_XS value table")
    import torch
    from gguf.quants import IQ3_S

    IQ3_S.init_grid()
    g = gk.grid32(torch.device("cpu"))
    if g.shape != (512,) or g.dtype != torch.int32 or int(g[0].item() & 0xFF) != int(IQ3_S.grid[0, 0][0, 0]):
        fail("Gluon IQ3_S grid table does not match the gguf package's grid")
    if gk.split_k_for(20, 10240) != 1 or gk.split_k_for(68, 5120) != 2:     # 0033: the measured table (attn_qkv unsplit, ffn_down at 2)
        fail(f"split-K table: {gk.split_k_for(20, 10240)}, {gk.split_k_for(68, 5120)}")
    # 0046: the 32-row form - bm in the launcher, regions 3 and 4 with the 32-row A stages, the keep-alive sentinel only
    # in region 0, the split tables to 32 rows with the budget's row count saturating at 16, the dispatch's one launch
    # up to 32 rows and the arrival loop to 128
    _src46 = _inspect.getsource(tiles_grouped.grouped_gemm)
    _txt46 = open(tiles_grouped.__file__, encoding="utf-8").read()
    _gi46 = _inspect.getsource(gi.gluon_mul_mat_tiles_grouped)
    if (
        "bm" not in _inspect.signature(tiles_grouped.grouped_gemm).parameters
        or tiles_grouped.REGIONS.get(3, {}).get("stage") != 2304 or tiles_grouped.REGIONS.get(4, {}).get("stage") != 1792
        or "elif REGION == 3:" not in _txt46 or "elif REGION == 4:" not in _txt46
        or "only region 0's second allocation is otherwise untouched" not in _txt46
        or "bm = 16 if M <= 16 else 32" not in _src46 or "region = 4 if max_rw <= 28 else 3" not in _src46
        or "min(m, 16)" not in _txt46 or "ms=range(1, 33)" not in _txt46
        or not getattr(tiles_grouped, "GROUPED_SPLITS_M32", None) or (32, 20) not in tiles_grouped.GROUPED_SPLITS_M32
        or "bm=16 if r1 - r0 <= 16 else 32" not in _gi46 or gi.GLUON_MAX_ROWS_GROUPED != 32 or gi.GLUON_ARRIVAL_ROWS != 128
    ):
        fail("the 32-row form (0046)")
    _p46 = tiles_grouped.prepare_grouped_layer([t21, t12], [70, 64], [21, 12])
    if _p46[3] != sorted({tiles_grouped.grouped_split_for(3, 3, m=m, block_bytes=_p46[5]) for m in range(1, 33)}) or any(
            tiles_grouped.grouped_split_for(3, 3, m=m, block_bytes=_p46[5]) != tiles_grouped.grouped_split_for(3, 3, m=16, block_bytes=_p46[5]) for m in (17, 24, 32)):
        fail("the split tables to 32 rows, the 16-row split above 16 (0046)")
    # 0047: the wide forms - bm 64 (four warps, region 8) and bm 128 (eight warps, region 9: the 64 KB A stages as the first
    # allocation, the weight stages raw-addressed at word 16384 of the second), the grid over M-blocks, split 1 in every
    # layer's tables, the dispatch above 32 rows for runs without a min type (bm 64 to 64 rows, bm 128 above). The scale prefetch and the A stages' 16-byte
    # swizzle are wide-form branches (in the decode forms they cost 29 registers and 4-5 %), so both guards are pinned here.
    _src47 = _inspect.getsource(tiles_grouped.grouped_gemm)
    _gi47 = _inspect.getsource(gi.gluon_mul_mat_tiles_grouped)
    if (
        tiles_grouped.REGIONS.get(8, {}).get("stage") != 2304 or tiles_grouped.REGIONS.get(9, {}).get("stage") != 2304
        or "elif REGION == 8:" not in _txt46 or "elif REGION == 9:" not in _txt46
        or "SOFF: gl.constexpr = 16384 if REGION == 9 else 0" not in _txt46 or "(_smem_base(ic) + SOFF * 4)" not in _txt46
        or "warps_per_cta=[NW // 4, 4]" not in _txt46 or "warp_bases=[[8, 0], [16, 0], [0, 0]]" not in _txt46
        or "r0 = gl.program_id(1) * BM" not in _txt46 or "SwizzledSharedLayout(16, 1, 8, [1, 0])" not in _txt46
        or "sx_n = gl.load(sxrow + kb + 1, mask=ms_ok & (kb + 1 < kb1), other=0.0)" not in _txt46
        or "if E == 1 and BM >= 64:" not in _txt46
        or "cpa: gl.constexpr = BlockedLayout([1, 1], [2, 16], [NW, 1], [1, 0])" not in _txt46
        or "tiles_grouped_kernel[(desc.shape[0], n_mblocks)](" not in _src47 or "region = 8 if bm == 64 else 9" not in _src47
        or "num_warps = 4 if bm == 64 else 8" not in _src47 or "| {1})" not in _txt46
        or gi.GLUON_DEQUANT_TYPES != frozenset({12, 13, 22}) or "\"desc\": {1: descs[desc_splits.index(1)]}" not in _gi47
        or "bm=64 if M <= 64 else 128)" not in _gi47 or "M > GLUON_MAX_ROWS_GROUPED and not (set(weight_types) & GLUON_DEQUANT_TYPES)" not in _gi47
        or "set(weight_types) & GLUON_DEQUANT_TYPES" not in _gi47 or 1 not in _p46[3]
    ):
        fail("the wide forms (0047)")
    print("gluon (0024 + 0025 + 0028 + 0030 + 0031 + 0032 + 0033 + 0034 + 0035 + 0036 + 0037 + 0038 + 0039 + 0040 + 0041 + 0042 + 0043 + 0044 + 0046 + 0047 + 0050 + kq): the decode kernels of all eight matmul types of the IQ3_S file are installed and dispatched at up to 16 rows (32 on row halves); every type but Q2_K tile-major only and through the grouped kernel (one op, one launch per layer; per-256 activations on every type), Q3_K / Q5_K / Q6_K / Q8_0 tile-major through the same kernel; the split-K from the measured tables, capped per batch; above 32 rows the wide form for runs without Q4_K / Q5_K / IQ2_S, bm 64 on four warps from 33 to 64 rows and bm 128 on eight warps above, one M-block to 128 rows and ceil(M / 128) above")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--names", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "iq3s-tensor-names.json"))
    a = p.parse_args()
    check_plugin()
    check_mapping(a.names)
    check_bridge()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
