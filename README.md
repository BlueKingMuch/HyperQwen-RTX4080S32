# HyperQwen on an RTX 4080 SUPER (32 GB)

A personal experiment fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen),
kept public in case it is useful to someone else with an Ada card. 
Everything that makes this work is upstream. 
What happens here is one card, one setup, and what I can measure on it.

**This is a vibed fork and may be abandoned at any time.** 
For a maintained project, a working install and the documentation, go upstream. 
Start with its [README](https://github.com/syv-ai/HyperQwen#readme). 
If anything here turns out to be worth keeping, the maintainer is welcome to take it; that is what the fork
link is for.

## The card

RTX 4080 SUPER 32 GB (sm89, Ada), capped at 250 W and undervolted, Windows 11 with WSL2 and Docker. 

Upstream's published numbers are an RTX 3090 (sm86), so the question this fork exists to answer is what transfers and what does not.

First data point, upstream's own harness at `c0c81bb`, second run, setup B:
[field report #149](https://github.com/syv-ai/HyperQwen/issues/149).

Single stream lands a few percent below the 3090 reference because decode is bandwidth-bound there and the 3090 has more of it.
Four and eight concurrent requests run roughly a third faster. GSM8K 0.960.

## What is different here

Nothing yet. Planned, not promised:

- vLLM 0.29 instead of 0.28 (upstream has this open as
  [#106](https://github.com/syv-ai/HyperQwen/issues/106))
- implementing the [GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)
  route for the same model, but set up a little cleaner, through an external fork of
  [vllm-gguf-plugin](https://github.com/BlueKingMuch/vllm-gguf-plugin).
  My own tests showed that this quantisation is, quality-wise and at least for my use cases,
  very close to W4A16. At one or two concurrent requests it measured comparable speeds
  and bought roughly twice the KV pool. Beyond that it gets slower: the 3-bit weights are
  unpacked before they are multiplied, and the kernel doing it stops grouping past 32 rows.

Anything that lands here will say how it was measured, with the command that
produced the numbers. Anything that is a guess will say that it is a guess.

## Licence

Apache-2.0, inherited from upstream. The Qwen3.8-27B weights and the quantised
checkpoints used here publish under Apache-2.0 as well, and the patches carried
here are derivative works of vLLM, also Apache-2.0.
