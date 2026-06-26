# Experimental Setup

**Paper section**: 4, Appendix D

## Models

| Model | HF ID | Reasoning mode | Evaluated on |
|-------|-------|----------------|--------------|
| Qwen3-4B | Qwen/Qwen3-4B | Thinking (AIME/MATH/HumanEval), non-thinking (IFEval/line-retrieval) | All benchmarks |
| Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct | No reasoning variant | IFEval, line-retrieval |
| Phi-4 | microsoft/phi-4 | Greedy | IFEval, line-retrieval |
| Phi-4-reasoning-plus | microsoft/Phi-4-reasoning-plus | Reasoning (temp 0.8, top-p 0.95, top-k 50) | AIME24, MATH500, HumanEval |

## Decoding parameters

| Model | Temp | Top-p | Top-k | Max tokens |
|-------|------|-------|-------|------------|
| Qwen3-4B (thinking) | 0.6 | 0.95 | 20 | 8192 (MATH), 16384 (AIME, HumanEval) |
| Qwen3-4B (non-thinking) | 0.7 | 0.8 | 20 | 1280 (IFEval) |
| Llama-3.1-8B | 0.6 | 0.9 | - | - |
| Phi-4 | greedy | - | - | - |
| Phi-4-reasoning-plus | 0.8 | 0.95 | 50 | - |

## Benchmarks

| Benchmark | Problems | Max new tokens | Metric | Runs |
|-----------|----------|----------------|--------|------|
| AIME 2024 | 30 | 16384 | Accuracy (exact integer answer) | Avg@3 |
| MATH-500 | 500 | 8192 | Accuracy (extract from `\boxed{}`) | Avg@3 |
| HumanEval | 164 (extended) | 16384 | Pass@1 (execution, 30s timeout) | Avg@3 |
| IFEval | 541 | 1280 | Prompt-level strict & loose accuracy | 1 |
| Line-retrieval | 100 per L | 32 | Exact-match 10-char code | 1 |

## Line-retrieval protocol

[App D] Contexts of L in {100, 200, 300, 400, 500, 600} numbered lines, each with
a random 10-character alphanumeric code (A-Z, 0-9). Model must retrieve the code
at a randomly chosen line. Prompt:

```
Below is a list of numbered lines, each with a unique 10-character alphanumeric code.
line 1: <c1>
line 2: <c2>
...
What is the code on line k? Reply with only the 10-character code, nothing else.
```

Answer = last 10-character alphanumeric token in the output.

## NiaH protocol

[App D] Paul Graham essays as haystack. Needle inserted at depth d in {0.1, ..., 0.9, 1.0}.
Context lengths 10%-100% of Qwen3-4B's 32768-token window (3112-31129 tokens).
10x10 evaluation grid. max_new_tokens=128. Scoring: 10 (both "sandwich" and "Dolores"),
5 (one), 0 (neither).

Two settings:
- **Static**: standard quantize-once-then-attend (prior work).
- **Accumulated**: pseudo-decode setting (this paper's contribution).

## Quantization configuration

[App D] All quantized runs:
- 2-bit KV cache compression
- Three-region layout: S sink (FP16), quantized body (groups of G), R recent (FP16)
- G = 128, S = 128, R = 128 (except IFEval: S = 32)
- KVarN auxiliary params: zeropoints and scales at 8-bit precision
- Mixed-precision baselines (KVQuant, PolarQuant, TurboQuant) retain their
  original per-element precision allocation inside each group

## Baselines

| Method | K/V | bits/elem | Notes |
|--------|-----|-----------|-------|
| KIVI | 2/2 | 2.3 | Per-channel K, per-token V, single scale |
| QuaRot | 2/2 | 2.3 | Hadamard rotation, no VarN |
| KVQuant | 2/2 | 2.4 | 1% channels in FP16 (mixed precision) |
| PolarQuant | 4/2 | 3.3 | Polar coordinate decomposition; needs 4-bit K |
| TurboQuant | 3/3 | 4.6 | vLLM community impl; first/last 2 layers unquantized |
| Kitty | 2/2 | 2.4 | Top 12% channels at 4-bit |

## Compute cost

[App J] ~50 GPU-days on a 500 TFLOP (fp16), 1.8 TB/s bandwidth GPU to reproduce
all MATH500, AIME24, HumanEval, and IFEval experiments.