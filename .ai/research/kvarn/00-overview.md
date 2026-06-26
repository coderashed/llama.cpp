# KVarN: Variance-Normalized KV-Cache Quantization

**Paper**: arXiv:2606.03458v1 (June 2026)
**Authors**: Lorenz K. Muller, Philippe Bich, Chiara Boretti, Hyun-Min Chang, Jiawei Zhuang, Lukas Cavigelli (Huawei)
**Code**: https://github.com/huawei-csl/KVarN (vLLM implementation)

## One-line summary

KVarN is a calibration-free KV-cache quantizer that combines a channel-dimension
Hadamard rotation with a dual-axis variance normalization (Sinkhorn-style),
then round-to-nearest quantizes to 2 bits. It targets the token-scale errors
that drive outlier degradation and accumulate across decoding timesteps in
long-form reasoning.

## Problem

- Test-time scaling (long reasoning chains) makes the KV-cache the memory bottleneck.
- Existing 2-bit KV quantizers (KIVI, KVQuant, etc.) are evaluated on prefill-style
  benchmarks that do not capture error accumulation across autoregressive steps.
- Under autoregressive decoding, quantization error at layer `l` corrupts the K/V
  produced by layer `l+1`, compounding over timesteps.
- The largest (top ~5%) errors cause most end-to-end degradation; these are
  overwhelmingly driven by incorrect per-token magnitude (scale), not direction.

## Method (KVarN)

```
K, V (fp16)
  |
  v  [1] Channel-dim Hadamard rotation (block-diagonal, head-wise)
  |
  v  [2] Dual-axis variance normalization (Alg 1, ~8 iters, per 128-token block)
  |      -> produces scale_c (FP8) and scale_r (FP8)
  |
  v  [3] Round-to-nearest quantize with zeropoint (FP16) per group of 128
  |
  v  Store: 2-bit quantized values + 2x FP8 scales + 1x FP16 zeropoint
           = 2.25 bits/element effective
```

Dequant: `K_dq = (K_q + z) * s1 * s2` (dual-scale fused into single kernel).

## Key results (2-bit, 2.3 bits/elem effective)

| Benchmark | KIVI | KVarN | FP16 |
|-----------|------|------|------|
| AIME24 (Qwen3-4B) | 55.5% | **60.0%** | 61.1% |
| MATH500 (Qwen3-4B) | 77.8% | **79.2%** | 82.6% |
| HumanEval (Qwen3-4B) | 86.4% | **88.4%** | 88.8% |
| IFEval strict (Qwen3-4B) | 80.3% | **80.4%** | 81.0% |

Overhead: 0.18% quantization latency, <1.4% dequantization latency vs KIVI.

## Parts documented here

| # | Topic | File |
|---|-------|------|
| 01 | Problem & motivation | [01-problem-motivation.md](01-problem-motivation.md) |
| 02 | Error decomposition (magnitude vs direction) | [02-error-decomposition.md](02-error-decomposition.md) |
| 03 | Hadamard transform arrangement | [03-hadamard-arrangement.md](03-hadamard-arrangement.md) |
| 04 | Variance normalization (Algorithm 1) | [04-variance-normalization.md](04-variance-normalization.md) |
| 05 | KVarN pipeline & storage format | [05-kvarn-pipeline.md](05-kvarn-pipeline.md) |
| 06 | Three-region cache layout | [06-three-region-layout.md](06-three-region-layout.md) |
| 07 | Dequantization | [07-dequantization.md](07-dequantization.md) |
| 08 | Pseudo-decode evaluation | [08-pseudo-decode-eval.md](08-pseudo-decode-eval.md) |
| 09 | Experimental setup | [09-experimental-setup.md](09-experimental-setup.md) |
| 10 | Results summary | [10-results-summary.md](10-results-summary.md) |
| 11 | Limitations | [11-limitations.md](11-limitations.md) |