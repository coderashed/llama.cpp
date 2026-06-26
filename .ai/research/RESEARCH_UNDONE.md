# Research Backlog — KVarN Paper

Source: https://arxiv.org/html/2606.03458v1
Title: KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks
Authors: Lorenz K. Muller, Philippe Bich, Chiara Boretti, Hyun-Min Chang, Jiawei Zhuang, Lukas Cavigelli (Huawei)
Reference impl: https://github.com/huawei-csl/KVarN

## Instructions

Each entry documents one part of the paper with implementation-relevant detail.
Pipeline: research.
Exit criterion: `kvarn/<NN>-<slug>.md` exists and describes the concept, math,
data flow, and concrete parameters needed for implementation, with references
to paper section numbers.

---

### 00 — Overview

- **Status**: pending
- **Depends on**: nothing
- **Exit criterion**: `kvarn/00-overview.md` summarizes the paper, its
  contributions, and how KVarN fits into the KV-cache quantization landscape.

### 01 — Problem & Motivation

- **Status**: pending
- **Depends on**: nothing
- **Exit criterion**: `kvarn/01-problem-motivation.md` describes error
  accumulation under autoregressive decoding, why prefill-style eval misses it,
  and the outlier-driven quality degradation mechanism (Sec 1, 3.2).

### 02 — Error Decomposition (Magnitude vs Direction)

- **Status**: pending
- **Depends on**: 01
- **Exit criterion**: `kvarn/02-error-decomposition.md` derives the
  magnitude/directional error split (Eq 3), shows magnitude dominates outliers,
  and explains why fixing token scale matters (Sec 3.1, Figs 1a, 3).

### 03 — Hadamard Transform Arrangement

- **Status**: pending
- **Depends on**: 01
- **Exit criterion**: `kvarn/03-hadamard-arrangement.md` describes the
  channel-dimension rotation, which transforms are absorbed vs online,
  block-diagonal head-wise layout, and complexity (Sec 2.2, App A, Fig 7).

### 04 — Variance Normalization (Algorithm 1)

- **Status**: pending
- **Depends on**: 03
- **Exit criterion**: `kvarn/04-variance-normalization.md` reproduces
  Algorithm 1 (log-domain dual-scale Sinkhorn-style balancing), defines
  Imb(), iteration count K, clamp limits, and the returned scales (App H).

### 05 — KVarN Pipeline & Storage Format

- **Status**: pending
- **Depends on**: 03, 04
- **Exit criterion**: `kvarn/05-kvarn-pipeline.md` describes the full pipeline
  (rotate -> VarN -> RTN), the stored tensors (2-bit quantized, FP8 scale,
  FP8 second scale, FP16 zeropoint), bits/element accounting, and the
  dequant formula (Sec 2.3, 3.3, App D).

### 06 — Three-Region Cache Layout

- **Status**: pending
- **Depends on**: 05
- **Exit criterion**: `kvarn/06-three-region-layout.md` describes the S/G/R
  layout (sink FP16, quantized body, trailing FP16), default sizes (128/128/128,
  S=32 for IFEval), and how groups are formed (App D).

### 07 — Dequantization

- **Status**: pending
- **Depends on**: 05
- **Exit criterion**: `kvarn/07-dequantization.md` describes the dequant
  formula (dual-scale fused), overhead (<1.4%), kernel fusion of s2, and
  comparison to KIVI single-scale (App I, Fig 11).

### 08 — Pseudo-Decode Evaluation

- **Status**: pending
- **Depends on**: 01
- **Exit criterion**: `kvarn/08-pseudo-decode-eval.md` describes the
  block-wise quantize-then-continue protocol, why it models decoding error
  accumulation, and how it differs from static prefill eval (Sec 3.2, Fig 4).

### 09 — Experimental Setup

- **Status**: pending
- **Depends on**: 05, 06
- **Exit criterion**: `kvarn/09-experimental-setup.md` lists models, decoding
  params, benchmarks, baselines, and effective memory accounting (App D).

### 10 — Results Summary

- **Status**: pending
- **Depends on**: 09
- **Exit criterion**: `kvarn/10-results-summary.md` summarizes AIME24,
  MATH500, HumanEval, IFEval, and line-retrieval results, KVarN vs baselines
  (Tables 1-4).

### 11 — Limitations

- **Status**: pending
- **Depends on**: 00
- **Exit criterion**: `kvarn/11-limitations.md` documents SSM inapplicability,
  MLA uncertainty, and the 2-bit serving framework gap (App E).