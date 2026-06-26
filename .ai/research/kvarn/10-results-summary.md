# Results Summary

**Paper section**: 4.1, Tables 1-4

## End-to-end reasoning (Table 1)

AIME24 / MATH500 — Accuracy / # Tokens (mean +/- std over 3 runs):

### Qwen3-4B

| Method | K/V | bits/elem | AIME24 Acc | AIME24 Tokens | MATH500 Acc | MATH500 Tokens |
|--------|-----|-----------|------------|--------------|-------------|----------------|
| FP16 | 16/16 | 16.0 | 61.1% | 12477 | 82.6% | 3857 |
| KIVI | 2/2 | 2.3 | 55.5% | 12794 | 77.8% | 3957 |
| QuaRot | 2/2 | 2.3 | 56.7% | 12732 | 78.9% | 3907 |
| KVQuant | 2/2 | 2.4 | 40.0% | 14794 | 67.5% | 5021 |
| PolarQuant | 4/2 | 3.3 | 52.2% | 13578 | 71.1% | 5259 |
| TurboQuant | 3/3 | 4.6 | 48.9% | 13642 | 77.0% | 3917 |
| Kitty | 2/2 | 2.4 | 53.3% | 13123 | 78.5% | 3834 |
| **KVarN** | **2/2** | **2.3** | **60.0%** | **12408** | **79.2%** | **3925** |

### Phi-4-14B (reasoning-plus)

| Method | AIME24 Acc | MATH500 Acc |
|--------|------------|-------------|
| FP16 | 62.2% | 84.9% |
| KIVI | 57.8% | 74.4% |
| QuaRot | 58.9% | 77.0% |
| **KVarN** | **61.7%** | **84.8%** |

**Key finding**: KVarN at 2.3 bits/elem nearly matches FP16 on Phi-4 (84.8% vs
84.9% on MATH500). KIVI loses 10.5 points on the same benchmark.

## HumanEval (Table 2)

| Model | Method | Accuracy |
|-------|--------|----------|
| Qwen3-4B | FP16 | 88.8% |
| Qwen3-4B | KIVI | 86.4% |
| Qwen3-4B | QuaRot | 86.3% |
| Qwen3-4B | **KVarN** | **88.4%** |
| Phi-4-14B | FP16 | 88.9% |
| Phi-4-14B | KIVI | 74.6% |
| Phi-4-14B | QuaRot | 87.0% |
| Phi-4-14B | **KVarN** | **88.2%** |

**Key finding**: KVarN recovers nearly all FP16 accuracy on code generation.
KIVI collapses on Phi-4 (74.6% vs 88.9%), while KVarN stays at 88.2%.

## IFEval (Table 3) — Instruction Following

Prompt-level Strict accuracy (%):

| Model | FP16 | KIVI | Hadamard | KVQuant | PolarQuant | TurboQuant | Kitty | **KVarN** |
|-------|------|------|----------|---------|------------|------------|-------|-----------|
| Qwen3-4B | 81.0 | 80.3 | 79.3 | 76.9 | 79.1 | 79.2 | 78.0 | **80.4** |
| Llama-3.1-8B | 71.1 | 70.9 | 70.8 | 57.7 | 69.5 | 66.5 | 70.2 | **71.0** |
| Phi-4-14B | 63.6 | 60.6 | 62.6 | 55.1 | 62.5 | 57.7 | 63.2 | **63.4** |

**Key finding**: KVarN matches or beats all 2-bit baselines and approaches FP16.
Notably it beats QuaRot (Hadamard-only) on every model, validating the VarN
contribution.

## Line-Retrieval (Table 4)

Accuracy (%) at various context lengths (number of lines):

### Qwen3-4B

| Method | 100 | 200 | 300 | 400 | 500 | 600 |
|--------|-----|-----|-----|-----|-----|-----|
| FP16 | 100 | 99 | 98 | 98 | 96 | 90 |
| KIVI | 98 | 97 | 88 | 84 | 84 | 74 |
| Hadamard | 98 | 98 | 91 | 98 | 92 | 83 |
| TurboQuant | 99 | 99 | 98 | 95 | 94 | 85 |
| **KVarN** | **100** | **99** | **98** | **97** | **97** | **85** |

### Phi-4-14B

| Method | 100 | 200 | 300 | 400 | 500 | 600 |
|--------|-----|-----|-----|-----|-----|-----|
| FP16 | 100 | 100 | 100 | 98 | 100 | 95 |
| KIVI | 95 | 96 | 88 | 88 | 91 | 82 |
| Hadamard | 99 | 99 | 97 | 97 | 91 | 94 |
| **KVarN** | **100** | **99** | **99** | **97** | **98** | **95** |

**Key finding**: KVarN matches FP16 on Phi-4 at 500 lines (98% vs 100%) and
600 lines (95% vs 95%). KIVI degrades severely at longer contexts. The gap
between KVarN and KIVI widens with context length, confirming the
anti-accumulation property.

## Ablation (Table 3 — Hadamard vs KVarN)

Comparing "Hadamard" (rotation only, no VarN) to KVarN:
- Qwen3-4B: 79.3% -> 80.4% (VarN adds 1.1 points)
- Llama-3.1-8B: 70.8% -> 71.0% (VarN adds 0.2 points)
- Phi-4-14B: 62.6% -> 63.4% (VarN adds 0.8 points)

VarN's IFEval contribution is modest (short outputs, less accumulation). The
biggest wins are on long-context benchmarks (AIME, MATH, line-retrieval) where
accumulation matters.

## Runtime overhead summary

| Operation | KIVI baseline | KVarN | Overhead |
|-----------|---------------|-------|----------|
| VarN normalization (quant time) | 0 | 1.9 ms / 1050 ms | 0.18% |
| Dequantization (per call, 4k ctx) | baseline | +1.4% | 1.4% |
| Dequantization (per call, 32k ctx) | baseline | ~0% | within noise |