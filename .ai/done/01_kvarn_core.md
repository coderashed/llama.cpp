# Spike 01: KVarN Core Algorithm Viability

- **Status**: done

## Postscript

**Verdict: MET** — KVarN core algorithm is viable.
VarN converges (Imb 312->12 in 8 iters), KVarN beats Hadamard+Q2 on 3/3
matrices, beats plain Q2 on heavy-tailed data (4.9x). Dequant formula
confirmed. FP16 scales add 0.125 bpe overhead (2.375 vs 2.25).
Evidence: `spikes/01_kvarn_core/EVIDENCE.md`
Self-study: `.ai/self-study/2026-06-25_01_kvarn_core.md`
- **Depends on**: nothing
- **Pipeline**: spike
- **Timebox**: one session (~2-3 hours)

## Question

Does the full KVarN pipeline (Hadamard rotation + Algorithm 1 variance
normalization + 2-bit RTN with dual-scale) actually reduce per-token
magnitude error (`E_M / E_T` for the top 5% worst tokens) compared to
plain Q4_0 RTN on realistic K-matrix data?

This is a standalone C++ prototype. No llama.cpp build integration.
Synthetic and quasi-real K matrices only. The goal is to validate the
paper's core claim before investing in build-system integration.

## Exit criterion (falsifiable)

`spikes/01_kvarn_core/EVIDENCE.md` contains:

1. **VarN convergence table**: For a 128x128 tile with known row/column
   variance imbalance, the `Imb()` metric after each of 8 iterations,
   showing it decreases monotonically (or at least reaches a value < 2.0
   from an initial value > 10.0).

2. **Magnitude error comparison table**: For at least 3 test K-matrices
   (Gaussian random, heavy-tailed, and a simulated attention K-matrix with
   outlier channels), the `E_M / E_T` ratio for the top 5% worst tokens,
   comparing:
   - FP16 (baseline, error = 0)
   - Q4_0 RTN (single scale, per-channel)
   - Hadamard + Q4_0 RTN (rotation only, no VarN)
   - Hadamard + VarN + 2-bit RTN dual-scale (full KVarN)

   KVarN must show lower `E_M / E_T` for the top 5% than both Q4_0 and
   Hadamard-only on at least 2 of 3 matrices.

3. **Bits-per-element accounting**: Confirmed effective bits/element for the
   KVarN storage format is <= 2.31 (2.25 with FP8 scales, ~2.31 with FP16
   scales as we will use initially).

If KVarN does NOT show lower magnitude error, the spike is FAILED and the
fallback decision is whether to abandon KVarN or investigate alternative
normalization approaches.

## What this spike does NOT answer

- Whether the new type compiles in the ggml build system (that's a separate spike)
- Whether VarN can be inserted into the KV-cache write path (separate spike)
- GPU kernel performance
- End-to-end model quality (requires full integration)