# Hadamard Transform Arrangement

**Paper sections**: 2.2 (Incoherence Processing), Appendix A, Fig 7

## Purpose

The Hadamard rotation "equalizes" the channel dimension of K and V, suppressing
outlier channels that would otherwise dominate quantization error. After rotation,
the distribution across channels approaches Gaussian (incoherence), making uniform
round-to-nearest quantization more effective.

This alone fixes **channel-space** outliers but is insufficient for **token-wise**
scaling errors (see Fig 1b) — hence the need for variance normalization on top.

## Layout (QuaRot-style)

[Fig 7, Appendix A] The Hadamard transforms are arranged in the attention layer
with two **absorbed** and two **online** transforms:

```
                ┌──────────────────────────────────────────────┐
                │              Attention Layer                  │
                │                                               │
  Q ──[H_online]──> Q'          K ──[H_online]──> K'           │
                                │                     │         │
                          (RoPE applied here)                  │
                                │                     │         │
                          ┌─────┴─────┐         ┌─────┴─────┐   │
                          │ QK^T attn  │         │  W_V * V  │   │
                          └─────┬─────┘         └─────┬─────┘   │
                                │                     │         │
                            [H_absorbed]          [H_absorbed] │
                                │                     │         │
                          (merged into W_V)   (merged into W_O) │
                                │                     │         │
                          V' ──[H_online]──> V''      │         │
                                │                     │         │
                                └────── concat ───────┘         │
                                           │                     │
                                      W_O * .                   │
                                           │                     │
                                    [H_absorbed]                │
                                           │                     │
                                       output                     │
                │                                               │
                └──────────────────────────────────────────────┘
```

- **Absorbed transforms**: merged into weight matrices `W_V` and `W_O`. These
  are precomputed and do not cost anything at inference time.
- **Online transforms**: applied to K (and Q, V) after RoPE embedding, per head.
  These are block-diagonal Hadamard transforms with block size = head dimension.

## Complexity

- Online transform: `O(N * log(N))` per head, where N = head_dim.
- For head_dim=128, this is ~128*7 = 896 multiply-adds per token per head.
- Negligible compared to the linear projections (W_Q, W_K, W_V) adjacent to it.

## Why not rotate in the token dimension?

[Sec 3.3] A token-dimension Hadamard would require undoing the rotation at
dequantization time for every token position — `128^2 = 16384` operations per 128
tokens per channel. This is prohibitively expensive for the online dequant path.

Instead, KVarN uses variance normalization (element-wise scaling) in the token
dimension, which costs only 1 extra FLOP per token per channel at dequant time.

## Implementation notes

- The Hadamard matrix `H_n` of size `n x n` (n = head_dim, must be power of 2)
  can be applied recursively without materializing the full matrix.
- In the reference vLLM implementation, the rotation is applied to K and V right
  after the W_K / W_V projection and after RoPE (for K).
- Q must also be rotated with the same online transform so that `Q * K^T` is
  rotationally consistent.
- The absorbed transforms into W_V and W_O are one-time weight preprocessing;
  the online transforms are applied during every forward pass.