#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define KVARN_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define KVARN_RESTRICT __restrict
#else
#define KVARN_RESTRICT
#endif

struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// ggml custom-op callback (KVARN_FAITHFUL/04): runs VarN on the input group tile
// and writes a packed output. Input dst->src[0] is F32 [n_tok, n_ch] (contiguous,
// data = T[ch*n_tok + t], i.e. VarN rows=channels, cols=tokens). Output dst is a
// flat F32 tensor of size n_ch*n_tok + n_ch + n_tok, laid out as:
//   [0 .. n_ch*n_tok)              normalized tile T_norm (same layout as input)
//   [n_ch*n_tok .. +n_ch)          S_r (per-channel row scale)
//   [n_ch*n_tok+n_ch .. +n_tok)    S_c (per-token column scale)
// Single-task (ith==0 does all work): VarN's best-Imb snapshot is not tile-parallel.
void kvarn_varn_op(struct ggml_tensor * dst, int ith, int nth, void * userdata);

void kvarn_variance_normalize(
    float* KVARN_RESTRICT T,  // [R][C] tile, modified in-place
    int R,                    // rows (e.g. 128)
    int C,                    // columns (e.g. 128)
    int K,                    // iteration count (default 12)
    float c_min,              // clamp lower bound (e.g. -5.0)
    float c_max,              // clamp upper bound (e.g. 5.0)
    float* KVARN_RESTRICT S_c,// [C] output column scale vector
    float* KVARN_RESTRICT S_r // [R] output row scale vector
);

#ifdef __cplusplus
}
#endif
