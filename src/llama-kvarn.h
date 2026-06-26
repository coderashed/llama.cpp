#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define KVARN_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define KVARN_RESTRICT __restrict
#else
#define KVARN_RESTRICT
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
