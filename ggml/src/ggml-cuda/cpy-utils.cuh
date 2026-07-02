#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

static __device__ void quantize_f32_q2_kvarn_block(const float * __restrict__ x, block_q2_kvarn * __restrict__ y) {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    for (int j = 0; j < QK2_KVARN; j++) {
        const float v = x[j];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    const float range = max_val - min_val;
    float lo, s1, inv;

    if (range <= 0.0f) {
        // degenerate block (all elements equal): skip the search and use the
        // s1=1 / code-0 convention. Dequant reconstructs the constant and s2
        // restores its magnitude.
        lo  = min_val;
        s1  = 1.0f;
        inv = 1.0f;
    } else {
        // MSE-optimal symmetric clip: try a small grid of clip fractions f of
        // the [min,max] interval and keep the one with the lowest pre-s2
        // reconstruction MSE. A large outlier wastes code levels at f=1.0, so a
        // tighter f clips it (it saturates to code 3) and spends resolution on
        // the bulk.
        //
        // Load-bearing invariant: F is iterated with 1.0 FIRST and ties use a
        // STRICT '<', so f=1.0 wins unless a tighter clip strictly lowers MSE.
        // When clipping does not help, lo==min and s1==range/3, reproducing the
        // plain min/max quantizer bit-for-bit. The CPU mirror keeps the same
        // order and strict compare so both pick the same f.
        const float F[5]   = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f};
        const float center = 0.5f * (min_val + max_val);
        float best_mse = FLT_MAX;
        float best_f   = 1.0f;

        for (int fi = 0; fi < 5; fi++) {
            const float f     = F[fi];
            const float half  = 0.5f * f * range;
            const float lo_f  = center - half;
            const float s1_f  = f * range / 3.0f;
            const float inv_f = 1.0f / s1_f;

            float mse = 0.0f;
            for (int j = 0; j < QK2_KVARN; j++) {
                float val = (x[j] - lo_f) * inv_f;
                val = fminf(fmaxf(val, 0.0f), 3.0f);
                const float q  = (float)(uint8_t)(val + 0.5f);
                const float r  = q * s1_f + lo_f;
                const float d_ = x[j] - r;
                mse += d_ * d_;
            }
            if (mse < best_mse) {
                best_mse = mse;
                best_f   = f;
            }
        }

        const float half = 0.5f * best_f * range;
        lo  = center - half;
        s1  = best_f * range / 3.0f;
        inv = 1.0f / s1;
    }

    // store the zeropoint in quantized units so that dequant (qval + d) * s1
    // reconstructs qval * s1 + lo exactly
    const float zp = lo * inv;

    y->d  = __float2half(zp);
    y->s1 = __float2half(s1);

    for (int j = 0; j < QK2_KVARN / 4; ++j) {
        uint8_t byte = 0;
        for (int b = 0; b < 4; ++b) {
            float val = (x[j*4 + b] - lo) * inv;
            val = fminf(fmaxf(val, 0.0f), 3.0f);
            byte |= ((uint8_t)(val + 0.5f)) << (b * 2);
        }
        y->qs[j] = byte;
    }

    // per-block s2 norm correction (matches the CPU reference path)
    float sum_sq_orig = 0.0f;
    float sum_sq_dq   = 0.0f;
    for (int j = 0; j < QK2_KVARN / 4; ++j) {
        const uint8_t byte = y->qs[j];
        for (int b = 0; b < 4; ++b) {
            const float qval  = (float)((byte >> (b * 2)) & 0x03);
            const float dq    = (qval + zp) * s1;
            sum_sq_dq   += dq * dq;
            sum_sq_orig += x[j*4 + b] * x[j*4 + b];
        }
    }
    const float norm_dq = sqrtf(sum_sq_dq);
    const float s2 = norm_dq > 1e-10f ? sqrtf(sum_sq_orig) / norm_dq : 1.0f;
    y->s2 = __float2half(s2);
}

static __device__ void cpy_blck_f32_q2_kvarn(const char * cxi, char * cdsti) {
    quantize_f32_q2_kvarn_block((const float *)cxi, (block_q2_kvarn *)cdsti);
}

// Per-channel K quantizer device function (Phase A, KVARN_FAITHFUL/03).
// Quantizes one channel of n_tok tokens into one block_q2_kvarn_k.
// Mirrors quantize_row_q2_kvarn_k_ref exactly: same 02b MSE-clip grid,
// same iteration order, same strict-< tie-break.  No s2 term.
static __device__ void quantize_f32_q2_kvarn_k_block(
        const float * __restrict__ x, int n_tok,
        block_q2_kvarn_k * __restrict__ y) {
    const float F[5] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f};

    float min_val =  FLT_MAX;
    float max_val = -FLT_MAX;
    for (int j = 0; j < n_tok; j++) {
        const float v = x[j];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    const float range = max_val - min_val;
    float lo, s, inv;

    if (range <= 0.0f) {
        lo  = min_val;
        s   = 1.0f;
        inv = 1.0f;
    } else {
        const float center = 0.5f * (min_val + max_val);
        float best_mse = FLT_MAX;
        float best_f   = 1.0f;

        for (int fi = 0; fi < 5; fi++) {
            const float f     = F[fi];
            const float half  = 0.5f * f * range;
            const float lo_f  = center - half;
            const float s_f   = f * range / 3.0f;
            const float inv_f = 1.0f / s_f;

            float mse = 0.0f;
            for (int j = 0; j < n_tok; j++) {
                float val = fminf(fmaxf((x[j] - lo_f) * inv_f, 0.0f), 3.0f);
                const float q  = (float)(uint8_t)(val + 0.5f);
                const float r  = q * s_f + lo_f;
                const float d_ = x[j] - r;
                mse += d_ * d_;
            }
            if (mse < best_mse) {
                best_mse = mse;
                best_f   = f;
            }
        }

        const float half = 0.5f * best_f * range;
        lo  = center - half;
        s   = best_f * range / 3.0f;
        inv = 1.0f / s;
    }

    const float z = lo * inv;
    y->s = __float2half(s);
    y->z = __float2half(z);

    for (int j = 0; j < n_tok / 4; j++) {
        uint8_t byte = 0;
        for (int b = 0; b < 4; b++) {
            float val = fminf(fmaxf((x[j*4 + b] - lo) * inv, 0.0f), 3.0f);
            byte |= ((uint8_t)(val + 0.5f)) << (b * 2);
        }
        y->qs[j] = byte;
    }
}

// cpy block adapter: the source row is one channel's QG2_KVARN contiguous tokens
// (the write path transposes each K group to channel-major first), so one block
// = one channel. Mirrors cpy_blck_f32_q2_kvarn (per-token).
static __device__ void cpy_blck_f32_q2_kvarn_k(const char * cxi, char * cdsti) {
    quantize_f32_q2_kvarn_k_block((const float *)cxi, QG2_KVARN, (block_q2_kvarn_k *)cdsti);
}

// Shared scalar-scale search for the KVarN 3/4-bit siblings. Mirrors
// kvarn_scalar_scale in ggml-quants.c: same iteration order, same strict-<
// tie-break, so CPU and CUDA pick the same clip fraction bit-for-bit.
static __device__ void kvarn_scalar_scale_cuda(const float * __restrict__ x, int n, float qmax,
        float * __restrict__ lo_out, float * __restrict__ s_out) {
    float min_val =  FLT_MAX;
    float max_val = -FLT_MAX;
    for (int j = 0; j < n; j++) {
        const float v = x[j];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    const float range = max_val - min_val;
    if (range <= 0.0f) {
        *lo_out = min_val;
        *s_out  = 1.0f;
        return;
    }

    const float F[5]   = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f};
    const float center = 0.5f * (min_val + max_val);
    float best_mse = FLT_MAX;
    float best_f   = 1.0f;

    for (int fi = 0; fi < 5; fi++) {
        const float f     = F[fi];
        const float half  = 0.5f * f * range;
        const float lo_f  = center - half;
        const float s_f   = f * range / qmax;
        const float inv_f = 1.0f / s_f;

        float mse = 0.0f;
        for (int j = 0; j < n; j++) {
            float val = (x[j] - lo_f) * inv_f;
            if (val < 0.0f) val = 0.0f;
            if (val > qmax) val = qmax;
            const float q  = (float)(uint8_t)(val + 0.5f);
            const float r  = q * s_f + lo_f;
            const float d_ = x[j] - r;
            mse += d_ * d_;
        }
        if (mse < best_mse) {
            best_mse = mse;
            best_f   = f;
        }
    }

    const float half = 0.5f * best_f * range;
    *lo_out = center - half;
    *s_out  = best_f * range / qmax;
}

static __device__ void quantize_f32_q3_kvarn_block(const float * __restrict__ x, block_q3_kvarn * __restrict__ y) {
    float lo, s1;
    kvarn_scalar_scale_cuda(x, QK3_KVARN, 7.0f, &lo, &s1);
    const float inv = 1.0f / s1;
    const float zp  = lo * inv;

    for (int j = 0; j < QK3_KVARN/4; j++) y->ql[j] = 0;
    for (int j = 0; j < QK3_KVARN/8; j++) y->qh[j] = 0;

    float sum_sq_orig = 0.0f;
    float sum_sq_dq   = 0.0f;
    for (int j = 0; j < QK3_KVARN; j++) {
        float val = (x[j] - lo) * inv;
        val = fminf(fmaxf(val, 0.0f), 7.0f);
        const uint8_t q = (uint8_t)(val + 0.5f);
        y->ql[j >> 2] |= (uint8_t)((q & 3) << ((j & 3) * 2));
        y->qh[j >> 3] |= (uint8_t)((q >> 2) << (j & 7));
        const float dq = ((float)q + zp) * s1;
        sum_sq_dq   += dq * dq;
        sum_sq_orig += x[j] * x[j];
    }
    const float norm_dq = sqrtf(sum_sq_dq);
    y->d  = __float2half(zp);
    y->s1 = __float2half(s1);
    y->s2 = __float2half(norm_dq > 1e-10f ? sqrtf(sum_sq_orig) / norm_dq : 1.0f);
}

static __device__ void quantize_f32_q4_kvarn_block(const float * __restrict__ x, block_q4_kvarn * __restrict__ y) {
    float lo, s1;
    kvarn_scalar_scale_cuda(x, QK4_KVARN, 15.0f, &lo, &s1);
    const float inv = 1.0f / s1;
    const float zp  = lo * inv;

    for (int j = 0; j < QK4_KVARN/2; j++) y->qs[j] = 0;

    float sum_sq_orig = 0.0f;
    float sum_sq_dq   = 0.0f;
    for (int j = 0; j < QK4_KVARN; j++) {
        float val = (x[j] - lo) * inv;
        val = fminf(fmaxf(val, 0.0f), 15.0f);
        const uint8_t q = (uint8_t)(val + 0.5f);
        y->qs[j >> 1] |= (uint8_t)(q << ((j & 1) * 4));
        const float dq = ((float)q + zp) * s1;
        sum_sq_dq   += dq * dq;
        sum_sq_orig += x[j] * x[j];
    }
    const float norm_dq = sqrtf(sum_sq_dq);
    y->d  = __float2half(zp);
    y->s1 = __float2half(s1);
    y->s2 = __float2half(norm_dq > 1e-10f ? sqrtf(sum_sq_orig) / norm_dq : 1.0f);
}

static __device__ void quantize_f32_q3_kvarn_k_block(const float * __restrict__ x, int n_tok, block_q3_kvarn_k * __restrict__ y) {
    float lo, s;
    kvarn_scalar_scale_cuda(x, n_tok, 7.0f, &lo, &s);
    const float inv = 1.0f / s;

    y->s = __float2half(s);
    y->z = __float2half(lo * inv);

    for (int j = 0; j < QG3_KVARN/4; j++) y->ql[j] = 0;
    for (int j = 0; j < QG3_KVARN/8; j++) y->qh[j] = 0;
    for (int j = 0; j < n_tok; j++) {
        float val = fminf(fmaxf((x[j] - lo) * inv, 0.0f), 7.0f);
        const uint8_t q = (uint8_t)(val + 0.5f);
        y->ql[j >> 2] |= (uint8_t)((q & 3) << ((j & 3) * 2));
        y->qh[j >> 3] |= (uint8_t)((q >> 2) << (j & 7));
    }
}

static __device__ void quantize_f32_q4_kvarn_k_block(const float * __restrict__ x, int n_tok, block_q4_kvarn_k * __restrict__ y) {
    float lo, s;
    kvarn_scalar_scale_cuda(x, n_tok, 15.0f, &lo, &s);
    const float inv = 1.0f / s;

    y->s = __float2half(s);
    y->z = __float2half(lo * inv);

    for (int j = 0; j < QG4_KVARN/2; j++) y->qs[j] = 0;
    for (int j = 0; j < n_tok; j++) {
        float val = fminf(fmaxf((x[j] - lo) * inv, 0.0f), 15.0f);
        const uint8_t q = (uint8_t)(val + 0.5f);
        y->qs[j >> 1] |= (uint8_t)(q << ((j & 1) * 4));
    }
}

static __device__ void cpy_blck_f32_q3_kvarn(const char * cxi, char * cdsti) {
    quantize_f32_q3_kvarn_block((const float *)cxi, (block_q3_kvarn *)cdsti);
}

static __device__ void cpy_blck_f32_q4_kvarn(const char * cxi, char * cdsti) {
    quantize_f32_q4_kvarn_block((const float *)cxi, (block_q4_kvarn *)cdsti);
}

static __device__ void cpy_blck_f32_q3_kvarn_k(const char * cxi, char * cdsti) {
    quantize_f32_q3_kvarn_k_block((const float *)cxi, QG3_KVARN, (block_q3_kvarn_k *)cdsti);
}

static __device__ void cpy_blck_f32_q4_kvarn_k(const char * cxi, char * cdsti) {
    quantize_f32_q4_kvarn_k_block((const float *)cxi, QG4_KVARN, (block_q4_kvarn_k *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
