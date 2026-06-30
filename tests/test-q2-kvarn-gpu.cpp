// Tests for Q2_KVARN GPU backend (item 09) and K vec_dot formula (DP4A item 01).
// Tests 1-3: CUDA/HIP backend support for Q2_KVARN operations.
// Test 4:    CPU float formula vs fp64 oracle for the K attention dot product.
//            Pins the contract that vec_dot_fattn_vec_KQ_q2_kvarn must satisfy;
//            the DP4A kernel (item 02) must pass the same oracle.

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

// Test 1: supports_op returns true for Q2_KVARN MUL_MAT on CUDA
// Currently returns false because Q2_KVARN is not in the CUDA supports_op switch
static void test_supports_op(void) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("  SKIP (no CUDA device)\n");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 4096, 4096);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
    struct ggml_tensor * op = ggml_mul_mat(ctx, a, b);

    bool supported = ggml_backend_supports_op(cuda_backend, op);
    // FAILS: Q2_KVARN not in CUDA supports_op switch -> returns false
    assert(supported && "CUDA backend should support Q2_KVARN MUL_MAT");

    ggml_free(ctx);
}

// Test 2: CUDA dequant kernel exists for Q2_KVARN
// Tests that CPY from Q2_KVARN to F32 is supported on CUDA backend
// Currently fails because no dequant kernel exists for Q2_KVARN in convert.cu
static void test_dequant_kernel(void) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("  SKIP (no CUDA device)\n");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 128, 1);
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
    struct ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

    bool supported = ggml_backend_supports_op(cuda_backend, cpy);
    // FAILS: no Q2_KVARN case in ggml_get_to_fp32_cuda / ggml_get_to_fp16_cuda
    assert(supported && "CUDA backend should support Q2_KVARN dequant (CPY to F32)");

    ggml_free(ctx);
}

// Test 3: Flash attention handles Q2_KVARN K/V
// Tests that FLASH_ATTN_EXT with Q2_KVARN K/V is supported on CUDA
// Currently fails because flash attention doesn't handle Q2_KVARN
static void test_flash_attn(void) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("  SKIP (no CUDA device)\n");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Q: F32, K: Q2_KVARN, V: Q2_KVARN
    struct ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
    struct ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 128, 1);
    struct ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 128, 1);
    struct ggml_tensor * mask = NULL;
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    bool supported = ggml_backend_supports_op(cuda_backend, attn);
    // FAILS: flash attention doesn't handle Q2_KVARN K/V
    assert(supported && "CUDA flash attention should handle Q2_KVARN K/V");

    ggml_free(ctx);
}

// Test 5: DP4A formula vs fp64 oracle for the K vec_dot.
// Validates the integer-dot + zeropoint path:
//   scale * (Q_ds.x * sumi + d * Q_ds.y / QI8_1)
// where sumi = sum_b(k2bit[b] * q_int8[b]) (integer dp4a emulation).
// Tests the /QI8_1 zeropoint correction that the DP4A kernel must get right.
static void test_vec_dot_k_dp4a_formula(void) {
    srand(54321);

    // QK8_1=32 elements per Q8_1 block; QI8_1=8 groups of 4 per Q8_1 block.
    const int QI8_1_val = 8;
    const int N = 128;  // one KVarN block
    const int nq8blocks = N / 32;  // 4 Q8_1 blocks per KVarN block
    const int ngroups   = N / 4;   // 32 groups of 4

    for (int run = 0; run < 20; run++) {
        // Random KVarN block.
        uint8_t qs[32];
        for (int i = 0; i < 32; i++) qs[i] = (uint8_t)(rand() & 0xFF);
        float d_in  = ((rand() % 201) - 100) * 0.01f;
        float s1_in = 0.5f + (rand() % 151) * 0.01f;
        float s2_in = 0.5f + (rand() % 151) * 0.01f;
        float d  = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d_in));
        float s1 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s1_in));
        float s2 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s2_in));
        float scale = s1 * s2;

        // Q8_1 blocks: int8 values, scale (ds.x), and sum-of-floats (ds.y).
        // ds.y is set to sum_i(q_int8[i] * ds.x) -- no extra quantization error.
        int8_t q_int8[128];
        float q_dx[4];  // ds.x per Q8_1 block
        float q_dy[4];  // ds.y per Q8_1 block (sum of original floats)
        for (int blk = 0; blk < nq8blocks; blk++) {
            q_dx[blk] = 0.01f + (rand() % 100) * 0.001f;
            float sum_f = 0.0f;
            for (int i = 0; i < 32; i++) {
                int8_t qi = (int8_t)((rand() % 255) - 127);
                q_int8[blk * 32 + i] = qi;
                sum_f += (float)qi * q_dx[blk];
            }
            q_dy[blk] = sum_f;
        }

        // fp64 oracle: sum_i (q_int8[i] * q_dx) * (k2bit[i] + d) * scale
        double ref64 = 0.0;
        for (int i = 0; i < N; i++) {
            int shift = (i % 4) * 2;
            double k2bit = (double)((qs[i / 4] >> shift) & 0x03);
            double q_f = (double)q_int8[i] * (double)q_dx[i / 32];
            ref64 += q_f * (k2bit + (double)d) * (double)s1 * (double)s2;
        }

        // DP4A emulation: scale * (Q_ds.x * sumi + d * Q_ds.y / QI8_1) per group.
        float result_dp4a = 0.0f;
        for (int grp = 0; grp < ngroups; grp++) {
            uint8_t qbyte = qs[grp];
            int sumi = 0;
            for (int b = 0; b < 4; b++) {
                sumi += (int)((qbyte >> (b * 2)) & 0x03) * (int)q_int8[grp * 4 + b];
            }
            int q8blk = grp / QI8_1_val;
            result_dp4a += scale * (q_dx[q8blk] * sumi + d * q_dy[q8blk] / QI8_1_val);
        }

        float ref_f = (float)ref64;
        float denom = fabsf(ref_f) > 1e-6f ? fabsf(ref_f) : 1e-6f;
        float relerr = fabsf(result_dp4a - ref_f) / denom;
        assert(relerr < 1e-4f);
    }
}

// Test 4: Float formula vs fp64 oracle for vec_dot_fattn_vec_KQ_q2_kvarn.
// K-dot formula: Sum_i Q[i] * (k2bit[i] + d) * s1 * s2
// Runs on CPU; no GPU required. Skipped at test-time if no DP4A device is
// present (the DP4A kernel, item 02, will invoke this same oracle on-device).
static void test_vec_dot_k_formula(void) {
    srand(12345);

    for (int run = 0; run < 20; run++) {
        // Random 2-bit K block: 128 elements packed in 32 bytes.
        uint8_t qs[32];
        for (int i = 0; i < 32; i++) qs[i] = (uint8_t)(rand() & 0xFF);

        // Scales stored as fp16 to match the GPU kernel's read precision.
        float d_in  = ((rand() % 201) - 100) * 0.01f;
        float s1_in = 0.5f + (rand() % 151) * 0.01f;
        float s2_in = 0.5f + (rand() % 151) * 0.01f;
        float d  = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d_in));
        float s1 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s1_in));
        float s2 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s2_in));
        float scale = s1 * s2;

        // Random Q values in [-1, 1].
        float Q[128];
        for (int i = 0; i < 128; i++) Q[i] = ((rand() % 201) - 100) * 0.01f;

        // fp64 oracle: reference value the float kernel must match.
        double ref64 = 0.0;
        for (int i = 0; i < 128; i++) {
            double k2bit = (double)((qs[i / 4] >> ((i % 4) * 2)) & 0x03);
            ref64 += (double)Q[i] * (k2bit + (double)d) * (double)s1 * (double)s2;
        }

        // Float emulation: same loop structure as vec_dot_fattn_vec_KQ_q2_kvarn.
        float result_f = 0.0f;
        for (int i = 0; i < 32; i++) {
            uint8_t qbyte = qs[i];
            for (int b = 0; b < 4; b++) {
                float kval = ((float)((qbyte >> (b * 2)) & 0x03) + d) * scale;
                result_f += kval * Q[i * 4 + b];
            }
        }

        float ref_f = (float)ref64;
        float denom = fabsf(ref_f) > 1e-6f ? fabsf(ref_f) : 1e-6f;
        float relerr = fabsf(result_f - ref_f) / denom;
        assert(relerr < 1e-4f);
    }
}

int main(void) {
    printf("test-q2-kvarn-gpu:\n");

    printf("  Test 1 (supports_op): ");
    fflush(stdout);
    test_supports_op();
    printf("PASSED\n");

    printf("  Test 2 (dequant_kernel): ");
    fflush(stdout);
    test_dequant_kernel();
    printf("PASSED\n");

    printf("  Test 3 (flash_attn): ");
    fflush(stdout);
    test_flash_attn();
    printf("PASSED\n");

    printf("  Test 4 (vec_dot_k_formula): ");
    fflush(stdout);
    test_vec_dot_k_formula();
    printf("PASSED\n");

    printf("  Test 5 (vec_dot_k_dp4a_formula): ");
    fflush(stdout);
    test_vec_dot_k_dp4a_formula();
    printf("PASSED\n");

    printf("\nAll tests passed\n");
    return 0;
}
