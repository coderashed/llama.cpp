// Failing tests for Item 09 - GPU backend kernel for Q2_KVARN dequantization
// These tests verify CUDA backend support for Q2_KVARN:
//   1. supports_op returns true for MUL_MAT with Q2_KVARN weights
//   2. CUDA dequant kernel exists for Q2_KVARN (CPY Q2_KVARN->F32)
//   3. Flash attention handles Q2_KVARN K/V

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"

#undef NDEBUG
#include <assert.h>
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

    printf("\nAll tests passed (BUG: this should not happen)\n");
    return 0;
}
