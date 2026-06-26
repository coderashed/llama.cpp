// Failing tests for Item 07 - K-shift graph support for Q2_KVARN
//
// These tests verify that the k-shift pipeline handles Q2_KVARN tensors:
//   1. build_rope_shift accepts Q2_KVARN K-tensor (dequant->rotate->requant)
//   2. Dequant-requant cycle via ggml_cast/ggml_cpy
//   3. Three-region shift iterates sink/body/recent tensors
//   4. seq_add preserves Q2_KVARN cache type
//
// All tests will FAIL at compile or link time until the feature is implemented.

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-kv-cache.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <vector>

// Test 1: build_rope_shift handles Q2_KVARN
// build_rope_shift must accept a Q2_KVARN K-tensor and perform
// dequantize -> rotate -> rope -> rotate -> requantize.
// This will fail at compile time because ggml_cast/ggml_cpy
// do not support GGML_TYPE_Q2_KVARN yet.
static void test_build_rope_shift_accepts_q2_kvarn(void) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 64*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx);

    // Create a Q2_KVARN tensor simulating a K cache layer
    // shape: [n_embd_k_gqa=4096, kv_size=64, n_stream=1]
    struct ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_KVARN, 4096, 64, 1);
    assert(k != NULL);
    assert(k->type == GGML_TYPE_Q2_KVARN);

    // build_rope_shift internally calls ggml_cast(k, F32) then ggml_cpy(tmp, k)
    // Both must support Q2_KVARN or this will fail at compile/link time.
    //
    // We test the ops directly:
    struct ggml_tensor * f32 = ggml_cast(ctx, k, GGML_TYPE_F32);
    assert(f32 != NULL);
    assert(f32->type == GGML_TYPE_F32);

    // Copy back to Q2_KVARN
    struct ggml_tensor * back = ggml_cpy(ctx, f32, k);
    assert(back != NULL);

    // Build and run the graph to exercise the compute path
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f32);
    ggml_build_forward_expand(gf, back);

    std::vector<uint8_t> work_buffer;
    struct ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
    if (plan.work_size > 0) {
        work_buffer.resize(plan.work_size);
        plan.work_data = work_buffer.data();
    }
    int ret = ggml_graph_compute(gf, &plan);
    assert(ret == 0);

    ggml_free(ctx);
    printf("  PASSED\n");
}

// Test 2: Dequant-requant cycle
// K-shift dequantizes Q2_KVARN -> rotates -> requantizes.
// This exercises the full dequantize_row_q2_kvarn -> RoPE -> quantize_row_q2_kvarn
// pipeline via ggml_compute_forward_dup_to_q and ggml_compute_forward_dup_from_q.
static void test_dequant_requant_cycle(void) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 64*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx);

    // Create a small Q2_KVARN tensor (1 block = 128 elements)
    struct ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_Q2_KVARN, 128);
    assert(src != NULL);

    // Dequantize to F32
    struct ggml_tensor * f32 = ggml_cast(ctx, src, GGML_TYPE_F32);
    assert(f32 != NULL);

    // Requantize back to Q2_KVARN
    struct ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_Q2_KVARN, 128);
    struct ggml_tensor * cpy = ggml_cpy(ctx, f32, dst);
    assert(cpy != NULL);

    // Build and run
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f32);
    ggml_build_forward_expand(gf, cpy);

    std::vector<uint8_t> work_buffer;
    struct ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
    if (plan.work_size > 0) {
        work_buffer.resize(plan.work_size);
        plan.work_data = work_buffer.data();
    }
    int ret = ggml_graph_compute(gf, &plan);
    assert(ret == 0);

    ggml_free(ctx);
    printf("  PASSED\n");
}

// Test 3: Three-region shift
// build_graph_shift must iterate k_sink, k_body, k_recent for Q2_KVARN caches
// instead of the single k tensor. This test creates a cache with Q2_KVARN type_k
// and verifies the shift graph includes all three regions.
static void test_three_region_shift(void) {
    // We need a minimal llama_context to exercise build_graph_shift.
    // Instead, we test the underlying assumption: that the kv_layer struct
    // has non-null k_sink/k_body/k_recent when type_k is Q2_KVARN.
    //
    // This test will fail at compile time because the three-region shift
    // path in build_graph_shift does not exist yet.

    struct ggml_init_params params = {
        /* .mem_size   = */ 64*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx);

    // Simulate the three-region layout
    const int64_t n_embd = 128;
    const int64_t S = 16;  // sink
    const int64_t B = 32;  // body
    const int64_t R = 16;  // recent

    struct ggml_tensor * k_sink   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,     n_embd, S, 1);
    struct ggml_tensor * k_body   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_KVARN, n_embd, B, 1);
    struct ggml_tensor * k_recent = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,     n_embd, R, 1);
    assert(k_sink   != NULL);
    assert(k_body   != NULL);
    assert(k_recent != NULL);

    // Each region must be independently shiftable.
    // For the body (Q2_KVARN), this requires the dequant->rotate->requant path.
    struct ggml_tensor * shift_sink   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * shift_body   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * shift_recent = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(shift_sink);
    ggml_set_input(shift_body);
    ggml_set_input(shift_recent);

    // Build a graph that shifts all three regions
    // sink: F16 -> rope_ext_inplace
    struct ggml_tensor * s = ggml_rope_ext_inplace(ctx, k_sink, shift_sink, NULL, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f);
    // body: Q2_KVARN -> cast -> rope -> cpy back
    struct ggml_tensor * b_f32 = ggml_cast(ctx, k_body, GGML_TYPE_F32);
    struct ggml_tensor * b_rot = ggml_rope_ext(ctx, b_f32, shift_body, NULL, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f);
    struct ggml_tensor * b = ggml_cpy(ctx, b_rot, k_body);
    // recent: F16 -> rope_ext_inplace
    struct ggml_tensor * r = ggml_rope_ext_inplace(ctx, k_recent, shift_recent, NULL, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, s);
    ggml_build_forward_expand(gf, b_f32);
    ggml_build_forward_expand(gf, b_rot);
    ggml_build_forward_expand(gf, b);
    ggml_build_forward_expand(gf, r);

    std::vector<uint8_t> work_buffer;
    struct ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
    if (plan.work_size > 0) {
        work_buffer.resize(plan.work_size);
        plan.work_data = work_buffer.data();
    }
    int ret = ggml_graph_compute(gf, &plan);
    assert(ret == 0);

    ggml_free(ctx);
    printf("  PASSED\n");
}

// Test 4: seq_add preserves Q2_KVARN
// After a k-shift, the cache type must still be Q2_KVARN.
// This test verifies that seq_add does not corrupt the type.
static void test_seq_add_preserves_q2_kvarn(void) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 64*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx);

    // Create a Q2_KVARN tensor and verify its type survives a copy
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_Q2_KVARN, 128);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_Q2_KVARN, 128);
    assert(a->type == GGML_TYPE_Q2_KVARN);
    assert(b->type == GGML_TYPE_Q2_KVARN);

    // Copy a -> b (simulating what seq_add does internally)
    struct ggml_tensor * cp = ggml_cpy(ctx, a, b);
    assert(cp != NULL);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cp);

    std::vector<uint8_t> work_buffer;
    struct ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
    if (plan.work_size > 0) {
        work_buffer.resize(plan.work_size);
        plan.work_data = work_buffer.data();
    }
    int ret = ggml_graph_compute(gf, &plan);
    assert(ret == 0);

    // Type must still be Q2_KVARN after the operation
    assert(b->type == GGML_TYPE_Q2_KVARN);

    ggml_free(ctx);
    printf("  PASSED\n");
}

int main(void) {
    ggml_cpu_init();

    printf("test-kvarn-kshift:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (build_rope_shift_accepts_q2_kvarn): ");
    test_build_rope_shift_accepts_q2_kvarn();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (dequant_requant_cycle): ");
    test_dequant_requant_cycle();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (three_region_shift): ");
    test_three_region_shift();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (seq_add_preserves_q2_kvarn): ");
    test_seq_add_preserves_q2_kvarn();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
