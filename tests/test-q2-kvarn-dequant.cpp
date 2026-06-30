// Failing tests for Item 02 - Q2_KVARN Dequantization
// These tests verify the dequantize_row_q2_kvarn function:
//   1. Bit mask correctness (& 0x03 vs & 0x01)
//   2. Known block roundtrip
//   3. Dual-scale formula
//   4. Block size invariant

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

// Test 1: Bit mask correctness
// Manually construct a block_q2_kvarn with known 2-bit values
// qs bytes = 0xE4 = binary 11 10 01 00 -> values 3,2,1,0
// With & 0x01 bug, values 2 and 3 are decoded as 0 and 1
static void test_bit_mask(void) {
    // Construct block manually using struct layout:
    // offset 0-31: qs[32]
    // offset 32-33: d (ggml_fp16_t)
    // offset 34-35: s1 (ggml_fp16_t)
    // offset 36-37: s2 (ggml_fp16_t)
    uint8_t block[38];
    memset(block, 0xE4, 32);

    ggml_fp16_t d_val  = ggml_fp32_to_fp16(0.0f);
    ggml_fp16_t s1_val = ggml_fp32_to_fp16(1.0f);
    ggml_fp16_t s2_val = ggml_fp32_to_fp16(1.0f);
    memcpy(&block[32], &d_val,  2);
    memcpy(&block[34], &s1_val, 2);
    memcpy(&block[36], &s2_val, 2);

    float y[128];
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    tt->to_float(block, y, 128);

    // 0xE4 = 0b_11_10_01_00 -> q0=0, q1=1, q2=2, q3=3
    // y[i] = (qval + 0.0) * 1.0 * 1.0 = qval
    for (int j = 0; j < 32; j++) {
        assert(fabsf(y[j*4 + 0] - 0.0f) < 1e-3f);
        assert(fabsf(y[j*4 + 1] - 1.0f) < 1e-3f);
        assert(fabsf(y[j*4 + 2] - 2.0f) < 1e-3f);
        assert(fabsf(y[j*4 + 3] - 3.0f) < 1e-3f);
    }
}

// Test 2: Known block roundtrip
// Quantize a known float vector (values 0..3 repeated), dequantize,
// assert per-element error < 1e-3. Fails because & 0x01 corrupts output.
static void test_roundtrip(void) {
    const int64_t nelem = 128;
    std::vector<float> src(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        src[i] = (float)(i % 4);
    }

    size_t dst_size = ggml_row_size(GGML_TYPE_Q2_KVARN, nelem);
    std::vector<uint8_t> dst(dst_size);

    size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q2_KVARN, src.data(), dst.data(),
        0, 1, nelem, NULL);
    assert(written == dst_size);

    std::vector<float> out(nelem);
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    tt->to_float(dst.data(), out.data(), nelem);

    for (int64_t i = 0; i < nelem; i++) {
        float diff = fabsf(out[i] - src[i]);
        assert(diff < 1e-3f);
    }
}

// Test 3: Dual-scale formula
// Create a block with s1=2.0, s2=3.0, d=1.0, qs with known values.
// Expected: y = (qval + 1.0) * 6.0
static void test_dual_scale(void) {
    uint8_t block[38];
    memset(block, 0xE4, 32);

    ggml_fp16_t d_val  = ggml_fp32_to_fp16(1.0f);
    ggml_fp16_t s1_val = ggml_fp32_to_fp16(2.0f);
    ggml_fp16_t s2_val = ggml_fp32_to_fp16(3.0f);
    memcpy(&block[32], &d_val,  2);
    memcpy(&block[34], &s1_val, 2);
    memcpy(&block[36], &s2_val, 2);

    float y[128];
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    tt->to_float(block, y, 128);

    // 0xE4 -> q0=0, q1=1, q2=2, q3=3
    // y = (qval + 1.0) * 2.0 * 3.0 = (qval + 1.0) * 6.0
    float expected[4] = {
        (0.0f + 1.0f) * 6.0f,  // 6.0
        (1.0f + 1.0f) * 6.0f,  // 12.0
        (2.0f + 1.0f) * 6.0f,  // 18.0
        (3.0f + 1.0f) * 6.0f,  // 24.0
    };
    for (int j = 0; j < 32; j++) {
        assert(fabsf(y[j*4 + 0] - expected[0]) < 1e-3f);
        assert(fabsf(y[j*4 + 1] - expected[1]) < 1e-3f);
        assert(fabsf(y[j*4 + 2] - expected[2]) < 1e-3f);
        assert(fabsf(y[j*4 + 3] - expected[3]) < 1e-3f);
    }
}

// Test 4: Block size invariant
// Calling with k=64 (not a multiple of 128) must trip GGML_ASSERT and abort.
// GGML_ASSERT is always active (not compiled out by NDEBUG), so this is a death
// test: run the call in a forked child and verify the child aborted instead of
// exiting cleanly.
static void test_block_size(void) {
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    assert(pid >= 0 && "fork failed");
    if (pid == 0) {
        // child: this call must abort via GGML_ASSERT(k % 128 == 0)
        uint8_t block[38] = {0};
        float y[128];
        const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
        tt->to_float(block, y, 64);
        // reached only if the assert did NOT fire
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    const bool clean_exit = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    assert(!clean_exit && "to_float(k=64) should have aborted via GGML_ASSERT");
}

int main(void) {
    ggml_cpu_init();

    printf("test-q2-kvarn-dequant:\n");

    printf("  Test 1 (bit_mask): ");
    fflush(stdout);
    test_bit_mask();
    printf("PASSED\n");

    printf("  Test 2 (roundtrip): ");
    fflush(stdout);
    test_roundtrip();
    printf("PASSED\n");

    printf("  Test 3 (dual_scale): ");
    fflush(stdout);
    test_dual_scale();
    printf("PASSED\n");

    printf("  Test 4 (block_size): ");
    fflush(stdout);
    test_block_size();
    printf("PASSED\n");

    printf("\nAll tests passed\n");
    return 0;
}
