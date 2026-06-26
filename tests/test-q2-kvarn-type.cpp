// Failing tests for Item 01 - Q2_KVARN Type Definition
// These tests verify the type system integration:
//   1. block_q2_kvarn struct layout (38 bytes)
//   2. GGML_TYPE_Q2_KVARN = 42 enum value
//   3. type_traits[Q2_KVARN] registration
//   4. quantize-dequantize roundtrip accuracy
//   5. ggml_quantize_chunk dispatch for Q2_KVARN

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <vector>

// Test 1: Struct size invariant
// Verify through the public API that the type has the correct size
static void test_struct_size(void) {
    // ggml_row_size verifies type_size * (nelements / blck_size) = 38 for 1 block
    size_t row_size = ggml_row_size(GGML_TYPE_Q2_KVARN, 128);
    assert(row_size == 38);

    // Verify via type_traits
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    assert(tt->type_size == 38);
}

// Test 2: Enum value registration
// GGML_TYPE_Q2_KVARN must be 42, directly before GGML_TYPE_COUNT = 43
static void test_enum_value(void) {
    // This will fail to compile until GGML_TYPE_Q2_KVARN is added to the ggml_type enum
    assert(GGML_TYPE_Q2_KVARN == 42);
    assert(GGML_TYPE_COUNT == 43);

    // ggml_type_name must return a non-null, non-empty string
    const char * name = ggml_type_name(GGML_TYPE_Q2_KVARN);
    assert(name != NULL);
    assert(name[0] != '\0');

    printf("  Q2_KVARN type name: %s\n", name);
}

// Test 3: Type traits registration
// The type_traits entry for Q2_KVARN must be fully populated
static void test_type_traits(void) {
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);

    // Core fields that must be set
    assert(tt->blck_size           == 128);  // 128 elements per block
    assert(tt->type_size           == 38);   // qs[32] + d(2) + s1(2) + s2(2)
    assert(tt->is_quantized        == true);

    // Function pointers must be non-NULL so backends can call them
    assert(tt->to_float            != NULL);  // dequantize_row_q2_kvarn
    assert(tt->from_float_ref      != NULL);  // quantize_row_q2_kvarn_ref

    // type_name must match the quantize chunk dispatch name
    assert(tt->type_name           != NULL);
    assert(strcmp(tt->type_name, "q2_kvarn") == 0);
}

// Test 4: Quantize-dequantize roundtrip
// Quantize 128 elements, dequantize, verify error is small
static void test_roundtrip(void) {
    const int64_t n_per_row = 128;  // one block
    const int64_t nrows = 1;
    const int64_t nelem = nrows * n_per_row;

    // Generate synthetic float data
    std::vector<float> src(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        src[i] = 0.1f + 2.0f * cosf((float)i);
    }

    // Allocate destination for quantized data + margin
    size_t dst_size = ggml_row_size(GGML_TYPE_Q2_KVARN, n_per_row) * nrows;
    std::vector<uint8_t> dst(dst_size + 64);

    // Quantize via dispatch
    size_t result = ggml_quantize_chunk(
        GGML_TYPE_Q2_KVARN,
        src.data(),
        dst.data(),
        0,        // start
        nrows,
        n_per_row,
        NULL);    // no importance matrix

    // Must have written the expected number of bytes
    assert(result == dst_size);

    // Dequantize back to float
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    assert(tt->to_float != NULL);

    std::vector<float> dst_f32(nelem);
    tt->to_float(dst.data(), dst_f32.data(), nelem);

    // Compute RMSE
    double sum_sq = 0.0;
    for (int64_t i = 0; i < nelem; i++) {
        double diff = (double)src[i] - (double)dst_f32[i];
        sum_sq += diff * diff;
    }
    float rmse = sqrtf((float)(sum_sq / (double)nelem));

    // For 2-bit quantization, RMSE should be bounded
    // 2-bit with 3 quantization levels: max error ~ range/6 = ~0.67 for range ~4
    // Allow generous threshold for initial reference implementation
    printf("  Q2_KVARN roundtrip RMSE: %f\n", rmse);
    assert(rmse < 1.0f);
}

// Test 5: Quantize chunk dispatch returns > 0
// Verify ggml_quantize_chunk dispatches correctly for Q2_KVARN
static void test_dispatch(void) {
    const int64_t n_per_row = 128;
    const int64_t nrows = 2;
    const int64_t nelem = nrows * n_per_row;

    std::vector<float> src(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        src[i] = (float)(i % 4);  // values 0..3
    }

    size_t expected = ggml_row_size(GGML_TYPE_Q2_KVARN, n_per_row) * nrows;
    std::vector<uint8_t> dst(expected + 64);

    size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q2_KVARN,
        src.data(),
        dst.data(),
        0,
        nrows,
        n_per_row,
        NULL);

    // Dispatch must return the correct byte count
    assert(written > 0);
    assert(written == expected);

    printf("  Q2_KVARN quantize_chunk: %zu bytes written (expected %zu)\n",
           written, expected);
}

int main(void) {
    // Initialize the CPU backend so quantize/dequantize tables are loaded
    ggml_cpu_init();

    printf("test-q2-kvarn-type:\n");
    int passed = 0;
    int failed = 0;

    // Each test is individually wrapped so we can report progress
    // (If a test fails to compile, the build itself fails — this is expected
    //  until the Q2_KVARN type is implemented.)

    printf("  Test 1 (struct_size): ");
    test_struct_size();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (enum_value): ");
    test_enum_value();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (type_traits): ");
    test_type_traits();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (roundtrip): ");
    test_roundtrip();
    printf("PASSED\n");
    passed++;

    printf("  Test 5 (dispatch): ");
    test_dispatch();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
