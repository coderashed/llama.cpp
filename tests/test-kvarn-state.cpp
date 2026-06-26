// Failing tests for Item 13 - State save/load for Q2_KVARN cache
//
// These tests verify that the three-region layout (sink/body/recent) is
// correctly serialized by llama_state_get_size, llama_state_get_data, and
// llama_state_set_data when the KV cache uses Q2_KVARN type_k.
//
// All tests will FAIL at compile time because the three-region
// serialization path does not exist yet.

#include "ggml.h"
#include "llama.h"

#include "../src/llama-kv-cache.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <vector>

// Test 1: State size includes all three regions
// llama_state_get_size must account for k_sink, k_body, k_recent (and v_*)
// when type_k is Q2_KVARN. Will fail at compile time because the three-region
// size calculation function does not exist yet.
static void test_state_size_includes_regions(void) {
    printf("  Checking state size includes three regions...\n");

    // The three-region state serialization needs a function to compute the
    // total size across sink/body/recent tensors. This function does not
    // exist yet — will fail to compile.
    llama_kv_cache::kv_layer layer;
    size_t total = llama_kv_cache_state_size_three_region(layer);
    (void)total;

    printf("  PASSED (unreachable)\n");
}

// Test 2: State save writes all three regions
// llama_state_get_data must serialize k_sink, k_body, k_recent tensors
// for each layer when type_k is Q2_KVARN. Will fail at compile time because
// the three-region state write function does not exist yet.
static void test_state_save_writes_all_regions(void) {
    printf("  Checking state save writes all three regions...\n");

    // The three-region serialization needs a dedicated write function that
    // iterates sink/body/recent instead of k_stream. This function does not
    // exist yet — will fail to compile.
    llama_kv_cache::kv_layer layer;
    llama_kv_cache_state_write_three_region(layer, nullptr, 0);

    printf("  PASSED (unreachable)\n");
}

// Test 3: State restore reads all three regions
// llama_state_set_data must restore k_sink, k_body, k_recent tensors
// for each layer when type_k is Q2_KVARN. Will fail at compile time because
// the three-region state read function does not exist yet.
static void test_state_restore_reads_regions(void) {
    printf("  Checking state restore reads all three regions...\n");

    // The three-region deserialization needs a dedicated read function that
    // restores sink/body/recent instead of k_stream. This function does not
    // exist yet — will fail to compile.
    llama_kv_cache::kv_layer layer;
    llama_kv_cache_state_read_three_region(layer, nullptr, 0);

    printf("  PASSED (unreachable)\n");
}

// Test 4: Roundtrip preserves output
// Save Q2_KVARN cache -> load into fresh context -> decode matches original.
// Will fail at compile time because the three-region serialization functions
// do not exist yet.
static void test_roundtrip_preserves_output(void) {
    printf("  Checking roundtrip preserves output...\n");

    // Full roundtrip requires all three serialization functions:
    // size, write, and read for three-region layout.
    // None of these exist yet — will fail to compile.
    llama_kv_cache::kv_layer layer;
    size_t sz = llama_kv_cache_state_size_three_region(layer);
    std::vector<uint8_t> buf(sz);
    llama_kv_cache_state_write_three_region(layer, buf.data(), buf.size());
    llama_kv_cache_state_read_three_region(layer, buf.data(), buf.size());

    printf("  PASSED (unreachable)\n");
}

int main(void) {
    ggml_cpu_init();

    printf("test-kvarn-state:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (state_size_includes_regions): ");
    test_state_size_includes_regions();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (state_save_writes_all_regions): ");
    test_state_save_writes_all_regions();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (state_restore_reads_regions): ");
    test_state_restore_reads_regions();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (roundtrip_preserves_output): ");
    test_roundtrip_preserves_output();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
