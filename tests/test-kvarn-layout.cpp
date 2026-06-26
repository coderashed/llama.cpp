// Failing tests for Item 06 - Three-region cache layout (sink/body/recent)
// These tests verify the kvarn_region enum, cell region metadata, three-tensor
// allocation for Q2_KVARN, and group quantization trigger at G=128 tokens.
// Tests 1-3 will FAIL at compile time; Test 4 will FAIL at link time.

#include "ggml.h"

#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <vector>

// Test 1: kvarn_region enum exists with SINK/BODY/RECENT values
// Will fail at compile time: kvarn_region is not defined
static void test_region_enum(void) {
    printf("  Checking kvarn_region enum...\n");

    // These will fail to compile: kvarn_region does not exist
    kvarn_region r = KVARN_REGION_SINK;
    (void)r;

    kvarn_region r2 = KVARN_REGION_BODY;
    (void)r2;

    kvarn_region r3 = KVARN_REGION_RECENT;
    (void)r3;

    // Verify enum values are distinct
    assert(KVARN_REGION_SINK   != KVARN_REGION_BODY);
    assert(KVARN_REGION_BODY   != KVARN_REGION_RECENT);
    assert(KVARN_REGION_RECENT != KVARN_REGION_SINK);

    printf("  PASSED (unreachable)\n");
}

// Test 2: llama_kv_cell has a region field
// Will fail at compile time: no region field on llama_kv_cell_ext
static void test_cell_region_metadata(void) {
    printf("  Checking cell region metadata...\n");

    // This will fail to compile: no region field
    llama_kv_cell_ext cell;
    cell.region = KVARN_REGION_RECENT;

    // Verify default is SINK
    llama_kv_cell_ext cell2;
    assert(cell2.region == KVARN_REGION_SINK);

    printf("  PASSED (unreachable)\n");
}

// Test 3: KV cache allocates sink/body/recent tensors when type is Q2_KVARN
// Will fail at compile time: kv_layer has no k_sink/k_body/k_recent fields
static void test_three_tensor_allocation(void) {
    printf("  Checking three-tensor allocation...\n");

    // We can't instantiate llama_kv_cache without a model, but we can check
    // that the kv_layer struct has the expected fields for Q2_KVARN layout.
    // This will fail to compile: no k_sink/k_body/k_recent in kv_layer
    llama_kv_cache::kv_layer layer;
    (void)layer.k_sink;
    (void)layer.k_body;
    (void)layer.k_recent;
    (void)layer.v_sink;
    (void)layer.v_body;
    (void)layer.v_recent;

    printf("  PASSED (unreachable)\n");
}

// Test 4: After G=128 tokens in recent region, they move to body
// Will fail at compile time: no kvarn_group_trigger function
static void test_group_quantization_trigger(void) {
    printf("  Checking group quantization trigger...\n");

    // This will fail to compile: kvarn_group_trigger does not exist
    uint32_t n_recent = 128;
    uint32_t group_size = 128;
    bool should_move = kvarn_group_trigger(n_recent, group_size);
    assert(should_move);

    // When recent count is below threshold, no move
    n_recent = 64;
    should_move = kvarn_group_trigger(n_recent, group_size);
    assert(!should_move);

    printf("  PASSED (unreachable)\n");
}

int main(void) {
    printf("test-kvarn-layout:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (region_enum): ");
    test_region_enum();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (cell_region_metadata): ");
    test_cell_region_metadata();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (three_tensor_allocation): ");
    test_three_tensor_allocation();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (group_quantization_trigger): ");
    test_group_quantization_trigger();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
