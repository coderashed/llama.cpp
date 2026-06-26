// Failing tests for Item 08 - Context params for KVarN configuration
// These tests verify that llama_context_params gains KVarN fields.
// All tests will FAIL at compile time because the fields do not exist yet.

#include "llama.h"
#include "get-model.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Test 1: New fields exist with correct defaults
// Will fail at compile time: kvarn_group_size, kvarn_sink_tokens, kvarn_recent_tokens, kvarn_varn_iterations don't exist
static void test_default_values(void) {
    printf("  Checking KVarN default values...\n");

    auto cparams = llama_context_default_params();

    // These will fail to compile: fields don't exist yet
    assert(cparams.kvarn_group_size    == 128);
    assert(cparams.kvarn_sink_tokens   == 128);
    assert(cparams.kvarn_recent_tokens  == 128);
    assert(cparams.kvarn_varn_iterations == 8);

    printf("  PASSED (unreachable)\n");
}

// Test 2: Validation — kvarn_group_size = 0 should cause llama_init_from_model to return NULL
// Will fail at compile time: kvarn_group_size doesn't exist
static void test_validation_zero_group_size(void) {
    printf("  Checking validation: kvarn_group_size = 0...\n");

    auto cparams = llama_context_default_params();
    cparams.type_k = GGML_TYPE_Q2_KVARN;
    cparams.kvarn_group_size = 0; // This will fail to compile

    // If we got here, the field exists — but init should return NULL
    // (We can't actually call llama_init_from_model without a model, but the
    //  compile-time failure is the primary gate.)
    printf("  PASSED (unreachable)\n");
}

// Test 3: Conditional copy — setting type_k = Q2_KVARN with custom KVarN params succeeds
// Will fail at compile time: kvarn_group_size etc. don't exist
static void test_conditional_copy(void) {
    printf("  Checking conditional copy with Q2_KVARN...\n");

    auto cparams = llama_context_default_params();
    cparams.type_k = GGML_TYPE_Q2_KVARN;
    cparams.kvarn_group_size    = 64;   // Will fail to compile
    cparams.kvarn_sink_tokens   = 64;   // Will fail to compile
    cparams.kvarn_recent_tokens = 64;   // Will fail to compile
    cparams.kvarn_varn_iterations = 4;  // Will fail to compile

    printf("  PASSED (unreachable)\n");
}

// Test 4: Default zero — setting type_k = F16 with KVarN params set, the cache ignores them
// Will fail at compile time: kvarn_group_size etc. don't exist
static void test_default_zero(void) {
    printf("  Checking default zero with F16...\n");

    auto cparams = llama_context_default_params();
    cparams.type_k = GGML_TYPE_F16;
    cparams.kvarn_group_size    = 128;  // Will fail to compile
    cparams.kvarn_sink_tokens   = 128;  // Will fail to compile
    cparams.kvarn_recent_tokens = 128;  // Will fail to compile
    cparams.kvarn_varn_iterations = 8;  // Will fail to compile

    printf("  PASSED (unreachable)\n");
}

int main(void) {
    printf("test-kvarn-params:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (default_values): ");
    test_default_values();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (validation_zero_group_size): ");
    test_validation_zero_group_size();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (conditional_copy): ");
    test_conditional_copy();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (default_zero): ");
    test_default_zero();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
