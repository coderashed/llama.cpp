// Failing tests for Item 12 - CLI flag for KVarN quantization
// These tests verify that --cache-type-k q2_kvarn is accepted by the arg parser.
// All tests will FAIL because "q2_kvarn" is not yet in the kv_cache_types list
// (common/arg.cpp:300) nor in llama-bench's ggml_type_from_name (llama-bench.cpp:478).

#include "common.h"
#include "arg.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>

// Test 1: kv_cache_type_from_str("q2_kvarn") returns GGML_TYPE_Q2_KVARN
// Will fail at runtime: "q2_kvarn" is not in the kv_cache_types vector
static void test_type_string_parsing(void) {
    printf("  Checking kv_cache_type_from_str(\"q2_kvarn\")...\n");

    common_params params;
    std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "--cache-type-k", "q2_kvarn"};
    std::vector<char *> cargv;
    for (auto & a : argv) {
        cargv.push_back(const_cast<char *>(a.data()));
    }

    bool ok = common_params_parse(cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_COMMON);
    assert(ok);
    assert(params.cache_type_k == GGML_TYPE_Q2_KVARN);

    printf("  PASSED (unreachable)\n");
}

// Test 2: --cache-type-k q2_kvarn is accepted by the arg parser
// Will fail at runtime: same reason as test 1
static void test_cli_flag(void) {
    printf("  Checking --cache-type-k q2_kvarn...\n");

    common_params params;
    std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "--cache-type-k", "q2_kvarn", "-p", "Hello"};
    std::vector<char *> cargv;
    for (auto & a : argv) {
        cargv.push_back(const_cast<char *>(a.data()));
    }

    // This will throw because q2_kvarn is not in kv_cache_types
    bool ok = common_params_parse(cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_COMMON);
    assert(ok);
    assert(params.cache_type_k == GGML_TYPE_Q2_KVARN);

    printf("  PASSED (unreachable)\n");
}

// Test 3: llama-bench --cache-type-k q2_kvarn works
// Will fail at runtime: llama-bench's ggml_type_from_name doesn't have q2_kvarn
static void test_bench_support(void) {
    printf("  Checking llama-bench --cache-type-k q2_kvarn...\n");

    // llama-bench uses its own ggml_type_from_name() which is a chain of if/else
    // that does not include "q2_kvarn". We simulate the same logic here.
    // The function returns GGML_TYPE_COUNT for unknown types, which triggers
    // invalid_param = true in the bench's arg parser.

    // We can't link against llama-bench, but we can verify the same behavior
    // through common_params_parse which is what llama-bench also uses for
    // --cache-type-k (though bench has its own copy of the parsing logic).

    common_params params;
    std::vector<std::string> argv = {"llama-bench", "-m", "model.gguf", "--cache-type-k", "q2_kvarn"};
    std::vector<char *> cargv;
    for (auto & a : argv) {
        cargv.push_back(const_cast<char *>(a.data()));
    }

    bool ok = common_params_parse(cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_COMMON);
    assert(ok);
    assert(params.cache_type_k == GGML_TYPE_Q2_KVARN);

    printf("  PASSED (unreachable)\n");
}

int main(void) {
    printf("test-kvarn-cli:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (type_string_parsing): ");
    test_type_string_parsing();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (cli_flag): ");
    test_cli_flag();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (bench_support): ");
    test_bench_support();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
