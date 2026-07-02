// RED tests for Phase B (KVARN_FAITHFUL/03): register GGML_TYPE_Q2_KVARN_K as a
// ggml type so the KV cache can allocate a per-channel K body tensor and backends
// can dequantize it. Spec: .ai/design/kvarn_faithful_build_design.md, Section 5.2.
//
// Test contract:
//   1. GGML_TYPE_Q2_KVARN_K enum value is registered, before GGML_TYPE_COUNT.
//   2. type_traits[Q2_KVARN_K]: blck_size=128, type_size=36, is_quantized, name set.
//   3. to_float is non-NULL and dequantizes a per-channel block correctly (row =
//      one channel's 128 tokens), matching dequantize_row_q2_kvarn_k.
//
// RED: fails to COMPILE (GGML_TYPE_Q2_KVARN_K undefined) until the enum + traits
// are added in GREEN.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <vector>

// Test 1: enum value registered at its pinned slot (43; the 3/4-bit
// KVARN_MULTIBIT siblings follow it).
static void test_enum_value(void) {
    assert(GGML_TYPE_Q2_KVARN_K == 43);
    const char * name = ggml_type_name(GGML_TYPE_Q2_KVARN_K);
    assert(name != NULL);
    assert(name[0] != '\0');
    printf("  Q2_KVARN_K type name: %s\n", name);
}

// Test 2: type traits registration.
static void test_type_traits(void) {
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN_K);
    assert(tt->blck_size    == 128);                       // QG2_KVARN tokens per block
    assert(tt->type_size    == sizeof(block_q2_kvarn_k));  // 36 bytes
    assert(tt->type_size    == 36);
    assert(tt->is_quantized == true);
    assert(tt->to_float     != NULL);                      // backends dequantize the body

    // ggml_row_size over one block (128 tokens of one channel) == 36 bytes.
    assert(ggml_row_size(GGML_TYPE_Q2_KVARN_K, 128) == 36);
}

// Test 3: to_float dequantizes per-channel blocks (channel-major) correctly.
// Build a 2-channel x 128-token tile, quantize with the Phase A ref, then
// dequantize via the registered type trait and compare to the direct dequant.
static void test_to_float_roundtrip(void) {
    const int n_ch  = 2;
    const int n_tok = 128;

    std::vector<float> tile(n_ch * n_tok);
    for (int ch = 0; ch < n_ch; ch++) {
        for (int tok = 0; tok < n_tok; tok++) {
            tile[ch * n_tok + tok] = 0.1f * (float)(tok - 64) * (float)(ch + 1);
        }
    }

    std::vector<block_q2_kvarn_k> blocks(n_ch);
    quantize_row_q2_kvarn_k_ref(tile.data(), blocks.data(), n_ch, n_tok);

    // Reference dequant via the Phase A function.
    std::vector<float> ref(n_ch * n_tok);
    dequantize_row_q2_kvarn_k(blocks.data(), ref.data(), n_ch, n_tok);

    // Dequant via the registered type trait (row = 128-token block, channel-major).
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN_K);
    std::vector<float> via_trait(n_ch * n_tok);
    tt->to_float(blocks.data(), via_trait.data(), (int64_t)n_ch * n_tok);

    for (int i = 0; i < n_ch * n_tok; i++) {
        assert(via_trait[i] == ref[i]);  // bit-identical: same underlying dequant
    }
}

// Test 4: from_float_ref quantizes a channel-major row (one channel's 128 tokens
// per block) identically to quantize_row_q2_kvarn_k_ref. This is the write-path
// adapter: cpy_k transposes each K group to channel-major, then this quantizes it.
static void test_from_float_ref(void) {
    const int n_ch  = 3;
    const int n_tok = 128;

    // Channel-major tile: row = channel, 128 contiguous tokens.
    std::vector<float> tile(n_ch * n_tok);
    for (int ch = 0; ch < n_ch; ch++) {
        float amp = (ch == 1) ? 8.0f : 1.0f;  // one outlier channel
        for (int tok = 0; tok < n_tok; tok++) {
            tile[ch * n_tok + tok] = amp * 0.05f * (float)((tok * 7 + ch) % 23 - 11);
        }
    }

    // Direct per-channel reference.
    std::vector<block_q2_kvarn_k> ref(n_ch);
    quantize_row_q2_kvarn_k_ref(tile.data(), ref.data(), n_ch, n_tok);

    // Via the registered from_float_ref over the whole channel-major buffer
    // (k = n_ch * n_tok contiguous floats -> n_ch independent blocks).
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN_K);
    std::vector<block_q2_kvarn_k> via_trait(n_ch);
    tt->from_float_ref(tile.data(), via_trait.data(), (int64_t)n_ch * n_tok);

    for (int ch = 0; ch < n_ch; ch++) {
        assert(ref[ch].s == via_trait[ch].s);
        assert(ref[ch].z == via_trait[ch].z);
        assert(memcmp(ref[ch].qs, via_trait[ch].qs, sizeof(ref[ch].qs)) == 0);
    }
}

int main(void) {
    ggml_cpu_init();
    printf("test-q2-kvarn-k-type:\n");
    int passed = 0;

    printf("  Test 1 (enum_value): ");
    fflush(stdout);
    test_enum_value();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (type_traits): ");
    fflush(stdout);
    test_type_traits();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (to_float_roundtrip): ");
    fflush(stdout);
    test_to_float_roundtrip();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (from_float_ref): ");
    fflush(stdout);
    test_from_float_ref();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed);
    return 0;
}
