---
title: KVarN Pseudo-Decode Evaluation -- Safety Contract & Model Requirements
---

```mermaid
graph TD
    subgraph "Preconditions (caller must ensure)"
        P1["Model supports KV-cache quantization<br/>type_k / type_v set to quantized type"]
        P2["Prompt length N >= block_size b<br/>(at least one full block)"]
        P3["block_size b == 128<br/>(matches G = 128 tile size)"]
        P4["head_dim % 128 == 0<br/>(tile alignment for quantizer)"]
        P5["n_layers > 0<br/>(at least one transformer layer)"]
        P6["Baseline run completes before eval run<br/>(reference outputs must exist)"]
    end

    subgraph "Model Requirements"
        M1["Architecture: any decoder-only transformer<br/>with KV-cache (Llama, Qwen, Mistral, etc.)"]
        M2["Verification target: Qwen3-4B<br/>1000-token wikitext prompt"]
        M3["Minimum: stories15M for CI<br/>(functional test, not verification)"]
        M4["KV type: Q2_KVARN or Q4_0<br/>(set via llama_context_params)"]
        M5["n_embd_head_k must be divisible by 128<br/>(Hadamard + tile alignment)"]
    end

    subgraph "Block Processing Contract"
        B1["Block 0: FP16 cache, no quantization before compute<br/>Quantize AFTER block 0"]
        B2["Block k>0: cache is quantized from previous blocks<br/>New K/V written in FP16, then quantized"]
        B3["Quantize step: all layers, all heads<br/>Not just newly written positions"]
        B4["New K/V entries: quantized immediately after block<br/>Before next block reads them"]
        B5["Partial final block: if N % b != 0<br/>Process remaining tokens, no quantize after"]
    end

    subgraph "Error Measurement Contract"
        E1["Error = RMSE of attention output vectors<br/>Per layer, per block"]
        E2["Attention output = result of softmax(Q*K) * V<br/>Captured after attention computation"]
        E3["Baseline: same model, same prompt, FP16 cache<br/>No quantization at any point"]
        E4["Eval: pseudo-decode protocol with quantization<br/>Error measured against same-token baseline output"]
        E5["Per-head error: RMSE per head per block<br/>Useful for debugging head-specific degradation"]
        E6["Error vs context length: x-axis = cumulative tokens<br/>y-axis = RMSE (log scale recommended)"]
    end

    subgraph "Verification Checks (exit criterion)"
        V1["Monotonicity: error[block] <= error[block+1]<br/>Error must not decrease with context length"]
        V2["KVarN vs KIVI: error_kvarn[block] < error_kivi[block]<br/>For all blocks, KVarN curve is lower"]
        V3["Qwen3-4B verification: run with 1000-token prompt<br/>Both Q2_KVARN and Q4_0 (KIVI)"]
        V4["CI fallback: stories15M, 256 tokens<br/>Functional check only (no monotonicity required)"]
    end

    subgraph "Thread Safety & Memory"
        S1["Test is single-threaded<br/>No concurrent llama_decode calls"]
        S2["Attention output buffers: pre-allocated<br/>[n_layers][n_blocks][n_heads * head_dim] floats"]
        S3["Baseline and eval buffers: separate allocations<br/>No aliasing between runs"]
        S4["KV-cache reset: llama_kv_cache_clear()<br/>Between baseline and eval runs"]
    end

    subgraph "Error Conditions"
        X1["Prompt too short (N < b)<br/>-> assert or error message"]
        X2["Model does not support quantized KV<br/>-> type_k ignored, fallback to FP16"]
        X3["head_dim not divisible by 128<br/>-> assert in quantize path"]
        X4["NaN in attention output<br/>-> propagate; measurement will show inf RMSE"]
        X5["OOM during baseline (large prompt)<br/>-> reduce prompt length"]
        X6["KIVI comparison without --kivi flag<br/>-> skip comparison, only KVarN reported"]
    end

    P1 --> M1
    P1 --> M4
    P2 --> B1
    P3 --> B2
    P4 --> M5
    P5 --> E1
    P6 --> E3

    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5

    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5
    E5 --> E6

    E6 --> V1
    E6 --> V2
    V2 --> V3
    V3 --> V4
```

### Safety Contract Summary

| Condition | Assertion / Check | Location |
|-----------|-------------------|----------|
| Prompt length | `assert(N >= b)` | `PseudoDecodeHarness::tokenize_prompt()` |
| Block size | `assert(b == 128)` | `PseudoDecodeHarness` constructor |
| Head dim alignment | `assert(head_dim % 128 == 0)` | `QuantizeStep::quantize_cache()` |
| Baseline exists | `assert(baseline_attn_outputs != nullptr)` | `ErrorMeasurer::compute_all_errors()` |
| KV-cache type | `if (type_k == GGML_TYPE_F32) { skip_quantize; }` | `QuantizeStep::quantize_cache()` |
| Buffer ownership | Caller allocates; harness owns for lifetime | `PseudoDecodeHarness` |
| NaN in output | Not detected; RMSE will be `inf` | `ErrorMeasurer::compute_rmse()` |
| Reset between runs | `llama_kv_cache_clear(ctx)` | Between baseline and eval |
| Monotonicity | `all(errors[block+1] >= errors[block])` | Verification check |
| KVarN < KIVI | `all(errors_kvarn[block] < errors_kivi[block])` | Verification check |

### Model Compatibility Matrix

| Model | head_dim | Compatible | Notes |
|-------|----------|------------|-------|
| Qwen3-4B | 128 | Yes | Verification target |
| Llama 3 8B | 128 | Yes | |
| Llama 2 7B | 128 | Yes | |
| Mistral 7B | 128 | Yes | |
| Qwen2 7B | 128 | Yes | |
| Gemma 2 9B | 256 | Yes | head_dim % 128 == 0 |
| Falcon 7B | 64 | **No** | head_dim < 128 |
| GPT-2 | 64 | **No** | head_dim < 128 |
| stories15M | 64 | **No** (full quant) | CI fallback: functional only |

### Key Design Decisions

1. **Two-pass design**: Baseline (FP16) and eval (quantized) are separate runs. This avoids interference and allows clean RMSE comparison. The KV-cache is cleared between runs.

2. **Block-level measurement**: Error is measured per block (every b tokens), not per token. This matches the quantization granularity and reduces measurement noise.

3. **Quantize after block, not during**: The entire cache is quantized after each block boundary. This matches the paper's protocol and ensures all subsequent tokens see a uniformly quantized cache.

4. **KIVI comparison via re-run**: The `--kivi` flag re-runs the entire pseudo-decode protocol with `type_k = type_v = Q4_0`. This is simpler than running both types simultaneously and avoids cache interference.

5. **CI uses tiny model**: The stories15M model has head_dim=64 which is incompatible with full tile quantization. For CI, the test runs a functional check (process tokens, verify no crash) but skips the error measurement and monotonicity verification. Full verification requires Qwen3-4B.

6. **Error curve properties**: Under pseudo-decode, error is expected to be monotonically increasing because quantization error accumulates across blocks. KVarN's curve grows more slowly than KIVI's because VarN normalization suppresses the magnitude error (E_M) component.
