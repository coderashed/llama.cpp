---
title: KVarN Pseudo-Decode Evaluation -- Sequence Diagram
---

```mermaid
sequenceDiagram
    participant M as main()
    participant PH as PseudoDecodeHarness
    participant BL as BaselineRunner
    participant BP as BlockProcessor
    participant QS as QuantizeStep
    participant EM as ErrorMeasurer
    participant ER as ErrorReporter
    participant LC as llama_context
    participant KC as llama_kv_cache
    participant VN as kvarn_variance_normalize
    participant QF as quantize_row_q2_kvarn_varn

    Note over M,ER: === PHASE 0: SETUP ===

    M->>PH: PseudoDecodeHarness(model, prompt, block_size=128, type=Q2_KVARN)
    PH->>PH: load_model(), tokenize_prompt()
    PH->>BL: create BaselineRunner(ctx, prompt)

    Note over M,ER: === PHASE 1: FULL-PRECISION BASELINE ===

    BL->>LC: llama_decode(ctx, batch[0..N-1])
    Note over LC: All tokens processed at once<br/>KV-cache stays FP16<br/>No quantization applied
    LC-->>BL: attention outputs per layer
    BL->>BL: capture_attention_output(il, token_idx)
    BL-->>PH: baseline_attn_outputs[n_layers][n_blocks][n_heads * head_dim]

    Note over M,ER: === PHASE 2: PSEUDO-DECODE EVALUATION ===

    PH->>PH: reset KV-cache (clear for eval run)

    Note over PH: --- Block 0: first b tokens (FP16 cache) ---

    PH->>BP: process_block(0, b, quantize_after=true)
    BP->>LC: llama_decode(ctx, batch[0..b-1])
    Note over LC: First block: cache is empty<br/>All K/V written in FP16
    LC-->>BP: attention outputs
    BP->>PH: capture eval_attn_outputs[block=0]
    BP->>QS: quantize_cache(ctx, n_layers)
    Note over QS: After first block, quantize entire cache<br/>All layers, all heads
    QS->>KC: for each layer il: quantize K and V
    KC->>VN: kvarn_variance_normalize(tile, G=128, head_dim, ...)
    VN-->>KC: S_c, S_r
    KC->>QF: quantize_row_q2_kvarn_varn(tile, block, G*head_dim, S_c, S_r)
    QF-->>KC: block_q2_kvarn[] written
    KC-->>QS: done
    QS-->>BP: cache quantized
    BP-->>PH: block 0 complete

    Note over PH: --- Block 1..N-1: subsequent blocks (quantized cache) ---

    loop for block_idx in 1..n_blocks-1
        PH->>BP: process_block(block_idx*b, (block_idx+1)*b, quantize_after)
        BP->>LC: llama_decode(ctx, batch[start..end])
        Note over LC: Cache is already quantized from previous block<br/>Attention reads quantized K/V<br/>New K/V written (will be quantized after block)
        LC-->>BP: attention outputs (with quantized cache)
        BP->>PH: capture eval_attn_outputs[block=block_idx]

        alt block_idx < n_blocks - 1
            BP->>QS: quantize_cache(ctx, n_layers)
            Note over QS: Quantize newly produced K/V entries<br/>Existing quantized entries stay as-is
            QS->>KC: quantize new K/V for each layer
            KC-->>QS: done
            QS-->>BP: new entries quantized
        end

        BP-->>PH: block block_idx complete
    end

    Note over M,ER: === PHASE 3: ERROR MEASUREMENT ===

    PH->>EM: compute_all_errors()
    Note over EM: For each layer il, for each block block_idx:<br/>  error = RMSE(baseline[il][block_idx], eval[il][block_idx])
    EM-->>PH: errors[n_layers][n_blocks]

    Note over M,ER: === PHASE 4: REPORTING ===

    PH->>ER: print_csv(stdout)
    ER-->>PH: CSV: block_idx, context_length, layer_0_error, ..., layer_N_error
    PH->>ER: print_summary()
    ER-->>PH: Summary statistics
    PH->>ER: export_json(path)
    ER-->>PH: JSON file written

    Note over M,ER: === VERIFICATION (exit criterion) ===

    PH->>PH: verify_monotonic_increasing(errors)
    PH->>PH: verify_kvarn_below_kivi(errors_kvarn, errors_kivi)
    PH-->>M: return 0 (pass) or 1 (fail)

    Note over M,ER: === KIVI COMPARISON (optional --kivi flag) ===

    PH->>PH: re-run with type=Q4_0 (KIVI)
    Note over PH: Same pseudo-decode protocol<br/>but using KIVI Q4_0 quantization<br/>Results compared side-by-side
    PH->>ER: print_comparison(errors_kvarn, errors_kivi)
    Note over ER: Verify: KVarN curve is lower<br/>than KIVI at every context length
```

### Block Processing Detail

```
Block 0 (tokens 0..127):
  Cache state: empty
  Action: process FP16, then quantize entire cache
  Error: measured against baseline block 0

Block 1 (tokens 128..255):
  Cache state: quantized (from block 0)
  Action: process with quantized cache, then quantize new entries
  Error: measured against baseline block 1

Block k (tokens k*128 .. (k+1)*128-1):
  Cache state: quantized (accumulated from blocks 0..k-1)
  Action: process with quantized cache, then quantize new entries
  Error: measured against baseline block k
  Note: error accumulates as k increases
```
