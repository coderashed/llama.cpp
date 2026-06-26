---
title: KVarN Integration Test (Reasoning Benchmark) -- Sequence Diagram
---

```mermaid
sequenceDiagram
    participant M as main()
    participant RB as ReasoningBenchmark
    participant PL as ProblemLoader
    participant CR16 as ContextRunner (FP16)
    participant CR40 as ContextRunner (Q4_0)
    participant CRKV as ContextRunner (Q2_KVARN)
    participant LC as llama_context
    participant SM as llama_sampler
    participant AE as AnswerExtractor
    participant AC as AccuracyComparator
    participant RS as ResultSet

    Note over M,AC: === PHASE 0: SETUP ===

    M->>RB: ReasoningBenchmark(model_path, problems_json, n_predict=256, seed=42)
    RB->>PL: load_problems(path)
    PL-->>RB: problems[0..N-1]
    Note over RB: Each problem has question + expected_answer

    Note over M,AC: === PHASE 1: FP16 BASELINE ===

    RB->>CR16: ContextRunner(model, type_k=F32, type_v=F32, seed=42)
    CR16->>LC: llama_init_from_model(model, params)
    Note over LC: KV-cache type = F32/F32
    LC-->>CR16: ctx_fp16
    CR16->>SM: llama_sampler_chain_init(dist, seed=42)
    SM-->>CR16: smpl

    loop for each problem in problems[0..N-1]
        CR16->>LC: llama_decode(ctx, prompt_batch)
        Note over LC: Prompt processed, KV-cache populated (FP16)
        LC-->>CR16: logits

        loop for step in 1..n_predict
            CR16->>SM: llama_sampler_sample(smpl, ctx, -1)
            SM-->>CR16: token_id
            CR16->>LC: llama_decode(ctx, batch[token_id])
            LC-->>CR16: logits
        end

        CR16->>AE: extract_answer(generated_text)
        AE-->>CR16: extracted_answer
        CR16->>AE: match_answer(extracted, expected)
        AE-->>CR16: correct (bool)
        CR16->>RS: record(correct, n_tokens)
        CR16->>LC: llama_kv_cache_clear(ctx)
    end

    CR16-->>RB: result_fp16: {correct_count, total_tokens}

    Note over M,AC: === PHASE 2: Q4_0 K-CACHE ===

    RB->>CR40: ContextRunner(model, type_k=Q4_0, type_v=Q4_0, seed=42)
    CR40->>LC: llama_init_from_model(model, params)
    Note over LC: KV-cache type = Q4_0/Q4_0
    LC-->>CR40: ctx_q4_0
    CR40->>SM: llama_sampler_chain_init(dist, seed=42)
    SM-->>CR40: smpl

    loop for each problem in problems[0..N-1]
        CR40->>LC: llama_decode(ctx, prompt_batch)
        Note over LC: Prompt processed, KV-cache populated (Q4_0)<br/>K/V quantized on write via ggml_set_rows
        LC-->>CR40: logits

        loop for step in 1..n_predict
            CR40->>SM: llama_sampler_sample(smpl, ctx, -1)
            SM-->>CR40: token_id
            CR40->>LC: llama_decode(ctx, batch[token_id])
            LC-->>CR40: logits
        end

        CR40->>AE: extract_answer(generated_text)
        AE-->>CR40: extracted_answer
        CR40->>AE: match_answer(extracted, expected)
        AE-->>CR40: correct (bool)
        CR40->>RS: record(correct, n_tokens)
        CR40->>LC: llama_kv_cache_clear(ctx)
    end

    CR40-->>RB: result_q4_0: {correct_count, total_tokens}

    Note over M,AC: === PHASE 3: Q2_KVARN K-CACHE ===

    RB->>CRKV: ContextRunner(model, type_k=Q2_KVARN, type_v=Q2_KVARN, seed=42)
    CRKV->>LC: llama_init_from_model(model, params)
    Note over LC: KV-cache type = Q2_KVARN/Q2_KVARN<br/>VarN normalization applied during cpy_k()
    LC-->>CRKV: ctx_q2_kvarn
    CRKV->>SM: llama_sampler_chain_init(dist, seed=42)
    SM-->>CRKV: smpl

    loop for each problem in problems[0..N-1]
        CRKV->>LC: llama_decode(ctx, prompt_batch)
        Note over LC: Prompt processed, KV-cache populated (Q2_KVARN)<br/>Hadamard rotation + VarN + quantize on write
        LC-->>CRKV: logits

        loop for step in 1..n_predict
            CRKV->>SM: llama_sampler_sample(smpl, ctx, -1)
            SM-->>CRKV: token_id
            CRKV->>LC: llama_decode(ctx, batch[token_id])
            LC-->>CRKV: logits
        end

        CRKV->>AE: extract_answer(generated_text)
        AE-->>CRKV: extracted_answer
        CRKV->>AE: match_answer(extracted, expected)
        AE-->>CRKV: correct (bool)
        CRKV->>RS: record(correct, n_tokens)
        CRKV->>LC: llama_kv_cache_clear(ctx)
    end

    CRKV-->>RB: result_q2_kvarn: {correct_count, total_tokens}

    Note over M,AC: === PHASE 4: COMPARISON & VERIFICATION ===

    RB->>AC: AccuracyComparator(result_fp16, result_q4_0, result_q2_kvarn, threshold=0.05)
    AC->>AC: verify_q2_kvarn_within_5pct_of_fp16()
    Note over AC: |q2_kvarn.accuracy - fp16.accuracy| <= 0.05
    AC->>AC: verify_q2_kvarn_ge_q4_0()
    Note over AC: q2_kvarn.accuracy >= q4_0.accuracy
    AC->>AC: print_report()
    Note over AC: Prints summary table:<br/>  Method    | Accuracy | Avg Tokens<br/>  FP16      | 84.1%    | 3857<br/>  Q4_0      | 80.2%    | 3957<br/>  Q2_KVARN  | 82.4%    | 3925

    AC-->>RB: pass (bool)
    RB-->>M: return 0 (pass) or 1 (fail)

    Note over M,AC: === KEY INVARIANT ===

    Note over M,AC: Same seed across all three runs ensures<br/>sampling noise is identical.<br/>Differences in accuracy are solely due to<br/>KV-cache quantization quality.<br/><br/>Token count differences reflect how quantization<br/>affects the model's confidence and branching.
```

### Per-Problem Detail

```
For each problem:
  1. Tokenize question -> prompt_tokens
  2. llama_decode(prompt) -> populate KV-cache
  3. Autoregressive loop (n_predict steps):
       sample -> decode -> append
  4. Collect full generated text
  5. Extract answer (heuristic: last number, boxed answer, etc.)
  6. Compare against expected_answer
  7. Record: correct? token_count
  8. Clear KV-cache for next problem
```
