# Sampling (Item 09)

## Purpose

After the model produces logits via `llama_decode`, sampling selects the next token from the probability distribution over the vocabulary. It is the bridge between raw model output and the text generation stream.

## Architecture overview

Sampling is built on the `llama_sampler` interface (`include/llama.h:1242-1273`), a virtual dispatch table (`llama_sampler_i`) with these hooks:

| Hook | Required | Purpose |
|------|----------|---------|
| `name` | No | Debug label |
| `apply` | **Yes** | Mutate `llama_token_data_array` |
| `accept` | No | Notify sampler of chosen token |
| `reset` | No | Clear state |
| `clone` | No | Deep copy |
| `free` | No | Destructor |
| `backend_init` | No | Allocate GPU resources |
| `backend_apply` | No | Build ggml graph ops |
| `backend_accept` | No | GPU-side accept hook |
| `backend_set_input` | No | Feed runtime input tensors |

A `llama_sampler` wrapper holds an `iface` pointer and an opaque `ctx` (`include/llama.h:1275-1279`).

## Sampler chain (`llama_sampler_chain`)

Individual samplers are composed into a **chain** via `llama_sampler_chain_init` (`src/llama-sampler.cpp:792`). Each sampler in the chain runs sequentially. The chain's `apply` iterates over its vector of samplers (`src/llama-sampler.cpp:642-661`). During backend execution, GPU-capable samplers are skipped on CPU and instead run in the ggml graph (`src/llama-sampler.cpp:647-652`).

Internal data structure (`src/llama-sampler.h:12-34`):

```cpp
struct llama_sampler_chain {
    llama_sampler_chain_params params;  // no_perf flag
    bool is_init;                       // backend_init called?
    struct info { bool is_backend; llama_sampler * ptr; };
    std::vector<info> samplers;
    std::vector<llama_token_data> cur;  // pre-allocated buffer
    int64_t t_sample_us;                // timing
    int32_t n_sample;
};
```

Chain API:
- `llama_sampler_chain_init(params)` — create chain (`src/llama-sampler.cpp:792`)
- `llama_sampler_chain_add(chain, smpl)` — append sampler (takes ownership) (`src/llama-sampler.cpp:876`)
- `llama_sampler_chain_get(chain, i)` — get nth sampler; `i == -1` returns chain itself (`src/llama-sampler.cpp:884`)
- `llama_sampler_chain_remove(chain, i)` — detach (caller frees) (`src/llama-sampler.cpp:906`)
- `llama_sampler_chain_n(chain)` — count (`src/llama-sampler.cpp:919`)

## Individual samplers

All defined in `src/llama-sampler.cpp`.

| Sampler | Init function | State | GPU backend | Description |
|---------|--------------|-------|-------------|-------------|
| **Greedy** | `llama_sampler_init_greedy` (line 1011) | None | Yes (`ggml_argmax`, line 992) | Selects token with highest logit |
| **Dist** | `llama_sampler_init_dist` (line 1230) | `mt19937` RNG, seed | Yes (softmax+cumsum+uniform sampling, line 1144) | Samples from softmax distribution |
| **Top-K** | `llama_sampler_init_top_k` (line 1321) | `k` | Yes (`ggml_top_k`, line 1288) | Keep only the `k` highest-logit tokens |
| **Top-P** | `llama_sampler_init_top_p` (line 1513) | `p`, `min_keep` | Yes (sorted CDF, line 1427) | Nucleus: keep tokens with cumulative prob >= `p` |
| **Min-P** | `llama_sampler_init_min_p` (line 1670) | `p`, `min_keep` | Yes (threshold masking, line 1618) | Keep tokens with prob >= `p * max_prob` |
| **Temperature** | `llama_sampler_init_temp` (line 1884) | `temp` | Yes (scale logits, line 1822) | Divide logits by `temp`; `temp <= 0` => argmax |
| **TempExt** | `llama_sampler_init_temp_ext` (line 2081) | `temp`, `delta`, `exponent` | Yes (entropy-based dynamic temp, line 2005) | Dynamic temperature from entropy |
| **Typical** | `llama_sampler_init_typical` (line 1780) | `p`, `min_keep` | No | Locally typical sampling by entropy distance |
| **Mirostat v1** | `llama_sampler_init_mirostat` (line 2307) | `tau`, `eta`, `m`, `mu` RNG | No | Adaptive surprise-based truncation |
| **Mirostat v2** | `llama_sampler_init_mirostat_v2` (line 2411) | `tau`, `eta`, `mu` RNG | No | Simplified: truncate tokens with surprise > `mu` |
| **XTC** | `llama_sampler_init_xtc` (line 2188) | `prob`, `threshold`, `min_keep`, RNG | No | Exclude top tokens with some probability |
| **Top-n-Sigma** | `llama_sampler_init_top_n_sigma` (line 2840) | `n` | No | Mask tokens below `max - n*stddev` |
| **Penalties** | `llama_sampler_init_penalties` (line 2743) | `last_n`, `repeat`, `freq`, `present` | No | Repeat/frequency/presence penalties |
| **DRY** | `llama_sampler_init_dry` (line 3180) | `multiplier`, `base`, `allowed_length`, `penalty_last_n`, seq breakers | No | Penalize repetitive sequences (Z-algorithm) |
| **Adaptive-P** | `llama_sampler_init_adaptive_p` (line 3400) | `target`, `decay`, RNG | No | Selects tokens near a target probability using EMA |
| **Logit Bias** | `llama_sampler_init_logit_bias` (line 3569) | `(token, bias)` pairs | Yes (scatter-add, line 3484) | Add biases to specific token logits |
| **Infill** | `llama_sampler_init_infill` (line 3812) | Vocab, buffers | No | FIM-aware: merges prefix tokens, favors EOG |
| **Grammar** | `llama_sampler_init_grammar` (line 2591) | `llama_grammar *` | No | GBNF grammar constraints |
| **Grammar Lazy** | `llama_sampler_init_grammar_lazy_patterns` (line 2609) | Grammar + trigger patterns | No | Lazy grammar: only activates on trigger |
| **Empty** | Internal `llama_sampler_init_empty` (line ~476) | None | — | No-op placeholder (prefixed `?`) |

## `llama_sampler_sample` — the sampling entry point

`llama_sampler_sample(smpl, ctx, idx)` (`src/llama-sampler.cpp:806`):

1. Check if a **backend sampler** already selected the token (`llama_get_sampled_token_ith`). If so, return it immediately (line 813).
2. Build `llama_token_data_array` from the logits (lines 836-855).
3. Call `llama_sampler_apply(smpl, &cur_p)` (line 864).
4. Assert `cur_p.selected >= 0` (line 866).
5. Call `llama_sampler_accept(smpl, token)` (line 870).
6. Return the selected token ID.

## GPU-side (backend) sampling

**Backend sampling allows samplers to run on the GPU as part of the ggml compute graph.**

The `llama_context_params` struct has `samplers` / `n_samplers` fields (`include/llama.h:387-390`), where each entry is a `llama_sampler_seq_config { seq_id, sampler }` (`include/llama.h:329-332`). During context init, samplers are installed via `set_sampler()` (`llama-context.cpp:115-129`). These samplers become part of the decode graph.

Samplers that implement `backend_init`, `backend_apply`, `backend_set_input` can run their operations as ggml ops on the device. During chain `apply`, backend-capable samplers are skipped on CPU; their work is done in the graph (`src/llama-sampler.cpp:647-652`).

Backend capability check (`src/llama-sampler.cpp:559-622`): constructs a test graph with the sampler's ops and checks that the device supports every op.

Hybrid sampling works: backend runs top-k/top-p on GPU, then CPU runs dist (`test-backend-sampler.cpp:327-337`).

Test file: `tests/test-backend-sampler.cpp` (1164 lines) tests `top_k`, `temp`, `temp_ext`, `min_p`, `top_p`, `greedy`, `dist`, `logit_bias`, and multi-sequence configurations.

## Performance metrics

`llama_perf_sampler` / `llama_perf_sampler_print` / `llama_perf_sampler_reset` (`src/llama-sampler.cpp:3853-3883`) — only works with `llama_sampler_chain`. Measures total sampling microseconds and count.

## `common_sampler` — higher-level wrapper

Defined in `common/sampling.h` and `common/sampling.cpp`. Extends the chain with:

- **Grammar support**: Grammar sampler (`grmr`) runs before or after the chain. When `grammar_first == false` (default), fast path: sample token, then check grammar validity via a single-token apply; if invalid, resample with grammar applied first (`common/sampling.cpp:540-621`).
- **Reasoning budget**: `rbudget` sampler enforces a budget on `think` blocks.
- **History**: Ring buffer of last accepted tokens.
- **Candidate access**: `common_sampler_get_candidates`.
- **Speculative decoding**: `common_sampler_sample_and_accept_n` cross-references sampled tokens with draft tokens (`common/sampling.h:68-86`, `common/sampling.cpp:624-660`). Used in `common/speculative.cpp:311,765,1236`.

## Touch points

- **Inference engine**: `llama_decode` → logits → `llama_sampler_sample` → token. Backend samplers integrate into the ggml graph at context init time.
- **Grammar**: `llama_sampler_init_grammar` wraps GBNF grammar parsing. Grammar is applied as a sampler in the chain, modifying token masks. `common_sampler` additionally implements grammar-based rejection sampling (sample→validate→resample if invalid).
- **Speculative decoding**: `common_sampler_sample_and_accept_n` accepts draft tokens that match the sampled distribution, stopping on first mismatch.

## Failure modes

- **No token selected**: `GGML_ASSERT(cur_p.selected >= 0)` in `llama_sampler_sample` (`src/llama-sampler.cpp:866`). Caused by an empty chain or a final sampler that doesn't set `selected`.
- **Empty chain**: No sampler ever sets `cur_p.selected`. Ensure chain ends with `greedy`, `dist`, `mirostat`, or `adaptive_p`.
- **Grammar + backend sampling**: Explicitly unsupported — `llama_context` backend sampling combined with grammar produces a hard assert (`common/sampling.cpp:563`).
- **Reasoning budget + backend sampling**: Also unsupported (`common/sampling.cpp:564`).
- **Backend sampler not a chain**: `llama_context_params.samplers` entries must be `llama_sampler_chain` instances (`llama-context.cpp:119-120`).
- **Performance degradation with wrong sampler order**: E.g., applying penalties or grammar on the full vocabulary is slow. Apply truncating samplers (top-k, top-p) first.
