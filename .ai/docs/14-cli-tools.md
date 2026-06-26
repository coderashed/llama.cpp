# CLI Tools (Item 14)

## Overview

llama.cpp ships several CLI tools built on the `llama.h` C API, `common/` helper
library, and GGUF file format. Most tools share argument parsing via
[`common/arg.cpp`](../../common/arg.cpp) but three
(`llama-bench`, `llama-quantize`, `llama-gguf-split`) have their own parsers.

---

## Tool: `llama-cli` — Interactive / One-Shot Generation

| Aspect | Details |
|---|---|
| Entry point | `tools/cli/main.cpp:4` → `llama_cli()` in `tools/cli/cli.cpp:367` |
| Build target | `llama-cli` (links `llama-cli-impl` library, `cli.cpp:5-9`) |
| Arg parser | `common_params_parse()` at `cli.cpp:374`, example `LLAMA_EXAMPLE_CLI` |

**Architecture**

`llama_cli()` (`tools/cli/cli.cpp:367`) creates a `cli_context` (`cli.cpp:56`)
that wraps `server_context` (`ctx_server`). This means the CLI reuses the
server's task/inference pipeline (`server-context.h`, `server-task.h`). The model
is loaded via `ctx_server.load_model(params)` (`cli.cpp:413`). Inference runs
on a background thread (`cli.cpp:424-426`).

**Interaction modes**

- **One-shot**: prompt is passed via `--prompt` arg, message is immediately sent
  for generation (`cli.cpp:507-525`).
- **Interactive**: no `--prompt`, user types at a `>` prompt (`cli.cpp:499-506`).
- **Commands**: `/exit`, `/regen`, `/clear`, `/read`, `/image`, `/audio`,
  `/video`, `/glob` (`cli.cpp:245-253`).

**Output formats**: streaming text with optional color support
(`common_params::use_color`). Supports reasoning/thinking tags
(`cli.cpp:169-197`).

**Dependencies**: `server-context`, `server-task`, `common`, `sampling`.

---

## Tool: `llama-bench` — Benchmarking

| Aspect | Details |
|---|---|
| Entry point | `tools/llama-bench/main.cpp:4` → `llama_bench()` in `tools/llama-bench/llama-bench.cpp:2169` |
| Build target | `llama-bench` (links `llama-bench-impl`, `CMakeLists.txt:17-20`) |
| Arg parser | Custom `parse_cmd_params()` at `llama-bench.cpp:507` |

**Architecture**

Has its own `cmd_params` struct (`llama-bench.cpp:321-364`) and
`cmd_params_instance` (`llama-bench.cpp:1150-1179`) — does **not** use
`common_params_parse`. Tests are represented by the `test` struct
(`llama-bench.cpp:1408-1507`) which records model info, batch sizes, and sample
timings.

Two core benchmark functions:
- `test_prompt()` (not shown above but referenced at line 2336)
- `test_gen()` (at line 2348)

**Output formats**: MARKDOWN (default), CSV, JSON, JSONL, SQL
(`output_formats` enum, `llama-bench.cpp:218`). Supports output to both stdout
and stderr independently.

**Key features**: warmup runs (opt-out via `--no-warmup`), configurable
repetitions (`--repetitions`), per-device testing (`--device`), KV cache type
override (`--cache-type-k/v`), depth-benchmarking with state caching
(`llama-bench.cpp:2361-2395`).

**Failure modes**: invalid param exits via `print_usage()`; model load failure
returns 1.

---

## Tool: `llama-quantize` — Quantization

| Aspect | Details |
|---|---|
| Entry point | `tools/quantize/main.cpp:4` → `llama_quantize()` in `tools/quantize/quantize.cpp:391` |
| Build target | built from `tools/quantize/quantize.cpp` + `main.cpp` |
| Arg parser | Custom inline parsing at `quantize.cpp:406-472` |

**Architecture**

Does **not** use `common_params_parse`. Parses `--` flags manually
(`quantize.cpp:406`), then reads positional args: `<input> [output] <ftype> [nthreads]`.

Calls `llama_model_quantize()` (`quantize.cpp:634`), the C API entry point for
quantization.

**Quantization types**: defined in `QUANT_OPTIONS` vector
(`quantize.cpp:34-74`). Supports ~30 types from `Q1_0` to `F32` and `COPY`.

**Importance matrix**: loaded via `--imatrix` flag using `common_imatrix_load()`
(`common/imatrix-loader.h`). The IMatrix data is stored as KV overrides in the
output GGUF (`quantize.cpp:496-525`).

**Advanced options**:
- `--tensor-type <name>=<type>` — per-tensor quantization override
- `--prune-layers` — remove specific layers
- `--keep-split` — preserve multi-shard structure
- `--dry-run` — calculate size without performing quantization

**Failure modes**: insufficient args calls `usage()`; invalid ftype prints
error; same input/output file detected returns 1 (`quantize.cpp:607-611`).

---

## Tool: `llama-perplexity` — Perplexity Evaluation

| Aspect | Details |
|---|---|
| Entry point | `tools/perplexity/main.cpp:4` → `llama_perplexity()` in `tools/perplexity/perplexity.cpp:2011` |
| Build target | built from `tools/perplexity/perplexity.cpp` + `main.cpp` |
| Arg parser | `common_params_parse()` at `perplexity.cpp:2021`, example `LLAMA_EXAMPLE_PERPLEXITY` |

**Architecture**

Uses `common_init_from_params()` to load model+context (`perplexity.cpp:2051`).

Dispatches to one of four evaluation modes based on `common_params` fields
(`perplexity.cpp:2080-2090`):

| Mode | Flag | Function | Line |
|---|---|---|---|
| Perplexity | (default) | `perplexity()` | 444 |
| Strided perplexity | `--ppl-stride` | `perplexity_v2()` | 296 |
| HellaSwag | `--hellaswag` | `hellaswag_score()` | 744 |
| Winogrande | `--winogrande` | `winogrande_score()` | 1101 |
| Multiple choice (TruthfulQA, etc.) | `--multiple-choice` | `multiple_choice_score()` | 1405 |
| KL divergence | `--kl-divergence` | `kl_divergence()` | 1695 |

All modes use `llama_decode()` to evaluate logits and compute negative
log-likelihood (`process_logits()` at `perplexity.cpp:109-142`).

**Failure modes**: requires `--ctx-size > 0` (`perplexity.cpp:2027`); requires
at least `2 * n_ctx` tokens in prompt (`perplexity.cpp:314`).

---

## Tool: `llama-tokenize` — Tokenization Utility

| Aspect | Details |
|---|---|
| Entry point | `tools/tokenize/tokenize.cpp:187` (self-contained `main()`) |
| Build target | built from `tools/tokenize/tokenize.cpp` |
| Arg parser | Custom inline parsing at `tokenize.cpp:221-276` |

**Architecture**

Simple standalone tool. Does **not** use `common_params_parse`.

Loads model with `vocab_only = true` (`tokenize.cpp:343`). Creates a minimal
context (`tokenize.cpp:352-353`). Tokenizes the prompt with `common_tokenize()`
(`tokenize.cpp:383`), then prints token IDs and/or pieces.

Prompt sources: `--prompt`, `--file`, or `--stdin` (mutually exclusive,
`tokenize.cpp:299-308`).

**Output**: by default shows `token_id -> 'piece'`; with `--ids` shows a
Python-parseable list `[1, 2, 3]`; with `--show-count` prints total tokens.

**Failure modes**: missing `--model` returns 1 (`tokenize.cpp:288`); mutually
exclusive prompt sources checked at `tokenize.cpp:299-308`.

---

## Tool: `llama-imatrix` — Importance Matrix Computation

| Aspect | Details |
|---|---|
| Entry point | `tools/imatrix/imatrix.cpp:1058` (self-contained `main()`) |
| Build target | built from `tools/imatrix/imatrix.cpp` |
| Arg parser | `common_params_parse()` at `imatrix.cpp:1070`, example `LLAMA_EXAMPLE_IMATRIX` |

**Architecture**

Uses `common_params_parse()` with default `out_file = "imatrix.gguf"` and
`n_ctx = 512` (`imatrix.cpp:1063-1065`).

Registers a callback `ik_collect_imatrix` → `IMatrixCollector::collect_imatrix()`
via `params.cb_eval` (`imatrix.cpp:1135`). This callback is invoked by the
backend scheduler during graph evaluation for every `GGML_OP_MUL_MAT` or
`GGML_OP_MUL_MAT_ID` node (`imatrix.cpp:237-238`). It accumulates squared
activations per tensor.

**Output formats**: GGUF (default, `save_imatrix()` at line 513) or legacy `.dat`
(`save_imatrix_legacy()` at line 407).

**Modes**:
1. **Compute** — with `--prompt`, evaluates the model and collects activations.
2. **Combine** — no prompt, with `--in-file`, merges pre-computed matrices.
3. **Statistics** — `--show-statistics` prints per-tensor stats without computing.

**Failure modes**: `--ctx-size <= 0` returns 1 (`imatrix.cpp:1087`); insufficient
tokens for one full chunk returns 1 (`imatrix.cpp:799-803`); no prompt and no
`--in-file` returns 1 (`imatrix.cpp:1114-1116`).

---

## Tool: `llama-gguf-split` — GGUF File Splitting / Merging

| Aspect | Details |
|---|---|
| Entry point | `tools/gguf-split/gguf-split.cpp:573` (self-contained `main()`) |
| Build target | built from `tools/gguf-split/gguf-split.cpp` |
| Arg parser | Custom `split_params_parse()` at `gguf-split.cpp:178` |

**Architecture**

Does **not** use `common_params_parse`. Has its own `split_params` struct
(`gguf-split.cpp:41-50`) and parser.

Two operations:
- **Split** (`gguf_split()`, `gguf-split.cpp:364`) — reads a single GGUF input,
  distributes tensors across multiple output files. Split strategies: by tensor
  count (`--split-max-tensors`, default 128) or by size (`--split-max-size N(M|G)`).
- **Merge** (`gguf_merge()`, `gguf-split.cpp:402`) — reads multiple GGUF shards
  (identified by `LLM_KV_SPLIT_COUNT` metadata) and writes a single output.

Uses `gguf_init_from_file()`, `gguf_init_empty()`, `gguf_add_tensor()`,
`gguf_write_to_file()` directly.

**Split metadata**: writes `split.no`, `split.count`, `split.tensors.count` KV
pairs into each shard (`gguf-split.cpp:238-240`).

**Failure modes**: invalid params throw `std::invalid_argument` and exit
(`gguf-split.cpp:183-187`); overwrite protection on merge
(`gguf-split.cpp:410-412`); zero-tensor splits detected
(`gguf-split.cpp:227-229`).

---

## Common Arg Parser: `common/arg.cpp`

**Location**: [`common/arg.cpp`](../../common/arg.cpp)

The shared argument parsing system is built around three types defined in
[`common/arg.h`](../../common/arg.h): `common_arg`, `common_params_context`,
`common_params`.

### `common_arg` (arg.h:21-106)

Represents a single CLI option. Supports:

- **Multiple aliases**: `args` vector (e.g. `{"-c", "--ctx-size"}`)
- **Negated aliases**: `args_neg` (e.g. `{"--no-display-prompt"}`) for boolean
  flags
- **Handler type**: one of `handler_void`, `handler_string`, `handler_int`,
  `handler_bool`, `handler_str_str`
- **Example filtering**: `set_examples()` / `set_excludes()` controls which
  tools see this option
- **Environment variable**: `set_env("LLAMA_ARG_...")` enables config via env
- **Type tagging**: `set_sampling()`, `set_spec()`, `set_preset_only()` for
  help grouping

### `common_params_parser_init` (arg.cpp:1126-...)

Populates a `common_params_context` with all `common_arg` definitions. Uses the
`add_opt` lambda (arg.cpp:1161) which filters by `llama_example` to show only
relevant args per tool. The giant lambda chain defines options spanning model
loading, sampling, speculative decoding, context sizing, device selection, etc.

### `common_params_parse` (arg.cpp:1013)

Top-level wrapper:
1. Calls `common_params_parser_init()` to build ctx
2. Stores original defaults
3. Calls `common_params_parse_ex()` (arg.cpp:542) for actual parsing
4. Handles `--usage`/`--help` and `--completion-bash` after parsing

### `common_params_parse_ex` (arg.cpp:542-725)

The core parsing logic:
1. Reads environment variables for each option (arg.cpp:559-581)
2. Reads command-line args into a `std::set<...>` to detect duplicates
   (arg.cpp:590-651)
3. Replaces `_` with `-` in `--` args (arg.cpp:598-599)
4. Post-processes CPU params (arg.cpp:657-661)
5. Handles model download via `common_models_handler` (arg.cpp:675-687)
6. Validates chat template (arg.cpp:716-722)

### `common_params` (common/common.h:440-715)

The shared parameter struct aggregating all configuration: model paths,
sampling params (`common_params_sampling`), speculative decoding
(`common_params_speculative`), device offloading, LoRA, embeddings, server
config, perplexity settings, imatrix settings, etc.

### Example-based filtering

Each `common_arg` has an `examples` set. The `add_opt` lambda in
`common_params_parser_init` (arg.cpp:1161-1166) adds an option only if it
matches the current `llama_example` OR is tagged `LLAMA_EXAMPLE_COMMON` (and
the current tool is not `LLAMA_EXAMPLE_DOWNLOAD`). Options can also be
excluded via `set_excludes()`.

---

## Touch Points

### Inference Engine

All tools that run the model use `llama.h` functions:
- `llama_model_load_from_file()` / `llama_init_from_model()`
- `llama_decode()` — used by perplexity, imatrix
- `llama_model_quantize()` — used by quantize
- `llama_batch` API — used by perplexity, imatrix, bench

`llama-cli` uses `server_context::load_model()` → `llama.h` indirectly.

### Sampling

Used by `llama-cli` and `llama-completion`. The `common_params_sampling` struct
(`common/common.h:211-291`) holds all sampler config. The actual samplers are
created by `common_init_result` → `common_sampler` (`common/common.h:872`).

### Tokenization

`common_tokenize()` (`common/common.h:963-973`) converts text to token IDs.
`common_token_to_piece()` (`common/common.h:977-985`) converts token IDs back
to text. Both have overloads accepting `llama_context *` or `llama_vocab *`.

Used by: all tools that process text (cli, perplexity, imatrix, tokenize).

### Quantization

`llama_model_quantize()` is the C API entry point. Called by `llama-quantize`
(`quantize.cpp:634`). The importance matrix system passes activations through
`llama_model_quantize_params::imatrix`.

---

## Failure Modes

| Failure | Tool(s) | Detection |
|---|---|---|
| Missing `--model` | all | `common_params_parse_ex` arg.cpp:682-686 |
| Invalid CLI arg | all | `parse_cli_args` arg.cpp:600-601 |
| Model load failure | all | returns NULL from `llama_model_load_from_file` |
| Insufficient context | perplexity, imatrix | length check + error message |
| No prompt / no input | imatrix | `imatrix.cpp:1112-1117` |
| Bad quantization type | quantize | `try_parse_ftype` returns false |
| Same input/output file | quantize | `std::filesystem::equivalent` check |
| Zero-tensor split | gguf-split | `gguf-split.cpp:227-229` |
| Mutually exclusive modes | tokenize, gguf-split | combinatorial check |
