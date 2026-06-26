# Grammar / Constrained Decoding

## Purpose

Constrained decoding solves the problem of steering autoregressive language model output to follow a formal grammar (e.g. valid JSON, a specific tool-calling format, a chess move notation). Without constraints, a model may produce tokens that violate the expected structure; grammar constraints prune invalid token choices at each step via **rejection sampling** -- setting disallowed logits to `-INFINITY` so they are never selected.

## GBNF Grammar Format and Parser (`src/llama-grammar.cpp`)

### GBNF Format

GBNF (GGML BNF) is an extended Backus-Naur Form notation documented in `grammars/README.md`. Key features:

- **Production rules**: `nonterminal ::= sequence...` (line 11)
- **Terminals**: characters/Unicode code points in quotes (`"a"`) or character ranges (`[a-z]`), with negation (`[^a]`) and any-char (`.`) (lines 43-51)
- **Alternatives**: `|` for alternation (line 56)
- **Repetition**: `*` (zero-or-more), `+` (one-or-more), `?` (optional), `{m,n}` (bounded) (lines 62-68)
- **Grouping**: parentheses `()` (line 58)
- **Token matching**: `<[token-id]>` or `<token_string>` to match specific vocabulary tokens; `!` prefix for negation (lines 71-92)
- **Comments**: `#` line comments (lines 96-100)
- **Root rule**: `root` always defines the grammar entry point (lines 104-112)

### GBNF Parser (`llama_grammar_parser`)

Defined in `src/llama-grammar.cpp` at line 86 (`struct llama_grammar_parser`). It is a recursive-descent parser that converts a GBNF string into an array of `llama_grammar_element` vectors (rules).

Parsing flow (`llama_grammar_parser::parse`, line 687):
1. Skips whitespace and comments (`parse_space`, line 125)
2. Iterates over rules (`parse_rule`, line 663): reads rule name, `::=`, then alternates
3. `parse_alternates` (line 434): parses `|`-separated sequences
4. `parse_sequence` (line 451): parses ordered elements -- literal strings, char ranges, token refs, rule refs, groups, repetition operators
5. Repetition operators (`*`, `+`, `?`, `{m,n}`) are expanded via `handle_repetitions` (line 462) which generates auxiliary rules for bounded/unbounded repetition
6. Left recursion is detected and rejected (`llama_grammar_detect_left_recursion`, line 955)

The parsed grammar is an array of `llama_grammar_rule` (each a `std::vector<llama_grammar_element>`), where each element has a type tag (`llama_gretype`) and a value.

### Grammar Element Types (`src/llama-grammar.h:13-45`)

| Type | Code | Description |
|------|------|-------------|
| `LLAMA_GRETYPE_END` | 0 | End of rule definition |
| `LLAMA_GRETYPE_ALT` | 1 | Alternative separator |
| `LLAMA_GRETYPE_RULE_REF` | 2 | Reference to another rule |
| `LLAMA_GRETYPE_CHAR` | 3 | Single character (code point) |
| `LLAMA_GRETYPE_CHAR_NOT` | 4 | Negated character |
| `LLAMA_GRETYPE_CHAR_RNG_UPPER` | 5 | Inclusive range upper bound |
| `LLAMA_GRETYPE_CHAR_ALT` | 6 | Alternate character in class |
| `LLAMA_GRETYPE_CHAR_ANY` | 7 | Any character (`.`) |
| `LLAMA_GRETYPE_TOKEN` | 8 | Token ID reference |
| `LLAMA_GRETYPE_TOKEN_NOT` | 9 | Negated token ID reference |

## JSON Schema to Grammar Bridge (`common/json-schema-to-grammar.cpp`)

The function `json_schema_to_grammar()` (line 1158) converts a JSON Schema (via `nlohmann::json`) into a GBNF grammar string.

Key conversion logic in `common_schema_converter::visit()` (line 839):
- **`type` field**: Maps to built-in GBNF primitives (e.g., `string`, `number`, `boolean`, `integer`, `object`, `array`, `null`) defined in `PRIMITIVE_RULES` (line 235)
- **`properties` / `required`**: Generates object rules with key-value pairs (`_build_object_rule`, line 640)
- **`items` / `prefixItems`**: Array rules with optional min/max items (`build_repetition`, line 17)
- **`enum` / `const`**: Literal matching (line 860-869)
- **`oneOf` / `anyOf`**: Union rules (line 847)
- **`allOf`**: Merged into intersection of property constraints (line 892)
- **`$ref`**: Resolves references via `_resolve_ref` (line 626), supports local `#/...` and remote `https://` refs
- **`minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`**: Integer range constraints via `build_min_max_int` (line 44)
- **`pattern`**: Converts regex to char-class based GBNF via `_visit_pattern` (line 348)
- **`format`**: Supports `date`, `time`, `date-time`, `uuid` (line 250)
- **`minLength` / `maxLength`**: String length constraints (line 971)

The `force_gbnf` parameter (line 1158) controls whether to use llguidance when available. When `LLAMA_USE_LLGUIDANCE` is defined and `force_gbnf` is false, the function emits a `%llguidance {}` header (line 1161).

The helper `common_schema_info::resolves_to_string()` (line 1050) probes a schema to determine if any path resolves to a string type.

## PEG Parser (`common/peg-parser.cpp`)

**Important distinction**: The PEG parser in `common/peg-parser.cpp` is NOT used for parsing GBNF. It is a **builder API for programmatically constructing GBNF grammars** from C++ code. It provides:

- A `common_peg_parser_builder` class (line 352 of `peg-parser.h`) with methods to compose grammars: `literal()`, `sequence()`, `choice()`, `one_or_more()`, `zero_or_more()`, `optional()`, `peek()`, `negate()`, `chars()`, `space()`, etc.
- The PEG arena (`common_peg_arena`, line 308 of `peg-parser.h`) holds a vector of `common_peg_parser_variant` nodes. These are variants of parser types (epsilon, start, end, literal, sequence, choice, repetition, and/not, any, space, chars, string, until, schema, rule, ref, atomic, tag, gbnf, ac).
- The `build_grammar()` method (line 1713 of `peg-parser.cpp`) converts the PEG IR into actual GBNF rules using `to_gbnf()` recursion.
- `common_peg_arena::parse()` (line 937) can execute PEG parsers directly against input strings for validation/parsing (used in the debug-template-parser tool).
- Built-in parsers: `json()`, `python_value()`, `marker()`, `quoted_string()`, etc. (lines 1391-1476)
- The Aho-Corasick integration (`ac` parser, `gbnf_including_grammar` / `gbnf_excluding_grammar`, lines 1609-1650) generates GBNF grammars for matching/avoiding specific substrings.
- Serialization: PEG parsers can be saved/loaded as JSON (`to_json`/`from_json`, lines 2028-2242).

The PEG parser relates to grammar by providing a type-safe C++ API to define structured output constraints without writing raw GBNF strings. The chat template parser in `common/chat.cpp` uses it to generate tool-calling grammars.

## llguidance Integration (`common/llguidance.cpp`)

llguidance is an external library (HTML) for grammar-constrained decoding, enabled by `cmake -DLLAMA_LLGUIDANCE=ON` (see `common/llguidance.cpp:253`).

When the grammar string starts with `%llguidance`, the sampler chain uses `llama_sampler_init_llg()` (line 219) instead of the built-in GBNF grammar sampler:

1. A `LlgTokenizer` is created from the vocab via `llama_sampler_llg_new_tokenizer()` (line 137), which translates llama tokenization into llguidance's format
2. A `LlgMatcher` is created via `llg_new_matcher()` (line 25) with the specified grammar kind (e.g. `"lark"`) and data
3. During sampling, `llama_sampler_llg_apply()` (line 46) retrieves a bitmask from `llg_matcher_get_mask()` and sets disallowed tokens' logits to `-INFINITY`
4. Token acceptance is forwarded via `llg_matcher_consume_token()` (line 42)

llguidance supports the `%llguidance {}` header in grammar strings and the `%json` directive that delegates JSON Schema validation to the library (generated by `json_schema_to_grammar()` at line 1161 when `LLAMA_USE_LLGUIDANCE` is defined).

## How Grammar Constraints Steer Sampling (Rejection Sampling)

The grammar sampler operates as a **pushdown automaton** on top of the token vocabulary:

### Initialization

`llama_grammar_init_impl()` (line 1126 / 1195) parses the grammar, builds the rule vectors, checks for left recursion, and initializes the set of active **stacks** from the start rule's alternatives. Each stack is a vector of pointers into the rule array, representing the current parsing state.

### Sampling Step (`llama_grammar_apply_impl`, line 1339)

1. If the grammar is awaiting a trigger, all tokens pass through (line 1342)
2. Each candidate token's piece is UTF-8 decoded and checked against the grammar stacks
3. `llama_grammar_reject_candidates()` (line 936) iterates over all active stacks, calling `llama_grammar_reject_candidates_for_stack()` (line 1053) to determine which tokens are disallowed
4. Context-free terminals match character-by-character; if any code point in a token fails to match, the token is rejected. Token-level terminals (`LLAMA_GRETYPE_TOKEN`) are checked against the token ID directly
5. Disallowed tokens have their logit set to `-INFINITY` (line 1378)

### Token Acceptance (`llama_grammar_accept_impl`, line 1382)

After a token is selected, `llama_grammar_accept_token()` (line 1456) advances the stacks: each stack either matches the token (consuming characters or matching token IDs) or is discarded. Stacks that accepted the token are advanced to the next grammar element, and `llama_grammar_advance_stack()` (line 853) resolves rule references by pushing all alternatives onto the worklist.

### Partial UTF-8 Handling

The grammar tracks `llama_partial_utf8` state (line 52) to handle tokens that split a multi-byte UTF-8 sequence across consecutive tokens. `decode_utf8()` (line 34) appends continuation bytes to the partial sequence and only emits complete code points once the full sequence is consumed.

## Public API

The grammar feature exposes a **sampler-based API** in `include/llama.h`:

### `llama_sampler_init_grammar()` (line 1376)

```c
struct llama_sampler * llama_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);
```

Initializes a grammar sampler from a GBNF string and root rule name. Internally calls `llama_sampler_init_grammar_impl()` (`src/llama-sampler.cpp:2532`).

### `llama_sampler_init_grammar_lazy_patterns()` (line 1395)

Lazy grammar variant that only activates grammar constraints when a trigger pattern matches the generated output. Used for tool calling where the model must generate free text before a function call. See PR #9639.

### Supporting internal API (`src/llama-grammar.h`)

- `llama_grammar_init_impl()` (line 158, 164) -- creates grammar from parsed rules or from a GBNF string
- `llama_grammar_apply_impl()` (line 179) -- applies constraints to token logits
- `llama_grammar_accept_impl()` (line 183) -- advances grammar state after token selection
- `llama_grammar_free_impl()` (line 174) -- frees grammar resources
- `llama_grammar_clone_impl()` (line 176) -- deep-copies grammar state (used for speculative decoding)

### Common layer types (`common/common.h`)

- `common_grammar_type` (line 174): `NONE`, `USER`, `OUTPUT_FORMAT`, `TOOL_CALLS`
- `common_grammar_trigger` (line 139): `TOKEN`, `WORD`, `PATTERN`, `PATTERN_FULL` trigger types
- `common_grammar` (line 182): struct pairing a type enum with a grammar string

## Touch Points

### Sampling (`src/llama-sampler.cpp`)

- `llama_sampler_grammar` struct (line 2428) holds the grammar reference
- `llama_sampler_grammar_apply` (line 2448) delegates to `llama_grammar_apply_impl`
- `llama_sampler_grammar_accept_impl` (line 2441) delegates to `llama_grammar_accept_impl`
- `llama_sampler_grammar_reset` (line 2468) reinitializes grammar via `llama_grammar_init_impl`
- `llama_sampler_grammar_clone` (line 2488) uses `llama_grammar_clone_impl`

### Common Sampling Setup (`common/sampling.cpp`)

- Detects `%llguidance` prefix to route to llguidance vs. built-in GBNF (line 201)
- Routes trigger types (word, pattern, token) from `common_grammar_trigger` (line 210)
- Handles prefill: for `OUTPUT_FORMAT` and `TOOL_CALLS` grammars, feeds generation prompt tokens into the grammar sampler before generation starts (line 285)

### Server (`tools/server/server-schema.cpp`, `server-common.cpp`)

- Accepts `json_schema` and `grammar` fields (line 248)
- Converts JSON Schema to GBNF via `json_schema_to_grammar()` (line 256)
- Supports `grammar_type: "tool_calls"` from chat template parser (line 267)
- Supports `grammar_lazy` flag and `grammar_triggers` array (line 278)
- Rejects custom grammar + tools combination (line 1065-1066)

### Inference Engine

The grammar sampler is part of the sampler chain and does not directly interact with the inference engine (no custom kernels, no backend hooks -- the `/* .backend_init */` fields in `llama_sampler_grammar_i` are all `nullptr` at `src/llama-sampler.cpp:2526`).

## Failure Modes

1. **Parse failure**: GBNF grammar fails to parse -- `llama_grammar_init_impl()` returns `nullptr` (`llama-grammar.cpp:1210`). Propagates as a runtime error in `common/sampling.cpp:263`.
2. **Left recursion**: Detected and rejected with an error log (`llama-grammar.cpp:1151`). `llama_grammar_init_impl()` returns `nullptr`.
3. **Missing root symbol**: Grammar doesn't contain the specified root symbol (`llama-grammar.cpp:1215`).
4. **Empty grammar stacks**: After accepting a token, if all stacks become empty (no valid continuation), a `std::runtime_error` is thrown (`llama-grammar.cpp:1507`).
5. **Undefined rule references**: Detected during parsing validation (`llama-grammar.cpp:701`).
6. **Excessive repetition**: Repetition count exceeds `MAX_REPETITION_THRESHOLD` (2000) (`llama-grammar.cpp:494, 652`).
7. **JSON Schema conversion errors**: Invalid schemas throw `std::invalid_argument` with details (`json-schema-to-grammar.cpp:1014`).
8. **llguidance errors**: `llg_matcher_compute_mask` can fail; error is logged and grammar is freed (`llguidance.cpp:54-57`).
9. **Tokenizer issues**: Token parsing in grammar fails if a `<token_string>` doesn't tokenize to exactly one token (`llama-grammar.cpp:226`).
10. **Prell failure**: Feeding generation prompt tokens into the grammar can fail if the grammar is too restrictive for the prefill text (`sampling.cpp:291-295`).
