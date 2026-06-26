# Tokenization / Vocabulary

## Purpose

Tokenization converts human-readable text into a sequence of integer token IDs
that the neural network consumes, and converts generated token IDs back into
text (detokenization). The vocabulary stores the mapping between token strings
and their IDs, along with per-token scores and attributes. This feature supports
six tokenizer types, 55+ pre-tokenizer variants, and full Unicode handling.

## Key Types

### `llama_vocab` (src/llama-vocab.h:72)

The public interface class. It is a handle owning a `unique_ptr<impl>` (PIMPL
idiom, line 201). The `impl` struct (src/llama-vocab.cpp:1765) holds all actual
state.

**Token storage** (llama-vocab.cpp:1811-1812):
- `token_to_id`: `unordered_map<string, llama_token>` — text lookup
- `id_to_token`: `vector<token_data>` — ordered by token ID

**`token_data`** (llama-vocab.h:73-77):
- `text` (string)
- `score` (float, typically log-probability)
- `attr` (llama_token_attr bitmask)

### `llama_vocab_type` (include/llama.h:72-80)

| Enum value | Description |
|---|---|
| `LLAMA_VOCAB_TYPE_NONE` | No vocabulary (e.g. embedding models) |
| `LLAMA_VOCAB_TYPE_SPM` | SentencePiece / unigram (LLaMA) |
| `LLAMA_VOCAB_TYPE_BPE` | Byte-Pair Encoding (GPT-2) |
| `LLAMA_VOCAB_TYPE_WPM` | WordPiece (BERT) |
| `LLAMA_VOCAB_TYPE_UGM` | Unigram (T5) |
| `LLAMA_VOCAB_TYPE_RWKV` | RWKV greedy tokenization |
| `LLAMA_VOCAB_TYPE_PLAMO2` | PLaMo-2 Aho-Corasick + DP |

### `llama_vocab_pre_type` (src/llama-vocab.h:10-67)

55+ pre-tokenizer types (e.g. LLAMA_VOCAB_PRE_TYPE_LLAMA3,
LLAMA_VOCAB_PRE_TYPE_GPT2, LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER etc.). Each
selects a regex pattern set in `llm_tokenizer_bpe`'s constructor
(llama-vocab.cpp:279-553) or a custom `unicode_regex_split_custom_*` handler
in unicode.cpp (lines 215-1072).

### Tokenizer struct hierarchy (llama-vocab.cpp)

```
llm_tokenizer (virtual base, line 75-78)
  |
  +-- llm_tokenizer_spm      (line 110)
  +-- llm_tokenizer_bpe      (line 279) — holds regex_exprs + byte_encode flag
  +-- llm_tokenizer_wpm      (line 758)
  +-- llm_tokenizer_ugm      (line 881) — holds XCDA array, trie, prefix_replacements
  +-- llm_tokenizer_rwkv     (line 1285) — builds naive_trie from vocab
  +-- llm_tokenizer_plamo2   (line 1337) — builds DP table from suffixes
```

Session structs (non-virtual, created per tokenize call):
- `llm_tokenizer_spm_session` (line 114)
- `llm_tokenizer_bpe_session` (line 555) — virtual, subclassed by
  `llm_tokenizer_hybriddna_session` (line 1621) and
  `llm_tokenizer_whitespace_session` (line 1698)
- `llm_tokenizer_wpm_session` (line 762)
- `llm_tokenizer_ugm_session` (line 948)
- `llm_tokenizer_rwkv_session` (line 1301)
- `llm_tokenizer_plamo2_session` (line 1604)

## Tokenizer Types

### SPM (SentencePiece / Unigram)

Used by: LLaMA 1/2, Gemma 1/2, etc.

Algorithm (llama-vocab.cpp:117-238):
1. Split text into UTF-8 code units
2. Seed a priority queue with all adjacent bigrams that exist in the vocab
   (`try_add_bigram`, line 134-136)
3. Repeatedly merge the highest-score bigram (line 139-167)
4. Fall back to byte tokens (`<0xXX>`) for unknown sequences (line 190-196)

Whitespace: spaces are replaced with U+2581 (escaped space) via
`llama_escape_whitespace` (line 3251-3253). Detokenization reverses this
(line 3255-3257).

### BPE (Byte-Pair Encoding)

Used by: GPT-2, LLaMA 3, DeepSeek, Falcon, Qwen2, etc.

Algorithm (llama-vocab.cpp:597-717):
1. Pre-tokenize via `unicode_regex_split` with model-specific regex patterns
   (line 599)
2. Optionally apply GPT-2 byte encoding (line 599, `byte_encode` flag)
3. Seed bigram queue with BPE merge ranks from `bpe_ranks` (line 635-636)
4. Merge lowest-rank bigrams (line 640-667)
5. Fallback: byte tokens via `<0xXX>` for SPM-style or per-byte GPT-2 mapping
   (line 695-711)

Pre-tokenizer regex selection happens in `llm_tokenizer_bpe` constructor
(lines 282-548). Custom C++ handlers exist for GPT2, LLaMA3, Qwen2, Qwen3.5,
Kimi-K2, AFMoE, and Newlines patterns (unicode.cpp:215-1072).

### WPM (WordPiece)

Used by: BERT.

Algorithm (llama-vocab.cpp:765-875):
1. `preprocess` normalizes NFD, strips accents (optional), lowercases
   (optional), splits on whitespace/punctuation/chinese chars (line 812-855)
2. For each word, greedily find the longest prefix token (line 785-807)
3. Unknown words become `token_unk()` (line 806)

### UGM (Unigram)

Used by: T5, ByT5.

Algorithm (llama-vocab.cpp:964-1048):
1. Normalize input using SentencePiece-style XCDA (XOR-compressed double array)
   charsmap (line 970, implementation line 1059-1103)
2. Viterbi search: for each UTF-8 codepoint position, find all matching token
   prefixes by traversing the `token_matcher` trie; keep the highest-score path
   (lines 981-1030)
3. Backtrack from the end to extract the optimal token sequence (line 1035-1047)
4. Unknown codepoints get `token_unk()` with a score penalty (line 942, 1018)

Built from precompiled charsmap in the GGUF file (line 882-904). The XCDA
structure is a compact double-array trie (line 1114-1143).

### RWKV

Used by: RWKV models.

Algorithm (llama-vocab.cpp:1304-1329):
1. Traverse input bytes through a `naive_trie` built from unescaped token texts
   (line 1291-1295)
2. Greedily match the longest token at each position (line 1307-1328)
3. Unknown bytes become `token_unk()` (line 1310)

Tokens in the RWKV vocab are escaped strings (e.g. `\x0a` for newline); the
`llama_unescape_rwkv_token` function (line 1231-1283) decodes them.

### PLaMo-2

Used by: PLaMo-2 models.

Algorithm: Aho-Corasick with dynamic programming (llama-vocab.cpp:1477-1573):
1. Convert input to Unicode codepoints (line 1478)
2. Process from end to beginning, using a suffix-table DP to find the optimal
   tokenization (lines 1500-1538)
3. Decode the best path; fall back to byte tokens (`<0xXX>`) for unknown
   sequences (lines 1544-1570)

## The Encode Path

### `llama_tokenize` — Public C API (llama-vocab.cpp:4303-4312)

Signature:
```c
int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);
```

Delegates to `vocab->tokenize()` (line 4311).

### `llama_vocab::tokenize` (llama-vocab.cpp:4022-4045)

- Calls the `vector<string>` overload (line 4029)
- If result exceeds `n_tokens_max`, returns negative count (line 4037)
- If result exceeds `INT32_MAX`, returns `INT32_MIN` (line 4032)

### `llama_vocab::impl::tokenize` (llama-vocab.cpp:3279-3483)

Pipeline:
1. **Special token partitioning** (`tokenizer_st_partition`, line 3290):
   Scans raw text for special tokens (control, user-defined, unknown tokens by
   text match). Splits the input into a `forward_list<fragment_buffer_variant>`
   of alternating RAW_TEXT and TOKEN fragments (line 3116-3231). Only tokens with
   `ATTR_CONTROL`, `ATTR_USER_DEFINED`, or `ATTR_UNKNOWN` are in the special cache
   (line 2902-2906); sorted by descending text length (line 2908-2912).
2. **Per-type tokenization**: dispatches to the appropriate session struct
   (lines 3293-3480):
   - SPM: prefixes space if `add_space_prefix` and previous was special
     (line 3314-3315), escapes whitespace (line 3323), runs spm_session
   - BPE: creates session (hybriddna/whitespace/standard, lines 3351-3358),
     optionally appends BOS (line 3361), runs per-fragment, optionally appends
     EOS (line 3381)
   - WPM: appends BOS, runs wpm_session, appends SEP (lines 3387-3410)
   - UGM: appends BOS, runs ugm_session, appends EOS (lines 3414-3442)
   - RWKV: runs rwkv_session (lines 3446-3459)
   - PLaMo-2: runs plamo2_session (lines 3463-3476)

### `common_tokenize` wrapper (common/common.cpp:1664-1684)

Convenience function that handles buffer sizing in a loop. Calls
`llama_tokenize` with an initial estimate, resizes if the return value is
negative (indicating the required size), and throws on `INT32_MIN`.

## The Decode Path

### `llama_token_to_piece` — Public C API (llama-vocab.cpp:4314-4322)

Signature:
```c
int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);
```

Delegates to `vocab->token_to_piece()` (line 4321).

### `llama_vocab::impl::token_to_piece` (llama-vocab.cpp:3485-3600)

1. If `!special` and token is UNKNOWN or CONTROL, returns 0 (line 3489-3491)
2. If the cache (`cache_token_to_piece`) is populated, reads from it (line
   3512-3519) — built at load time in `load()` (line 2923-2932)
3. Otherwise, per-type decoding:
   - **SPM/WPM/UGM**: normal tokens get whitespace unescaped (line 3534);
     byte tokens become a single byte (line 3538-3539)
   - **BPE**: normal tokens either unescape whitespace (SPM-style BPE, line
     3552-3553) or decode via `llama_decode_text` (GPT-2 byte encoding, line
     3556); byte tokens become a single byte (line 3560-3561)
   - **RWKV**: unescapes escaped token strings (line 3566)
   - **PLaMo-2**: byte tokens decode from `<0xXX>` format (line 3580-3586);
     normal tokens copied as-is (line 3591)
4. Returns negative size if buffer too small (line 3504-3506)

### `llama_detokenize` — multi-token decode (llama-vocab.cpp:3606-3716)

Iterates tokens calling `token_to_piece` for each, then applies
`clean_spaces` post-processing (lines 3657-3713): removes spaces before
punctuation, strips single apostrophes between spaces, handles apostrophe
contractions.

## Unicode Data Tables

### Source files

- `src/unicode-data.h` (line 16-20): declarations
- `src/unicode-data.cpp`: large data tables
  - `unicode_ranges_flags` (line 10): 0x110000 codepoint range → category flags
    (letter, number, punctuation, etc.)
  - `unicode_set_whitespace` (line 2286): explicit whitespace codepoints
  - `unicode_map_lowercase` (line 2315): uppercase → lowercase mapping
  - `unicode_map_uppercase` (line 2319): lowercase → uppercase mapping
  - `unicode_ranges_nfd`: NFD decomposition ranges

### Runtime (src/unicode.h:8-111)

- `unicode_cpt_flags` (line 8): bitfield struct for category/whitespace/case
  flags, decoded from `unicode_ranges_flags` via the `unicode_cpt_flags_array`
  static function (unicode.cpp:116-146)
- `unicode_len_utf8` (unicode.cpp:16-20): UTF-8 byte length from lead byte
- `unicode_cpt_to_utf8` / `unicode_cpt_from_utf8`: codepoint ↔ UTF-8
- `unicode_cpts_from_utf8`: full string → vector of codepoints
- `unicode_cpts_normalize_nfd`: NFD normalization
- `unicode_cpt_flags_from_cpt`: codepoint → flags struct
- `unicode_tolower` (unicode.cpp:1174-1178): lowercase mapping via binary search
  on `unicode_map_lowercase`
- `unicode_byte_to_utf8` / `unicode_utf8_to_byte`: GPT-2-style byte encoding
  (unicode.cpp:148-194): maps bytes 0x00-0xFF to Unicode codepoints (bytes
  outside printable ASCII/ISO-8859-1 range mapped to 0x0100+)
- `unicode_regex_split` (unicode.cpp:1216): splits text by regex patterns,
  applies byte encoding if requested; dispatches to custom C++ handlers for
  specific patterns (GPT2, LLaMA3, Qwen2, etc.)

## Public API Surface

All functions in `include/llama.h`.

### Tokenization

| Function | Line | Description |
|---|---|---|
| `llama_tokenize` | 1135 | Text → tokens |
| `llama_token_to_piece` | 1149 | Single token → text |
| `llama_detokenize` | 1163 | Token array → text |

### Vocabulary Introspection

| Function | Line | Description |
|---|---|---|
| `llama_vocab_type` | 581 | Vocab type enum |
| `llama_vocab_n_tokens` | 583 | Vocabulary size |
| `llama_vocab_get_text` | 1064 | Token → string |
| `llama_vocab_get_score` | 1066 | Token → score |
| `llama_vocab_get_attr` | 1068 | Token → attribute bitmask |
| `llama_vocab_is_eog` | 1071 | Is end-of-generation token |
| `llama_vocab_is_control` | 1074 | Is control token |
| `llama_vocab_bos/eos/eot/sep/nl/pad/mask` | 1077-1083 | Special token IDs |
| `llama_vocab_get_add_bos/eos/sep` | 1085-1087 | Auto-add flags |
| `llama_vocab_fim_pre/suf/mid/pad/rep/sep` | 1089-1094 | FIM token IDs |
| `llama_vocab_get_text` (etc.) | 1064+ | Per-token data |

### Deprecated API (renamed to llama_vocab_* prefix)
- `llama_token_get_text` → `llama_vocab_get_text`, etc. (lines 1096-1115)

## Touch Points

Other features that call into tokenization:

- **Sampling** (`src/llama-sampler.cpp`): grammar-based samplers call
  `vocab->tokenize()` and `vocab->detokenize()` (line 2874, 2891); infill
  sampler uses `token_to_piece` (lines 3676-3686)
- **Grammar** (`src/llama-grammar.cpp`): parses special `<token>` syntax via
  `vocab->tokenize()` (line 223); evaluates grammar using `token_to_piece`
  (line 1362, 1385)
- **Batching** (`src/llama-batch.cpp`): logging uses `token_to_piece` (line 847)
- **Training/cvector** (`tools/cvector-generator`): tokenizes prompts
- **Server** (`tools/server`): prompt tokenization, output detokenization,
  token counting
- **CLI** (`tools/cli`): prompt tokenization, chat template rendering uses
  tokenization for special token matching
- **Perplexity** (`tools/perplexity`): tokenizes evaluation contexts
- **Speculative decoding** (`common/speculative.cpp`): compares draft/target
  token strings via `token_to_piece` (lines 113-114)
- **Sampling** (`common/sampling.cpp`): filters by token string, generation
  prompt tokenization (line 270)
- **Chat templates** (`common/chat.cpp`): tokenizes delimiter strings (line 146)
- **Reasoning budget** (`common/reasoning-budget.cpp`): counts tokens in
  reasoning content (line 89)
- **TTS** (`tools/tts`): tokenizes voice data and guide tokens
- **Benchmark** (`tools/llama-bench`): uses BOS token via `llama_vocab_bos`

## Loading

`llama_vocab::impl::load` (llama-vocab.cpp:1907-3002):
1. Reads `tokenizer_model` and `tokenizer_pre` from GGUF metadata (line 1912-1913)
2. Determines vocab `type` from string (lines 1939-2090): "no_vocab" → NONE,
   "llama" → SPM, "bert" → WPM, "gpt2"/"hybriddna"/"whitespace" → BPE,
   "t5" → UGM, "rwkv" → RWKV, "plamo2" → PLAMO2, "gemma4" → BPE (special case)
3. For BPE: reads BPE merge list from GGUF, populates `bpe_ranks` (lines 1969-1992)
4. For UGM: reads precompiled charsmap from GGUF (lines 2013-2032)
5. Sets default special token IDs per type (e.g. BOS=1 for SPM, BOS=101 for WPM)
6. Determines pre-tokenizer type from `tokenizer_pre` string (lines 2106-2346)
7. Reads token list from GGUF `tokenizer.ggml.tokens` (line 2378)
8. Reads optional scores and token types (lines 2385-2403)
9. Populates `token_to_id` and `id_to_token` (lines 2405-2434)
10. Reads special token IDs from GGUF metadata with overrides (lines 2484-2543)
11. Auto-detects EOT, EOM, FIM tokens by text matching (lines 2564-2717)
12. Builds special tokens cache and token_to_piece cache (lines 2900-2932)
13. Initializes the tokenizer (`init_tokenizer`, line 2454)

## Failure Modes

| Condition | Behavior | Source |
|---|---|---|
| Buffer too small (tokenize) | Returns `-(required size)` | llama-vocab.cpp:4037 |
| Buffer too small (token_to_piece) | Returns `-(required size)` | llama-vocab.cpp:3504-3506 |
| Result exceeds `INT32_MAX` | Returns `INT32_MIN` | llama-vocab.cpp:4031-4032 |
| Tokenizer not initialized | `GGML_ASSERT` abort | llama-vocab.cpp:3283, 3617 |
| Unknown tokenizer model string | `throw std::runtime_error` | llama-vocab.cpp:2089 |
| Unknown pre-tokenizer string | `throw std::runtime_error` | llama-vocab.cpp:2345 |
| BPE merges missing for BPE model | `throw std::runtime_error` | llama-vocab.cpp:1971 |
| Tokenizer list missing in GGUF | `throw std::runtime_error` | llama-vocab.cpp:2380 |
| Token not found (`text_to_token`) | Returns `LLAMA_TOKEN_NULL` (-1) | llama-vocab.cpp:3854 |
| Invalid UTF-8 in tokenize | Caught exception, returns U+FFFD | unicode.cpp:1139, llama-vocab.cpp:1217-1219 |
| XCDA/prefix_replacements OOB | `throw std::runtime_error` | llama-vocab.cpp:891, 1137, 1205 |
| `byte_to_token` not found | `std::out_of_range` from `.at()` | llama-vocab.cpp:3831, 3835, 3841 |
| Vocab type is NONE + non-NONE operation | `GGML_ABORT` | llama-vocab.cpp:3479, 3595 |
| common_tokenize INT32_MIN | `throw std::runtime_error` | common/common.cpp:1674 |
