# Chat Templates & Auto Parser

## Purpose

Large language models each define their own prompt format for multi-turn conversations. The chat templates feature renders a sequence of chat messages (with roles like `system`, `user`, `assistant`, `tool`) into a model-specific prompt string using a Jinja2 template stored in the model's metadata. It solves two problems:

1. **Prompt formatting**: converting `[{role: "user", content: "Hello"}, {role: "assistant", content: "Hi"}]` into e.g. `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>` (ChatML).

2. **Output parsing**: after the model generates a response, the PEG-based auto parser extracts structured fields (reasoning blocks, content, tool calls) from the raw output string, eliminating fragile regex-based parsing.

---

## 1. Public C API Types

### `llama_chat_message` (`include/llama.h:436-439`)

The minimal public representation of a single chat message:

```c
typedef struct llama_chat_message {
    const char * role;    // "system", "user", "assistant", "tool"
    const char * content; // message body text
} llama_chat_message;
```

This is the C-compatible struct used by the legacy `llama_chat_apply_template` API. It carries only `role` and `content` strings; it does not support tool calls, reasoning content, or content parts.

### `common_chat_msg` (`common/chat.h:81-129`)

The richer internal representation used by the Jinja pipeline:

| Field | Type | Description |
|-------|------|-------------|
| `role` | `std::string` | Message role |
| `content` | `std::string` | Plain text content |
| `content_parts` | `vector<common_chat_msg_content_part>` | Typed content parts (text, media_marker) |
| `tool_calls` | `vector<common_chat_tool_call>` | Tool call invocations |
| `reasoning_content` | `std::string` | Separated reasoning/thinking text |
| `tool_name` | `std::string` | Tool name (for role=tool messages) |
| `tool_call_id` | `std::string` | Tool call ID (for role=tool messages) |

`common_chat_msg` also provides:
- `to_json_oaicompat()` at `common/chat.cpp:184` — serializes to OpenAI-compatible JSON
- `render_content()` at `common/chat.cpp:73` — concatenates text content parts with a delimiter

---

## 2. The Chat Template Module

### `common_chat_template` (`common/chat.h:52-79`)

A compiled, ready-to-use template. Construction at `common/chat.h:59-69`:

```
lexer(tokenize(src)) -> parser(parse_from_tokens(tokens)) -> jinja::program + caps
```

Fields:
- `prog` — the compiled `jinja::program` (AST)
- `bos_tok`, `eos_tok` — BOS/EOS token strings from the vocab
- `src` — raw Jinja source string
- `caps` — detected capabilities of the template

### `common_chat_templates` (`common/chat.cpp:334-340`)

Manages up to two template variants:
```
struct common_chat_templates {
    bool add_bos, add_eos;
    bool has_explicit_template;
    unique_ptr<common_chat_template> template_default;   // always set, falls back to ChatML
    unique_ptr<common_chat_template> template_tool_use;  // tool-use variant if the model provides one
};
```

The `tool_use` variant (obtained via `llama_model_chat_template(model, "tool_use")`) replaces the default when tools are present (`common/chat.cpp:2478-2479`).

### Initialization: `common_chat_templates_init` (`common/chat.cpp:707-805`)

1. Reads chat template from `llama_model_chat_template(model, nullptr)` or `chat_template_override`
2. Reads `tool_use` variant with `llama_model_chat_template(model, "tool_use")`
3. Falls back to hardcoded ChatML template (`CHATML_TEMPLATE_SRC` at `common/chat.cpp:658-664`) if neither is found
4. Applies known workarounds for problematic templates (Qwen, Gemma, etc. at `common/chat.cpp:738-759`)
5. Retrieves BOS/EOS tokens from the model's vocab
6. Creates `common_chat_template` instances (compiling Jinja source)
7. Returns a `unique_ptr` wrapped in `common_chat_templates_ptr`

### Template Application: `common_chat_templates_apply` (`common/chat.cpp:2671-2676`)

The main entry point. Routes to one of two paths:

**Jinja path** (`common_chat_templates_apply_jinja` at `common/chat.cpp:2474-2603`):

1. Converts `common_chat_msg` messages to JSON via `render_message_to_json()` (`common/chat.cpp:471-501`)
2. Applies workarounds: maps `developer`/`system` roles, ensures non-null content, converts tool argument types
3. Checks for specialized per-model handlers (Ministral 3, GPT-OSS, Functionary v3.2, Kimi K2, LFM2, GigaChat V3, DeepSeek V3.2, Gemma 4, Cohere2 MoE) via `common_chat_try_specialized_template()` (`common/chat.cpp:2389-2472`)
4. Falls through to the **differential auto parser** (`common/chat.cpp:2575-2602`): runs `autoparser::analyze_template()`, then `peg_generator::generate_parser()` to produce a PEG parser for the model's output format

**Legacy path** (`common_chat_templates_apply_legacy` at `common/chat.cpp:2606-2669`):

Calls the C API `llama_chat_apply_template()` which only supports a pre-defined list of templates and does not use Jinja.

### `common_chat_template_direct_apply` (`common/chat.cpp:935-939`)

Low-level function that renders a Jinja template with given parameters. Builds a `jinja::context`, populates it via `jinja::global_from_json()`, then runs `jinja::runtime::execute(tmpl.prog)` and gathers string parts. It strips BOS/EOS tokens if `add_bos`/`add_eos` flags are set.

### `common_chat_template_generation_prompt` (`common/chat.cpp:965-969`)

Renders the template both with and without `add_generation_prompt=true`, computes the difference (the suffix added by the generation prompt), and returns only that suffix. This gives the "prefix" that the model should continue from.

### `common_chat_format_single` (`common/chat.cpp:607-635`)

Formats a single new message in the context of past messages by computing the diff between the formatted history-without-new-msg and the formatted history-with-new-msg.

### `common_chat_format_example` (`common/chat.cpp:637-656`)

Returns an example rendered conversation for display/debugging purposes.

### Caps: `jinja::caps` (`common/jinja/caps.h:10-28`)

Detected template capabilities:
```
supports_tools, supports_tool_calls, supports_system_role,
supports_parallel_tool_calls, supports_preserve_reasoning,
supports_string_content, supports_typed_content, supports_object_arguments
```

Computed by `jinja::caps_get()` (`common/jinja/caps.cpp`) which statically analyzes the template AST to determine what features the template uses. Exposed via `common_chat_templates_get_caps()` (`common/chat.cpp:2758-2766`) for the server's `/v1/models` endpoint.

---

## 3. The PEG-based Auto Parser

### Architecture

The auto parser is located in `common/chat-auto-parser.h`, `common/chat-diff-analyzer.cpp`, `common/chat-auto-parser-generator.cpp`, and `common/chat-auto-parser-helpers.cpp/h`.

Instead of using regex patterns to extract structured data from model output, llama.cpp automatically generates a **PEG (Parsing Expression Grammar) parser** tailored to each model's chat template. This approach handles streaming, partial outputs, and complex nested structures (reasoning + content + tool calls) more robustly than regex.

### `autoparser` struct (`common/chat-auto-parser.h:379-406`)

The top-level analysis result:

```cpp
struct autoparser {
    jinja::caps          jinja_caps;
    std::string          user_start;        // detected user message start marker
    std::string          assistant_start;   // detected assistant message start marker
    analyze_reasoning    reasoning;         // reasoning block analysis
    analyze_content      content;           // content wrapping analysis
    analyze_tools        tools;             // tool call format analysis
    bool                 analysis_complete = false;
    std::vector<std::string> preserved_tokens;  // tokens to preserve in tokenization
};
```

### Differential Analysis: `autoparser::analyze_template()` (`common/chat-diff-analyzer.cpp`)

The auto parser works by **comparing rendered template outputs** with different input configurations. It feeds known "probe" messages (`USER_MSG`, `ASSISTANT_MSG`, `THINKING_CONTENT`, etc.) into the template and compares the resulting strings to detect structure.

#### Analyzers

Each analyzer inherits from `analyze_base` (`common/chat-auto-parser.h:239-248`) and has two roles:
1. Analyze the template (constructor performs diff-based detection)
2. Build the PEG sub-parser (`build_parser()` virtual method)

##### `analyze_reasoning` (`common/chat-auto-parser.h:254-275`)

Detects reasoning/thinking markers by:
1. `compare_reasoning_presence()` — checks if `<think>`/`</think>` or similar markers appear when reasoning content is present vs absent
2. `compare_thinking_enabled()` — compares output with `enable_thinking=true` vs `false`
3. `compare_reasoning_scope()` — checks if reasoning only appears for tool calls or always

Result: `reasoning_mode::NONE | TAG_BASED | TOOLS_ONLY`, with `start` and `end` marker strings.

##### `analyze_content` (`common/chat-auto-parser.h:281-296`)

Detects content wrapping markers:
- `content_mode::PLAIN` — no wrapping
- `content_mode::ALWAYS_WRAPPED` — e.g. `<|CHATBOT_TOKEN|>...<|END_OF_TURN_TOKEN|>`
- `content_mode::WRAPPED_WITH_REASONING` — only wrapped when reasoning is present

##### `analyze_tools` (`common/chat-auto-parser.h:302-373`)

The most complex analyzer. Detects tool call format by comparing outputs with tool calls vs without. Composed of:

- `tool_format_analysis` (`common/chat-auto-parser.h:174-192`) — overall format structure
- `tool_function_analysis` (`common/chat-auto-parser.h:194-198`) — function name markers
- `tool_arguments_analysis` (`common/chat-auto-parser.h:200-208`) — argument markers
- `tool_id_analysis` (`common/chat-auto-parser.h:210-215`) — call ID position

Tool format classifications (`tool_format` enum, lines 148-153):
- `JSON_NATIVE` — pure JSON: `{"name": "X", "arguments": {...}}`
- `TAG_WITH_JSON` — tag-based with JSON args: `<function=X>{...}</function>`
- `TAG_WITH_TAGGED` — tag-based with tagged args: `<param=key>value</param>`

### PEG Parser Generation

#### `common_chat_peg_builder` (`common/chat-peg-parser.h:46-163`)

Extends `common_peg_parser_builder` with chat-specific tags:
- Tag constants: `REASONING_BLOCK`, `REASONING`, `CONTENT`, `TOOL`, `TOOL_OPEN`, `TOOL_CLOSE`, `TOOL_ID`, `TOOL_NAME`, `TOOL_ARGS`, `TOOL_ARG`, etc.
- High-level helpers: `standard_json_tools()`, `standard_constructed_tools()`, `python_style_tool_calls()`
- Low-level helpers: `reasoning()`, `content()`, `tool()`, `tool_name()`, etc.

#### `peg_generator::generate_parser()` (`common/chat-auto-parser-generator.cpp:33-114`)

1. Runs `autoparser.analyze_template(tmpl)` to detect structure
2. Calls `autoparser.build_parser(inputs, generation_prompt)` which produces a `common_peg_arena`
3. Builds a GBNF grammar for tool call argument validation if tools are present
4. Returns `common_chat_params` with the serialized parser, grammar, and metadata

#### `autoparser::build_parser()` (`common/chat-auto-parser-generator.cpp:116-152`)

Delegates to the sub-analyzers:
```
ctx.reasoning_parser = reasoning.build_parser(ctx)
if (has_response_format) -> response_format parser
elif (has_tools)        -> tools.build_parser(ctx)  (dispatches to json_native/tag_json/tag_tagged)
else                    -> content.build_parser(ctx)
```

### Output Parsing: `common_chat_peg_parse` (`common/chat.cpp:2684-2756`)

At inference time, the model's raw output string is parsed by the previously-generated PEG parser:

1. Prepends the `generation_prompt` to the input
2. Runs the PEG parser with `COMMON_PEG_PARSE_FLAG_LENIENT`
3. On success, maps the AST to a `common_chat_msg` via `common_chat_peg_mapper` (or `common_chat_peg_gemma4_mapper` for Gemma)
4. On partial failure (streaming), returns whatever was successfully parsed
5. On full failure, throws `std::runtime_error` with format description

---

## 4. The Jinja Engine

Located in `common/jinja/`. Refer to `common/jinja/README.md` for the developer documentation.

### Pipeline

```
Source string
  -> jinja::lexer::tokenize()      (common/jinja/lexer.h, common/jinja/lexer.cpp)
  -> jinja::parse_from_tokens()    (common/jinja/parser.h, common/jinja/parser.cpp)
  -> jinja::program (AST)          (common/jinja/runtime.h:146-155)
  -> jinja::runtime::execute()     (common/jinja/runtime.h:623-651)
  -> value results                 (common/jinja/value.h)
  -> gather_string_parts()         (common/jinja/runtime.h:636-650)
```

### Key Components

#### `jinja::lexer` (`common/jinja/lexer.h:86-150`)

Predictive lexer that converts Jinja source into a `lexer_result` (tokens + source string). Recognizes `{% %}`, `{{ }}`, `{# #}` delimiters, string/numeric literals, operators, identifiers. Unlike huggingface.js, input is not pre-processed — the lexer works on raw source to enable source-level error tracing.

#### `jinja::parser` (`common/jinja/parser.h`)

Consumes tokens and compiles into a `jinja::program` (AST). May throw `parser_exception` on error.

#### AST Node Types (`common/jinja/runtime.h:105-578`)

All nodes inherit from `statement` (line 105), which has a virtual `execute_impl(context &)` method.

| Node | File:Line | Purpose |
|------|-----------|---------|
| `program` | `runtime.h:146` | Root node, holds body statements |
| `if_statement` | `runtime.h:157` | `{% if %}...{% endif %}` |
| `for_statement` | `runtime.h:178` | `{% for %}...{% endfor %}` |
| `set_statement` | `runtime.h:231` | `{% set x = ... %}` |
| `identifier` | `runtime.h:310` | Variable reference |
| `string_literal` | `runtime.h:337` | String constant |
| `binary_expression` | `runtime.h:396` | Arithmetic/comparison operators |
| `filter_expression` | `runtime.h:414` | Pipe filters: `val \| filter` |
| `member_expression` | `runtime.h:280` | `obj.prop` or `obj[expr]` |
| `call_expression` | `runtime.h:294` | Function/method calls |
| `ternary_expression` | `runtime.h:558` | `x if cond else y` |

#### `jinja::runtime` (`common/jinja/runtime.h:623-651`)

Executes a compiled program with a given context. `execute()` iterates over the program's body statements, calling `execute(ctx)` on each. Results are collected into a `value_array`.

`gather_string_parts()` (line 636) flattens nested string/array results into a `value_string` with `is_input` provenance metadata, then joins consecutive parts with the same input flag.

#### `jinja::context` (`common/jinja/runtime.h:50-100`)

The execution environment. Contains:
- `src` — shared pointer to the source string (for error reporting)
- `env` — a `value_object` with built-in variables (`true/false/none`) and user-defined variables
- `current_time` — for `now` variable support

Created per template application in `common_chat_template_direct_apply_impl()` (`common/chat.cpp:889`).

#### `jinja::value` (`common/jinja/value.h`)

Primitive types:
- `value_int_t` (line 214), `value_float_t` (line 248), `value_bool_t` (line 326)
- `value_string_t` (line 290) — carries `string` with provenance tracking
- `value_array_t` (line 354), `value_tuple_t` (line 463), `value_object_t` (line 483)
- `value_none_t` (line 602), `value_undefined_t` (line 620)
- `value_func_t` (line 697) — for built-in functions

#### `jinja::string` (`common/jinja/string.h`)

Wraps `std::string` with input provenance tracking:
- `string_part` (line 16): has `is_input` flag and `val`
- `string` (line 24): vector of `string_part`s, with `mark_input()`, `append()`, `uppercase()`, `lowercase()`, `strip()`, `capitalize()`, `titlecase()`
- Transformations preserve `is_input` according to one-to-one, one-to-many, and many-to-one rules

#### Input Marking

Security feature (see `common/jinja/README.md` lines 32-88). User-provided message content is marked with `is_input=true` via `jinja::global_from_json()` with `mark_input=true`. This prevents special token injection attacks — downstream parsers can reject or escape strings that contain special tokens within user input.

### `jinja::global_from_json()` (`common/jinja/value.h:90-91`)

Converts a `nlohmann::ordered_json` object into Jinja runtime values, recursively populating the context's environment. When `mark_input=true`, user message contents are wrapped with the input-provenance flag.

---

## 5. Template-to-llama_chat_message Mapping

The `common_chat_msg` struct (`common/chat.h:81-129`) maps to the OpenAI chat completion format:

| JSON field | `common_chat_msg` member |
|------------|--------------------------|
| `role` | `role` |
| `content` (string) | `content` |
| `content` (array of parts) | `content_parts` |
| `tool_calls` | `tool_calls` (as `common_chat_tool_call`) |
| `reasoning_content` | `reasoning_content` |
| `name` | `tool_name` |
| `tool_call_id` | `tool_call_id` |

Parsing from JSON: `common_chat_msgs_parse_oaicompat()` (`common/chat.cpp:370-469`)
Serialization to JSON: `common_chat_msg::to_json_oaicompat()` (`common/chat.cpp:184-262`)
Rendering for Jinja context: `render_message_to_json()` (`common/chat.cpp:471-501`)

`render_message_to_json()` respects the template's `caps` to determine whether to send content as a plain string, as typed parts, or to allow both formats.

---

## 6. The `llama_chat_apply_template` Function

### Declaration (`include/llama.h:1186-1192`)

```c
LLAMA_API int32_t llama_chat_apply_template(
    const char * tmpl,
    const struct llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length);
```

### Implementation (`src/llama.cpp:470-499`)

1. Defaults template to `"chatml"` if `tmpl` is null
2. Detects the template type via `llm_chat_detect_template()` — returns `LLM_CHAT_TEMPLATE_UNKNOWN` if not in the pre-defined list, causing `-1` return
3. Applies the template via `llm_chat_apply_template()` which uses hardcoded C++ logic per template type
4. Returns the formatted string length (or `-1` on error)

**Important**: This function does **not** use the Jinja engine. It supports only a pre-defined list of templates documented at the GitHub wiki link in the comment (`include/llama.h:1178`).

---

## 7. Specialized Model-Specific Handlers

When a template is recognized by `common_chat_try_specialized_template()` (`common/chat.cpp:2389-2472`), it bypasses the differential auto parser and uses a handcrafted PEG parser:

| Template | Detection string | Handler | File:Line |
|----------|-----------------|---------|-----------|
| Ministral/Mistral Large 3 | `[SYSTEM_PROMPT]`, `[TOOL_CALLS]`, `[ARGS]`, no `[CALL_ID]` | `common_chat_params_init_ministral_3` | `chat.cpp:971-1103` |
| GPT-OSS | `<\|channel\|>` | `common_chat_params_init_gpt_oss` | `chat.cpp:1105-1266` |
| Functionary v3.2 | `>>>all`, `>>>${recipient}` | `common_chat_params_init_functionary_v3_2` | `chat.cpp:1432-1529` |
| Kimi K2 | `<\|tool_calls_section_begin\|>`, `<\|tool_call_begin\|>` | `common_chat_params_init_kimi_k2` | `chat.cpp:1533-1661` |
| Cohere2 MoE / North Code | `<\|START_TEXT\|>`, `<\|START_ACTION\|>` | `common_chat_params_init_cohere2moe` | `chat.cpp:2050-2172` |
| LFM2 | `<\|tool_list_start\|>` | `common_chat_params_init_lfm2` | `chat.cpp:1667-1772` |
| LFM2.5 | `List of tools: [`, no `<\|tool_list_start\|>` | `common_chat_params_init_lfm2` (tool_list_tokens=false) | `chat.cpp:1667-1772` |
| GigaChat V3 | `<\|role_sep\|>`, `<\|message_sep\|>`, no `<\|function_call\|>` | `common_chat_params_init_gigachat_v3` | `chat.cpp:1774-1852` |
| DeepSeek V3.2 | `dsml_token`, `function_calls`, `DSML` | `common_chat_params_init_deepseek_v3_2` | `chat.cpp:1854-2032` |
| Gemma 4 | `'<\|tool_call\|>call:'` | `common_chat_params_init_gemma4` | `chat.cpp:1268-1429` |

Each handler produces a `common_chat_params` with:
- `prompt` — the fully rendered prompt string
- `generation_prompt` — the suffix that triggers model generation
- `parser` — the serialized PEG arena
- `grammar` — GBNF grammar for tool call argument validation
- `grammar_triggers` — trigger words for lazy grammar activation
- `message_delimiters` — role-based delimiters for splitting previous turns

---

## 8. Touch Points

### Server (`tools/server/`)

- `server-context.cpp:1497-1527` — creates `common_chat_templates_ptr` during context initialization, sets up template kwargs
- `server-common.cpp:1034-1091` — applies templates with `common_chat_templates_apply()` for chat completions
- `server-context.cpp:3989` — reports `chat_template_caps` in model metadata
- `server-context.cpp:4533-4562` — reports chat template sources in `/v1/models` endpoint

### CLI / `main` example (`examples/main/`)

Uses `common_chat_templates_init()` and `common_chat_templates_apply()` for interactive chat sessions. The `--chat-template` and `--chat-template-file` CLI flags allow overrides.

### `simple-chat` example (`examples/simple-chat/simple-chat.cpp:170-194`)

Uses the legacy `llama_chat_apply_template()` C API directly.

### Android app (`examples/llama.android/`)

Uses `common_chat_templates_init()` at app startup and `common_chat_format_single()` for individual message formatting (with `use_jinja=false`).

### Diffusion example (`examples/diffusion/diffusion-cli.cpp:75-97`)

Applies chat templates to format text-to-image prompts.

---

## 9. Failure Modes

| Error | Source | Mechanism |
|-------|--------|-----------|
| Invalid Jinja template syntax | `common/jinja/lexer.cpp`, `common/jinja/parser.cpp` | Throws `lexer_exception` or `parser_exception` (both inherit `std::runtime_error`) |
| Template application failure | `common_chat_templates_init()` at `chat.cpp:789-796` | Catches exceptions, logs error, suggests `--no-jinja`, rethrows |
| Unsupported custom template (legacy) | `common_chat_templates_apply_legacy()` at `chat.cpp:2646` | Returns `-1` from `llama_chat_apply_template()`, throws `std::runtime_error("this custom template is not supported, try using --jinja")` |
| Model output doesn't match format | `common_chat_peg_parse()` at `chat.cpp:2710-2734` | On full parse failure throws `std::runtime_error("model produced output that does not match the expected ... format")` |
| Grammar + tools conflict | `common_chat_templates_apply_jinja()` at `chat.cpp:2546` | `std::runtime_error("Cannot specify grammar with tools")` |
| Autoparser generation failure | `common_chat_templates_apply_jinja()` at `chat.cpp:2600-2602` | Catches all `std::exception`, wraps in `std::invalid_argument("Unable to generate parser for this template...")` |
| Content + content_parts both set | `common_chat_msg::render_content()` at `chat.cpp:74` | `std::runtime_error("Cannot specify both content and content_parts")` |
| Invalid diff (tool call count decreased) | `common_chat_msg_diff::compute_diffs()` at `chat.cpp:283-294` | `std::runtime_error("Invalid diff: now finding less tool calls!...")` |
| Messages JSON parsing error | `common_chat_msgs_parse_oaicompat()` at `chat.cpp:370-469` | `std::runtime_error("Failed to parse messages: ...")` or `std::invalid_argument` for specific field errors |

Errors are generally reported through C++ exceptions. In the server, these are caught at the HTTP handler level and returned as JSON error responses. In CLI tools, exceptions propagate to `main()` which prints the error message.
