# 13 - llama-server: HTTP API Server for Model Inference

## Purpose

llama-server wraps the llama.cpp inference engine behind an HTTP API, providing
OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, etc.) plus Anthropic Messages, Responses, and Vertex AI
compatibility. It supports both single-model mode and a multi-model **router
mode** where a front-end process spawns child server instances per model.

## Entry Point

The `llama_server()` function in `server.cpp:77` parses CLI args, initializes
the backend, then either spawns a router or a single-model server. Two modes:

- **Single-model**: loads one model and serves all requests locally.
- **Router** (`tools/server/server-models.h`): when `--model` is empty, the
  process acts as a proxy router (`server.cpp:105-106`). It spawns child
  server processes for each model defined in a preset/ini file and forwards
  HTTP requests to them via `server_http_proxy`.

## HTTP Layer (`server-http.cpp`)

Uses the **cpp-httplib** library (`server-http.cpp:6`). The
`server_http_context` class (`server-http.h:64`) wraps an `httplib::Server`
instance.

### Key structure

- `server_http_req` (`server-http.h:46`): captures path, query, body, files,
  headers and a `should_stop` callback (tied to client connection close).
- `server_http_res` (`server-http.h:20`): can hold a full response or a
  streaming generator callback (`next` function).
- `server_http_res_ptr` (`server-http.h:37`): unique pointer alias.

### Initialization (`server_http_context::init`, `server-http.cpp:77`)

1. Reads GCP compatibility env vars (Vertex AI) (`server-http.cpp:50-75`).
2. Creates `httplib::Server` (optionally SSL) (`server-http.cpp:94-113`).
3. Installs **middlewares** via `set_pre_routing_handler`
   (`server-http.cpp:271`):
   - **CORS**: mirrors `Access-Control-Allow-Origin`, handles OPTIONS
     preflight (`server-http.cpp:272-279`).
   - **Server state**: returns 503 if model is not ready yet
     (`server-http.cpp:238-268`).
   - **API key validation**: checks `Authorization: Bearer` or `X-Api-Key`
     against configured keys (`server-http.cpp:191-236`).
4. Sets up thread pool (`server-http.cpp:296-302`): `n_threads_http` fixed
   threads + up to 1024 dynamic.
5. Serves embedded Web UI assets from `llama_ui_*` built-in data
   (`server-http.cpp:313-402`).

### Route registration methods

- `get()` (`server-http.cpp:529`): wraps httplib `Get`.
- `post()` (`server-http.cpp:546`): wraps httplib `Post`, handles
  multipart form data translation.
- `del()` (`server-http.cpp:593`): wraps httplib `Delete`.

All routes are registered in `server.cpp:200-253`. Each handler is wrapped
with `ex_wrapper()` (`server.cpp:40`) that catches `std::invalid_argument`
(-> 400) and other exceptions (-> 500).

### Streaming response

For streaming, `process_handler_response` (`server-http.cpp:492`) uses
httplib's `set_chunked_content_provider` (`server-http.cpp:521`) with a
generator lambda that calls `response->next(chunk)` repeatedly until it
returns `false`.

### GCP Vertex AI compatibility

`register_gcp_compat()` (`server-http.cpp:656`) registers a `/predict` route
that accepts Vertex AI's `instances[]` array format and dispatches each
instance to the appropriate internal handler by `@requestFormat`
(`server-http.cpp:722-761`).

## Task Queue (`server-queue.cpp` / `server-queue.h`)

The inference engine processes tasks via a producer-consumer queue.

### `server_queue` (`server-queue.h:13`)

- `post(task, front)` (`server-queue.cpp:22`): pushes a task to the deque;
  if front=true pushes to the front for high priority (e.g. cancel).
- `defer(task)` (`server-queue.cpp:63`): pushes to a separate deferred deque;
  moved to main queue when a slot frees up.
- `start_loop(idle_sleep_ms)` (`server-queue.cpp:125`): the main processing
  loop. Runs on the main thread after `ctx_server.start_loop()`
  (`server.cpp:407`).
  - Processes all tasks in `queue_tasks` via `callback_new_task`
    (`server-queue.cpp:157`).
  - Calls `callback_update_slots()` to run inference
    (`server-queue.cpp:163`).
  - Sleeps if idle for `idle_sleep_ms` (`server-queue.cpp:178-196`).
- `pop_deferred_task(id_slot)` (`server-queue.cpp:77`): moves a deferred
  task (preferring one for the specified slot) to the main queue.
- `terminate()` (`server-queue.cpp:119`): stops the loop.

### `server_response` (`server-queue.h:119`)

Results channel. Key methods:

- `recv(id_tasks)` (`server-queue.cpp:266`): blocks until a result matching
  any of the given task IDs arrives.
- `send(result)` (`server-queue.cpp:319`): pushes a result and notifies
  waiters.
- `broadcast(result)` (`server-queue.cpp:334`): sends a clone to every
  waiting task ID (used in router mode for SSE).

### `server_response_reader` (`server-queue.h:168`)

RAII wrapper that combines `server_queue` and `server_response` for
generator-style iteration:

- `post_task(task)` / `post_tasks(tasks)` (`server-queue.cpp:354,364`):
  submits tasks and registers their IDs for result collection.
- `next(should_stop)` (`server-queue.cpp:389`): polls `recv_with_timeout`,
  returns results one by one. Stops on error.
- `wait_for_all(should_stop)` (`server-queue.cpp:419`): aggregates all
  results into a `batch_response`.

### Task types (`server-task.h:16-30`)

```
SERVER_TASK_TYPE_COMPLETION, EMBEDDING, RERANK, INFILL, CANCEL,
CONTROL, NEXT_RESPONSE, METRICS, SLOT_SAVE, SLOT_RESTORE, SLOT_ERASE,
GET_LORA, SET_LORA
```

### `server_task` (`server-task.h:133`)

Carries `task_params` (sampling config, streaming flags, grammar,
response_format), tokens, and optional child tasks for parallel sampling.

### Task result types (`server-task.h`)

- `server_task_result_cmpl_final` (line 337): final completion, includes
  content, tokens, timings, stop reason.
- `server_task_result_cmpl_partial` (line 412): streaming chunk, includes
  single-token output and diff state.
- `server_task_result_embd` (line 470), `server_task_result_rerank` (line
  485), `server_task_result_error` (line 493), etc.
- Diff-based streaming state is tracked via `task_result_state` (line 103)
  which accumulates `common_chat_msg_diff` entries for partial tool calls
  and reasoning content.

## Chat Completions Endpoint (`/v1/chat/completions`)

Registered in `server.cpp:211` as:

```cpp
ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
```

The handler is defined in `server-context.cpp:4683`:

1. **Parse body** as JSON (`server-context.cpp:4686`).
2. **Call `oaicompat_chat_params_parse()`** (`server-common.cpp:904`):
   - Extracts tools, tool_choice, stop sequences, response_format.
   - Parses messages, handling multimodal content parts (image_url,
     input_audio, input_video) by replacing them with media markers
     (`server-common.cpp:984-1031`).
   - Converts OAI messages to `common_chat_msg` via
     `common_chat_msgs_parse_oaicompat()` (`server-common.cpp:1037`).
   - Calls `common_chat_templates_apply()` to render the chat template
     into a single prompt string.
   - Merges response_format (json_schema/grammar) into the request.
3. **Call `handle_completions_impl()`** (`server-context.cpp:4049`) with
   `TASK_RESPONSE_TYPE_OAI_CHAT`:
   - Tokenizes the prompt (`server-context.cpp:4082-4088`):
     - If multimodal (`mctx != nullptr`), calls `process_mtmd_prompt()`
       (`server-common.cpp:690`) to process both text and media.
     - Otherwise calls `tokenize_input_prompts()`.
   - Creates a `server_task` with type `SERVER_TASK_TYPE_COMPLETION`
     and fills `task_params` via `server_schema::eval_llama_cmpl_schema()`
     (`server-context.cpp:4102`).
   - For parallel sampling (`n_cmpl > 1`), creates child tasks
     (`server-context.cpp:4119-4124`).
   - Posts tasks to `server_response_reader` (`server-context.cpp:4129`).
   - **Non-streaming**: calls `rd.wait_for_all()` and formats the result
     as OAI chat JSON (`server-context.cpp:4138-4166`).
   - **Streaming**: calls `rd.next()` in a loop via a generator lambda
     (`server-context.cpp:4201-4302`), sending SSE-formatted chunks.
     SSE format depends on `res_type`:
     - `TASK_RESPONSE_TYPE_OAI_CHAT`: uses `format_oai_sse()`
       (`server-common.h:344`).
     - `TASK_RESPONSE_TYPE_ANTHROPIC`: uses `format_anthropic_sse()`
       (`server-common.h:349`).
     - `TASK_RESPONSE_TYPE_OAI_RESP`: uses `format_oai_resp_sse()`
       (`server-common.h:346`).

### Response formatting

`server_task_result_cmpl_final::to_json_oaicompat_chat()` (declared at
`server-task.h:397`) and the streaming variant
`to_json_oaicompat_chat_stream()` (`server-task.h:399`) produce the
OpenAI chat completions response shape including `choices`, `usage`, and
`tool_calls`.

## Tools / Function Calling Support

Tools flow through the chat completions pipeline:

1. **Parsing**: `oaicompat_chat_params_parse()` reads the `tools` array from
   the request body, parses it via `common_chat_tools_parse_oaicompat()`
   (`server-common.cpp:1038`).
2. **Template rendering**: the Jinja engine renders the tool definitions into
   the system prompt (requires `--jinja` flag, `server-common.cpp:916-918`).
3. **Grammar**: tool calling is enforced by a PEG-generated GBNF grammar that
   constrains the output to valid function call JSON.
4. **Diff-based tracking**: during streaming, `task_result_state`
   (`server-task.h:103`) accumulates partial tool calls via
   `common_chat_msg_diff` entries, updating `common_chat_msg` on each chunk
   (`server-task.h:126`, `update_chat_msg()`).
5. **Built-in server tools** (`server-tools.cpp`): llama-server also provides
   experimental built-in tools (read_file, write_file, edit_file, grep_search,
   exec_shell_command, get_datetime, file_glob_search, apply_diff) that run
   as subprocesses on the server host (`server-tools.cpp:735-745`). Enabled
   via `--tools` CLI flag (`server.cpp:255-268`). Exposed at `/tools` GET/POST
   endpoints.
6. **Tool choice**: `tool_choice` can be `"auto"`, `"none"`, `"required"`, or
   a specific tool name (`server-common.cpp:914`). Parallel tool calls is
   controlled by `parallel_tool_calls` (`server-common.cpp:1043`).

## Model Management (`server-models.cpp` / `server-models.h`)

### State machine (`server-models.h:27-35`)

```
DOWNLOADING -> DOWNLOADED -> (replaced by new instance)
UNLOADED -> LOADING -> LOADED <-> SLEEPING
```

### `server_models` (`server-models.h:105`)

Manages a `map<string, instance_t>` of model instances, each containing a
`server_subproc` (child process handle) and `server_model_meta`.

- `load_models()` (`server-models.cpp:243-261`): scans presets, model dirs,
  and cache to populate `mapping`.
- `load(name)` (`server-models.h:176`): spawns a child server process with
  args rendered from the preset.
- `unload(name)` (`server-models.h:178`): kills the child process.
- `unload_all()` (`server-models.h:179`): kills all child processes.
- `unload_lru()` (`server-models.h:139`): evicts least recently used model
  if `models_max` is reached.
- `ensure_model_ready(name)` (`server-models.h:205`): if not running, loads
  the model and blocks until ready.
- `proxy_request(req, method, name)` (`server-models.cpp:1266`): forwards an
  HTTP request to a child server instance via `server_http_proxy`.

### Router mode

In router mode (`server.cpp:154-198`), the top-level server:
- Registers proxy handlers: most API routes are redirected to
  `models_routes->proxy_post` which calls `models.proxy_request()`.
- Exposes management endpoints: `POST /models/load`, `POST /models/unload`,
  `GET /models/sse` (for real-time status updates).
- The `server_child` struct (`server-models.h:220`) handles the child side:
  `notify_to_router()` sends JSON state updates via stdout
  (`server-models.h:236`). The router receives them via
  `handle_child_state()` (`server-models.cpp:1299`).

### `server_http_proxy` (`server-models.h:277`)

A streaming HTTP proxy that forwards requests to child servers. Uses a
`pipe_t<msg_t>` (`server-models.cpp:1832-1875`) for streaming data between
the HTTP reading thread and the response writing thread.

## Multimodal Support

Multimodal input (images, audio, video) requires an `mmproj` file. Detected
via `mtmd_get_cap_from_file()` (`server-models.cpp:231`).

- Media files are received as base64 data URIs in the request body or as
  multipart file uploads (`server-http.cpp:552-577`).
- `oaicompat_chat_params_parse()` (`server-common.cpp:904`) processes
  `content` parts with `type: "image_url"`, `"input_audio"`, or
  `"input_video"`, decoding the base64 data into `raw_buffer` entries and
  replacing the content with a media marker token.
- `process_mtmd_prompt()` (`server-common.cpp:690`) takes the text prompt and
  the decoded files, creates `mtmd::bitmaps` from each file, and calls
  `mtmd_tokenize()` to produce `server_tokens` with interleaved media chunks.
- `server_tokens` (`server-common.h:130`) maps token positions to media chunks
  via `map_idx_to_media` for KV cache management.
- `/props` endpoint exposes modalities as `{"vision": bool, "video": bool,
  "audio": bool}` (`server-context.cpp:4541-4545`).

## CORS Proxy (`server-cors-proxy.h`)

Experimental feature enabled via `--cors-proxy` (`server.cpp:246-253`).

- Exposes `GET /cors-proxy?url=...` and `POST /cors-proxy`
  (`server-cors-proxy.h:77-82`).
- Parses the target URL, validates scheme (http/https only)
  (`server-cors-proxy.h:38-39`).
- Uses `server_http_proxy` to stream the response back
  (`server-cors-proxy.h:60-74`).
- Headers prefixed with `x-llama-server-proxy-header-` are forwarded as
  request headers to the target (`server-cors-proxy.h:45-57`).
- Intended for the Web UI's MCP (Model Context Protocol) proxy use case.

## Touch Points

### Inference engine (`src/llama.h`, `src/llama-context.cpp`)

The server calls `llama_decode()` and `llama_encode()` through
`server_context_impl` (defined in `server-context.cpp`). The slot system
(`server-context.cpp:57+`) manages parallel request processing slots, each
with its own KV cache cell range.

### Sampling (`sampling.h`, `common/sampling.h`)

`task_params` carries `common_params_sampling` (`server-task.h:76`), which
configures temperature, top-p, top-k, min-p, mirostat, etc.
`server_schema::make_llama_cmpl_schema()` (`server-schema.h:94`) builds the
field mapping from JSON request keys to sampling parameters.

### Grammar (`common/grammar.h`, `common/peg-parser.h`)

Grammar constraints are set via `response_format.type = "json_schema"` or
`"grammar"`. The `server_schema` module (`server-schema.h:16-105`) maps these
JSON fields to the internal grammar/schema structures. Tool calling uses PEG
parsing (`common/chat-auto-parser.h`) to build a GBNF grammar from the
chat template output.

### Chat templates (`common/chat.h`, `common/jinja/`)

`common_chat_templates_apply()` renders the conversation (messages, tools)
into a single prompt string. Jinja engine is the primary path
(`server-common.cpp:1042`). The compiled template is cached per model.

### Multimodal (`mtmd.h`, `mtmd-helper.h`)

`mtmd_tokenize()` processes images/audio/video into interleaved token+chunk
sequences. `mtmd_get_cap_from_file()` reads the mmproj to determine which
modalities a projection supports.

## Failure Modes

| Condition | HTTP Code | Source |
|---|---|---|
| Model not loaded / sleeping | 503 | `middleware_server_state`, `server-http.cpp:238` |
| Invalid API key | 401 | `middleware_validate_api_key`, `server-http.cpp:191` |
| Invalid request (bad JSON, missing fields) | 400 | `ex_wrapper`, `server.cpp:46-48` |
| Internal exception | 500 | `ex_wrapper`, `server.cpp:50-53` |
| Context size exceeded | 400 (as error result) | `server_task_result_error`, server-task.h:493 |
| Tool use without `--jinja` | throws runtime_error | `server-common.cpp:918` |
| Image without mmproj | throws runtime_error | `server-common.cpp:988` |
| Server not ready (loading model) | 503 | `middleware_server_state`, `server-http.cpp:254` |
| Model not found (router mode) | 404 | `router_validate_model`, `server-models.cpp:1616` |
| Connection closed (streaming) | (stream ends) | `req.should_stop()`, `server-http.cpp:539,586` |
| Thread pool exhaustion | (connection queued) | httplib internal |

## Files

- `tools/server/server.cpp` - Entry point
- `tools/server/server-http.cpp` / `.h` - HTTP server layer
- `tools/server/server-queue.cpp` / `.h` - Task queue / response channel
- `tools/server/server-task.h` - Task and result types
- `tools/server/server-context.cpp` / `.h` - Slot management, route handlers
- `tools/server/server-models.cpp` / `.h` - Multi-model router
- `tools/server/server-chat.cpp` / `.h` - API format conversions
- `tools/server/server-common.cpp` / `.h` - Shared utilities, OAI parsing
- `tools/server/server-schema.cpp` / `.h` - JSON schema field mapping
- `tools/server/server-tools.cpp` / `.h` - Built-in tools
- `tools/server/server-cors-proxy.h` - CORS proxy
- `tools/server/server-common.cpp` - `process_mtmd_prompt`, tokenization
