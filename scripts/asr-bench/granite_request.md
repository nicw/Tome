# Granite llama-server request contract

**Status: PINNED.** This is the single source of truth for the request shape
used to transcribe audio via the `llama-server` sidecar running the
IBM granite-speech-4.1-2b GGUF. Task 8's Swift `GraniteRequest` and the
Phase 0 Python benchmark both implement this verbatim. Do not change this
file casually — anything that depends on the request shape (Swift and
Python) must be updated together with it.

## Result: the brief's proposed shape worked on the first try

The OpenAI-compatible `input_audio` content part, exactly as drafted in the
Task 2 brief, was accepted by `llama-server` b9910 with no modification.
No fallback (alternate content-type, `llama-mtmd-cli`) was needed.

## Server launch (verbatim, matches `scripts/setup-granite-shadow.sh` and
the spec's `GraniteSidecar.start()`)

```bash
/opt/homebrew/bin/llama-server \
  -m "$HOME/Library/Application Support/Tome/Granite/granite-speech-4.1-2b-Q8_0.gguf" \
  --mmproj "$HOME/Library/Application Support/Tome/Granite/mmproj-model-f16.gguf" \
  --host 127.0.0.1 --port 8873
```

- Binds `127.0.0.1` only (never `0.0.0.0`).
- Model load + mmproj load took ~1.2 s combined in this probe run (llama.cpp
  b9910, M2 Max); server log line `llama_server: model loaded` confirms
  readiness alongside `GET /health`.
- Readiness check: poll `GET http://127.0.0.1:8873/health` until it returns
  HTTP 200 with body `{"status":"ok"}`. In this probe run it was already
  `ok` on the very first poll (~0.1 s after process spawn), well under the
  60 s timeout the spec's `GraniteSidecar.start()` uses.
- Server log emits one experimental-feature warning on load, expected and
  harmless:
  `W init_audio: audio input is in experimental stage and may have reduced quality`
  (https://github.com/ggml-org/llama.cpp/discussions/13759).

## Endpoint

```
POST http://127.0.0.1:8873/v1/chat/completions
Content-Type: application/json
```

## Request body (exact JSON shape, prompt string verbatim)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": { "data": "<base64-encoded WAV bytes>", "format": "wav" }
        },
        {
          "type": "text",
          "text": "can you transcribe the speech into a written format?"
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 2048,
  "stream": false
}
```

- `input_audio.data`: base64 of the raw WAV file bytes (16 kHz mono PCM16,
  see below — the model/mmproj combo expects this format; other sample
  rates/channel counts were not tested here and are out of scope for this
  pin).
- `input_audio.format`: literal string `"wav"`.
- Prompt text is verbatim: `can you transcribe the speech into a written format?`
  Do not reword — it has not been varied/tested against alternatives in this
  task, and the fidelity-gate work in Phase 0 assumes this exact string.
- `temperature: 0` — greedy decoding, deterministic output, required per spec.
- `max_tokens: 2048` — generous ceiling; the fox-sentence probe used only 10
  completion tokens.
- `stream: false` — the sidecar and Python client both use blocking requests.

## Response extraction path

```python
response_json["choices"][0]["message"]["content"]
```

The content is a plain string (not itself an array/content-parts structure)
containing the transcript. Strip leading/trailing whitespace before use.

Example full response body observed (probe run, see below for audio):

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "the quick brown fox jumps over the lazy dog"
      }
    }
  ],
  "created": 1783653774,
  "model": "/Users/nic/Library/Application Support/Tome/Granite/granite-speech-4.1-2b-Q8_0.gguf",
  "system_fingerprint": "b9910-f5525f7e7",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 10,
    "prompt_tokens": 46,
    "total_tokens": 56,
    "prompt_tokens_details": { "cached_tokens": 0 }
  },
  "id": "chatcmpl-MncjOJMwl0vLLOyI6PLzAhJJDAlvzrhc",
  "timings": {
    "cache_n": 0,
    "prompt_n": 46,
    "prompt_ms": 651.714,
    "prompt_per_token_ms": 14.167695652173915,
    "prompt_per_second": 70.58310854147678,
    "predicted_n": 10,
    "predicted_ms": 73.517,
    "predicted_per_token_ms": 7.351699999999999,
    "predicted_per_second": 136.02296067576208
  }
}
```

## Probe audio

Generated per the brief:

```bash
say -o /tmp/probe.aiff "the quick brown fox jumps over the lazy dog"
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/probe.aiff /tmp/probe.wav
```

Verified with `afinfo /tmp/probe.wav`:

```
Data format:     1 ch,  16000 Hz, Int16
estimated duration: 2.533250 sec
```

16 kHz, mono, Int16 (PCM16) WAV — confirmed. Note: `say`'s rendering of the
fox sentence is ~2.53 s long, not the "10 s" placeholder mentioned in the
brief's Step 1 preamble (that referred to the general idea of recording ~10
s of speech, not a hard requirement — the brief's own repro command is the
`say`/`afconvert` one-liner above, which is what was actually used).

## Probe transcript: observed vs expected

- Expected (per brief): `"the quick brown fox jumps over the lazy dog"`
  (case/punctuation may vary)
- Observed: `"the quick brown fox jumps over the lazy dog"`
- Exact match, no case or punctuation drift.

## Latency / RTF (first M2 Max datapoint)

Four requests were sent to the same warm server (model resident, KV cache
cold each time since each request is a fresh conversation):

| run | latency (s) |
|-----|-------------|
| 1 (server just became ready) | 0.75 |
| 2 | 0.109 |
| 3 | 0.091 |
| 4 | 0.092 |

Run 1 (0.75 s) includes one-time warmup costs (first-token / graph
compilation, mmproj activation) not present in steady-state calls; runs 2-4
(~0.09-0.11 s) are the representative steady-state figure.

Audio duration: 2.533 s.

- Steady-state RTF (using run 2, 0.109 s / 2.533 s): **~0.043** (i.e. ~23x
  faster than real time for this short 2.5 s clip on an M2 Max, warm
  server).
- First-call RTF (using run 1, 0.75 s / 2.533 s): **~0.30** (~3.3x
  real-time), representative of the very first request after server
  readiness.

Caveat: this is a single short (2.5 s) utterance, not the 10 s clip
originally envisioned, and is not representative of RTF on the longer
AMI/Earnings-22 Phase 0 benchmark clips — those will produce the
authoritative RTF numbers. This datapoint only confirms the request/response
plumbing and gives a rough order-of-magnitude sanity check.

## What was NOT needed

The brief's contingency path (`llama-server --help` / `docs/multimodal.md`
consultation for an alternate content type, or the `llama-mtmd-cli` HTTP
fallback) was not exercised — the OpenAI-compatible `input_audio` shape
worked immediately. This section is left here for completeness in case a
future llama.cpp upgrade breaks the shape and someone needs to know it was
deliberately not investigated further (not because alternatives don't
exist, but because the primary path succeeded).
