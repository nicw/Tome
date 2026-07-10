# Granite Shadow Transcription — Design

**Date:** 2026-07-09
**Status:** Approved by Nic (pending spec review)
**Author:** Claude + Nic

## Why (evidence summary)

Goal: highest transcription accuracy for meetings (accented speakers, imperfect
audio); speed secondary; Tome never ships publicly (internal tool for Nic + Dan,
so licenses and sidecar processes are not gates).

Research (2026-07-09, adversarially verified against primary sources):

- **ibm-granite/granite-speech-4.1-2b** is the best meeting-audio model with a
  credible Apple Silicon path. On the Open ASR Leaderboard's AMI (meeting
  corpus): WER 7.06 (cleaned refs) vs whisper-large-v3-turbo 13.87 (−49%) and
  parakeet-tdt-0.6b-v3 9.41 (−26%). Earnings22: 8.23 vs 11.07 / 10.77.
  ~2B params, Apache-2.0, punctuation + capitalization, keyword biasing.
  (There is **no granite-speech 4.2 / “4.2.1”** — 4.1 is the latest family.)
- **Runtime:** IBM ships official GGUFs
  (`ibm-granite/granite-speech-4.1-2b-GGUF`: Q8_0 1.96 GB + f16 mmproj
  1.16 GB) with documented macOS `llama.cpp` usage; granite-speech support is
  merged in llama.cpp's multimodal (mtmd) stack. `llama-server` accepts audio.
  No published Apple Silicon RTF numbers exist; estimate RTF ≈ 0.03–0.15
  single-stream on M2 Max (unverified — measured by this feature).
- **Rejected:** `-nar` variant (batch-throughput artifact, word-drop risk in
  noise, no GGUF); `-plus` variant for now (speaker attribution + timestamps
  but drops punctuation/caps and slightly worse WER — Tome's per-segment
  pipeline gets speakers from SpeakerKit upstream anyway; revisit later);
  bosonai/higgs-audio-v3-8b-stt-v2 (worse than granite on AMI/Earnings22,
  17.8 GB, no Mac runtime); nvidia/canary-qwen-2.5b (worst meeting profile of
  the top set, NeMo/CUDA-only, no port).
- **Honest discounts:** granite trained on AMI + Earnings22 *train* splits
  (in-domain inflation — expect a smaller real-world gap), and llama.cpp's
  audio front-end WER is unvalidated vs the Python reference. **We do not
  know granite beats the current models on Nic's real meetings.** Hence:
  shadow mode, not a committed migration.

## Goals

- Behind a hidden flag (no UI), every post-processed session is *additionally*
  transcribed by granite-speech-4.1-2b via a local `llama-server` sidecar,
  producing a parallel transcript and a per-segment comparison artifact.
- Both models see **byte-identical segment audio** (same diarization, same
  merge/pad logic) so the comparison is apples-to-apples.
- Shadow is strictly best-effort: no shadow failure may fail, delay-block, or
  alter the primary transcript or the session lifecycle guarantees
  (WAV-preservation-on-failure, orphan recovery, retention).
- After ~a week of real meetings, a report script renders all comparison
  artifacts into one side-by-side HTML so Nic can judge whether granite's
  leaderboard edge is real on his audio.
- TDD throughout; existing 89-test suite stays green and untouched.

## Non-Goals (deferred until shadow results are in)

- Dual-slot coordinator / second provisioner ("live model" + "accuracy model"
  pickers) — the ~1–1.5 week full design. The shadow week decides if it's paid for.
- `supportsLive` capability flags, ModelDescriptor registry refactor, ASRBench
  library extraction (debt noted in the scalability review, not needed here).
- Any `TranscriberModel` enum case, ModelProvisioner integration, or Settings
  UI for granite — the shadow model is not user-selectable.
- `-plus` speaker-attribution vs SpeakerKit comparison (own experiment, later).
- WER scoring against ground truth (no ground truth exists; human judgment on
  disagreements is the metric).

## Decisions already made (with Nic)

| Decision | Choice |
|---|---|
| Model | `granite-speech-4.1-2b` (AR base; not -nar, not -plus) |
| Runtime | `llama-server` sidecar on IBM's official GGUFs (Q8_0 + f16 mmproj) |
| Scope | Hidden-flag shadow comparison, ~a week of real meetings, then decide |
| Speaker tagging | Stays SpeakerKit's job (per-segment pipeline); granite sees single-speaker segments |
| Setup | Manual script (brew llama.cpp + curl model download) — **curl, not URLSession** (known Tome issue: URLSession can't reach the HF CDN on some networks) |
| Bake-off | Folded into implementation task 1 (llama-server API smoke test + RTF measurement) rather than pre-design |

## Current architecture facts this design builds on

- `PostProcessingJob.run(using:)`
  ([PostProcessingJob.swift](../../../Tome/Sources/Tome/Transcription/PostProcessingJob.swift))
  is the whole post-session pipeline: diarize → re-transcribe → rebuild →
  `finalizeFrontmatter` (savedPath exists after :179) → voiceprints →
  retention → `cleanupCaptureFiles()` (:242-244) → `.complete`.
  **The capture WAVs are deleted on the success path** — any shadow work that
  reads session audio MUST run inside the job, before cleanup.
- Re-transcription is per-diarized-segment:
  `TranscriptionEngine.reTranscribe` → `SegmentReTranscriber`
  ([SegmentReTranscriber.swift](../../../Tome/Sources/Tome/Transcription/SegmentReTranscriber.swift)),
  which merges same-speaker segments < 0.5 s apart (:25-37), pads to ≥ 1.5 s
  (:41-56, Parakeet minimum), reads each segment's `AVAudioPCMBuffer` from the
  WAV, and calls `asrCoordinator.transcribe(buffer:source:)` (:65). Failures
  produce the `"[transcription failed]"` placeholder per segment (:71-78).
  Its output `[ReTranscribedSegment]` (speaker label, text, startTime) is
  exactly the per-segment primary text the comparison needs.
- Solo voice memos (≤ 1 detected speaker) skip re-transcription entirely and
  keep the live transcript (PostProcessingJob.swift:119-126).
- `ASRCoordinator` is the user-selected-model machinery (install tokens,
  provisioner ladder). The shadow path deliberately does NOT touch it.
- Jobs run serially on `PostProcessingQueue`; SettingsView locks the model
  picker while jobs run — shadow lengthens jobs, so it lengthens that lock
  window (accepted for the experiment; called out in Risks).
- `swift test` needs `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.

## Design

### 1. Flag and configuration

A `ShadowConfig` value type, read from UserDefaults **at job creation** (so
toggling applies from the next session, no restart):

- `graniteShadowEnabled` (Bool, default false) — master switch.
- `graniteShadowServerPath` (String, default `/opt/homebrew/bin/llama-server`).
- `graniteShadowModelDir` (String, default
  `~/Library/Application Support/Tome/Granite`) — must contain the Q8_0 model
  GGUF and the f16 mmproj file under the exact names the setup script
  downloads (the script is the source of truth for filenames).
- `graniteShadowPort` (Int, default 8873; server binds 127.0.0.1 only).

Enable: `defaults write com.dloomis.tome graniteShadowEnabled -bool YES`.
If the flag is on but the binary or model files are missing, the shadow phase
logs one clear diagnostic per job and skips — never errors.

### 2. Setup script

`scripts/setup-granite-shadow.sh`:
1. Checks `llama-server` exists (advises `brew install llama.cpp` if not;
   requires llama.cpp ≥ b9045 — the script checks `llama-server --version`).
2. Downloads the two GGUF files from
   `https://huggingface.co/ibm-granite/granite-speech-4.1-2b-GGUF` into the
   model dir **via curl** (resumable, `-C -`), verifying file sizes.
3. Prints the `defaults write` commands and a one-shot smoke-test command
   (transcribe a bundled 10 s WAV) so setup ends with proof-of-life.

### 3. `GraniteSidecar` (actor) — process lifecycle

Owns exactly one `llama-server` child process. Spawn-per-job (not resident):
launched at shadow-phase start, terminated at phase end — keeps ~4 GB of
model RAM off the machine between jobs; GGUF mmap reload costs seconds,
negligible against job length.

States: `idle → launching → ready → terminating → idle`, plus `failed`.
- `start()`: spawns `llama-server -m <gguf> --mmproj <mmproj> --host 127.0.0.1
  --port <port>` via `Process`, then polls `GET /health` until ready (timeout
  60 s → kill + `failed`).
- `transcribe(wavData:) async throws -> String`: POST to the server's
  OpenAI-compatible chat completion endpoint with the audio as a base64
  `input_audio` content part and the granite ASR prompt, greedy decoding
  (temperature 0). **Exact request shape/prompt is pinned by implementation
  task 1's smoke test against the live server**, then frozen in one place
  (`GraniteRequest.build(...)`, a pure function with tests).
- `stop()`: SIGTERM, escalate to SIGKILL after 5 s. Also invoked from app
  termination and `deinit` defensively — a leaked llama-server must not
  outlive Tome.
- One mid-phase relaunch: if a request fails with a connection-level error
  while `ready`, the sidecar relaunches once; a second failure marks the
  phase `failed` and remaining segments are recorded as errored.

The actor is built behind a `SidecarProcessRunning` protocol seam (spawn,
poll, request, terminate) so the state machine is fully unit-testable with a
fake — same discipline as `FakeBackend` in the existing suite.

### 4. Shared segment mechanics (small refactor, tested)

`SegmentReTranscriber` gains a `SegmentTranscribing` seam:

```swift
protocol SegmentTranscribing: Sendable {
    func transcribe(buffer: AVAudioPCMBuffer) async throws -> String
}
```

- Default conformance wraps `ASRCoordinator` (existing behavior, byte-for-byte).
- `GraniteSegmentTranscriber` wraps `GraniteSidecar` (buffer → 16 kHz mono
  WAV data → HTTP).
- The merge (< 0.5 s gap) and pad (≥ 1.5 s) logic is extracted into pure
  static functions with unit tests, used identically by both runs. The
  Parakeet-motivated padding intentionally applies to granite too — identical
  input audio is the point of the comparison.

The shadow run therefore produces `[ReTranscribedSegment]` with the same
speaker labels and start times as the primary run. Pairing keys on the merged
segment's `startTime` (both runs iterate the identical merged list, so the
key is exact), NOT on array position: the existing primary path *skips*
segments whose transcription came back empty (`guard !text.isEmpty else
continue`, SegmentReTranscriber.swift:67), so output arrays can differ in
length. A merged segment with no entry on one side is recorded as `""` for
that side in the comparison JSON.

### 5. Shadow phase placement in `PostProcessingJob`

New optional step after the voiceprint step (§2b in the job — savedPath
exists, primary transcript durable, sidecar path refreshed) and immediately
before the retention step (§3) — i.e. while the capture WAVs are still on
disk:

1. Runs only when: config present ∧ the primary path actually re-transcribed
   (`shouldRebuild` && diarized segments non-empty) ∧ the primary
   `[ReTranscribedSegment]` results were captured. Solo memos and
   relabel-fallback sessions skip with a logged reason ("no re-transcribed
   segments — nothing to compare").
2. Start sidecar → transcribe each merged segment → stop sidecar.
3. Honors cancellation cooperatively: checks `Task.isCancelled` between
   segments; on cancel, stops the sidecar, writes artifacts marked
   `"incomplete": true`, and **returns normally** (the job's own
   cancellation semantics after finalize are unchanged — shadow never throws).
4. Phase reporting: the job stays in `.finalizing` (no new `Phase` case — the
   enum is observed by UI/tests; shadow is invisible by design). Progress and
   timing go to `diagLog` with a `[SHADOW]` prefix.
5. Nothing in the phase mutates `handle.transcript`, savedPath content,
   retention behavior, or cleanup decisions.

Failure policy inside the phase: per-segment errors record
`{"error": "..."}` for that segment and continue; sidecar-level failure
(launch timeout, double connection failure, missing binary/models) abandons
remaining segments, records the reason in the JSON, logs, and returns.

### 6. Artifacts

Written to `~/Library/Application Support/Tome/GraniteShadow/` (NOT next to
the transcript — keeps the notes vault clean; the HTML report is the review
surface. Flip to vault-adjacent later if in-Obsidian reading proves wanted):

- `<session-id>.granite.md` — human-readable shadow transcript: header
  (session title, date, primary model, granite model+quant, total shadow time,
  realtime factor) + `Speaker N: text` lines.
- `<session-id>.comparison.json` — machine-readable:
  session id, transcript path, session type, primary model raw value, granite
  model id + quant + server build, incomplete flag, per-segment records
  `{startTime, speaker, durationSec, primaryText, graniteText | error,
  graniteLatencySec}`, and totals (segments, errored, audio seconds, shadow
  wall-clock, RTF).

### 7. Comparison report

`scripts/granite-shadow-report.py` (Python 3 stdlib only):
reads all `*.comparison.json`, emits `report.html` with:
- Aggregate table: sessions, segments, disagreement rate (word-level
  Levenshtein on normalized text), granite RTF distribution, error counts.
- Per-session side-by-side view, word-level diff highlighting, sorted so the
  highest-disagreement segments float to the top (those are the ones worth
  human judgment).
No WER claims — no ground truth. The report structures Nic's eyeball pass.

### 8. Testing (all without real models, network, or processes)

- **Merge/pad pure functions:** existing behavior pinned (gap merge, padding
  arithmetic, boundary clamps) — these tests also protect the primary path
  through the refactor.
- **`GraniteSidecar` state machine** (fake `SidecarProcessRunning`): launch →
  ready; launch timeout → failed + kill; connection error → one relaunch;
  second error → failed; stop always terminates; termination escalation.
- **`GraniteRequest`/response parsing:** pure request-build + response-extract
  functions, golden-file JSON.
- **Shadow phase policy** (fake `SegmentTranscribing` + temp dirs): flag off →
  no-op; missing binary/models → skip + log; happy path → both artifacts
  written, counts/pairing correct; per-segment error → recorded, phase
  completes; sidecar failure mid-run → incomplete artifacts, job still
  `.complete`; cancellation mid-run → incomplete artifacts, job semantics
  unchanged; solo-memo session → skip. Also: WAVs still present when the
  phase runs (ordering regression test).
- **`ShadowConfig`:** defaults parsing, path expansion, disabled default.
- Report script: golden-input test run in CI via `python3` if available,
  else exercised manually (script is stdlib-only, deterministic).

## Risks / open questions

- **llama-server audio API details** (exact endpoint shape/prompt for mtmd
  audio) are pinned by implementation task 1's smoke test before any Swift
  HTTP code is written. If the server path proves broken for granite audio,
  fallback is shelling out to `llama-mtmd-cli` per segment (same artifacts,
  worse latency) — decided at task 1, not later.
- **llama.cpp front-end WER fidelity** vs Python reference is exactly what
  the shadow week measures — but a gross fidelity bug (e.g. resampling error)
  would masquerade as "granite is bad". Task 1's smoke test includes one
  known-content clip sanity check.
- **Job duration grows** by the shadow time (est. 2–10 min per meeting hour);
  the Settings model-picker lock window grows with it. Accepted for the
  experiment; the report's RTF numbers feed the eventual dual-slot design.
- **Memory spike** during shadow phase: ~4 GB (Q8_0 + mmproj + KV) on top of
  the resident live model. Fine on both target machines (64 GB / Studio).
- Segment-level transcription forfeits cross-segment context granite could
  use (it's a 128k-context LLM); the comparison measures granite *in Tome's
  pipeline shape*, not granite's ceiling. Noted so a mediocre result prompts
  "try longer windows" before "reject model".

## Success criteria (end-of-week decision)

Promote granite to the full dual-slot design if, on real meetings:
1. Granite's RTF on M2 Max ≤ ~0.25 (hour meeting in ≤ 15 min), and
2. Nic's judgment on the top-disagreement segments favors granite clearly
   more often than the primary (names, accents, cross-talk are the cases to
   watch), and
3. No systemic pathologies (hallucinated segments, dropped words in noise,
   repetition loops) beyond what the primary shows.
Otherwise: keep the artifacts, write up findings, revisit when runtimes/models
move (the research memo lists granite-4.1-2b-plus and higgs-2.7B as the next
candidates to re-check).
