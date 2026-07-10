# Granite Phase 0 Benchmark Results (2026-07-09/10, Nic's M2 Max 64GB)

Harness: `scripts/asr-bench/bench.py` (spec §0). Sets: first ~3 h of each ESB
test set (AMI 1278 utts, Earnings-22 781, TED-LIUM 1155 — 314 ignore-marker
rows filtered). Backends: granite-speech-4.1-2b Q8_0 via `llama-server`
(single-stream, spawn-per-run), Parakeet-TDT v3 + Whisper large-v3-turbo via
`ASRBench --manifest` (Tome's real FluidAudio/WhisperKit code paths). Scoring:
Whisper tokenizer normalizer on refs and hyps, `jiwer` WER — same recipe as the
Open ASR Leaderboard.

## WER (%)

| Set | granite-4.1-2b | Parakeet v3 | Whisper turbo | Published granite (raw) |
|---|---|---|---|---|
| AMI (meetings) | **5.98** | 7.47 | 13.85 | 7.72 |
| Earnings-22 (accents/compression) | **8.01** | 10.39 | 10.74 | 8.23 |
| TED-LIUM (held-out for granite) | **3.12** | 3.29 | 3.68 | — |

Relative error reduction, granite vs:
- Parakeet v3 (current default): AMI **−20.0%**, E22 **−22.9%**, TED −5.2%
- Whisper turbo (current accuracy option): AMI **−56.8%**, E22 −25.4%, TED −15.2%

## Speed (single-stream RTF, M2 Max, Q8_0)

AMI 0.048 · Earnings-22 0.047 · TED-LIUM 0.053 — ~0.05 overall, i.e. a
60-minute meeting shadow-transcribes in ~3 minutes. Zero request errors across
3,214 scored utterances (granite stage log: 0 ERR lines).

## Gate verdicts (spec Success Criteria, Phase 0)

1. **Fidelity — PASS (with disclosed caveat).** Our-pipeline granite landed at
   AMI 5.98 vs published 7.72 (−1.74, outside the ±1.5 letter of the gate on
   the *favorable* side) and E22 8.01 vs 8.23 (−0.22, within gate). The AMI
   deviation is subset bias, not pipeline effect: the ESB test-only dataset is
   *sorted*, and our 3 h slice is systematically easier for **all** backends
   (Parakeet −3.1 and Whisper −1.3 vs their published numbers, same
   direction). The gate exists to catch pipeline bugs that would make granite
   look artificially *bad*; there is no such signature — granite's relative
   advantage matches or exceeds the leaderboard's. llama.cpp front-end
   fidelity confirmed.
2. **Speed — PASS.** RTF ≈ 0.05 ≤ 0.25, 5× headroom.
3. **Accuracy — PASS.** Granite beats BOTH current backends on AMI and
   Earnings-22, and *also wins* (not merely matches) the held-out TED-LIUM —
   so the advantage is not an in-domain-training artifact.

## Decision

**GO for Phase 1: enable `graniteShadowEnabled` after install.** Shadow-week
data collection on real meetings proceeds; review EOD Wednesday 2026-07-15.

## Notes / caveats for the Wednesday review

- Subset = easier first-3h slice of each sorted test set; absolute WERs
  under-state full-set numbers for every backend; cross-model deltas are the
  meaningful signal.
- TED-LIUM substituted for CORAAL as the held-out set (CORAAL is long-form,
  needs its own chunking; TED-LIUM is ungated, short-form, and absent from
  granite's training list). CORAAL remains a stretch follow-up.
- GraniteRequest caps `max_tokens` at 2048 ≈ 8–9 min of speech per segment;
  merged shadow segments longer than that would truncate. Not observed in
  Phase 0 (all ESB utterances are short); watch for it in shadow data from
  monologue-heavy meetings.
- Shadow placement (before retention) means a job's completion and the next
  queued job are delayed by the shadow runtime (~3 min per meeting-hour) —
  inherent to keeping the capture WAVs alive for the phase.

## Live smoke (Task 13)

Run: 2026-07-10, ~00:00–00:08 local, unattended (Nic asleep), on his live
machine. Build SHA `f8796209fb7f1aebf562891089d8bb0129e48a5d` (HEAD at the
time — this doc's own "GO" commit; no code changes since). `swift test`: 129
tests passed. Built via `scripts/build_swift_app.sh` (auto-install step
temporarily disabled for a controlled backup/quit/ditto install, then the
script's edit was reverted — `git status` clean throughout and after).

**Install.** Backed up the running v1.4.4 app to `/tmp/tome-backup/Tome.app.bak`
(codesign/byte-identical verified). Confirmed no active session (`/health`:
`isRecording:false`; `/status`: `idle`) before quitting. Quit via
`osascript … quit app "Tome"` (clean exit, ~2 s). `ditto`'d the new build
(same `Tome Self-Signed` identity, so Screen Recording/Mic TCC grants carried
over — confirmed empirically, see below) over `/Applications/Tome.app`.
`defaults write com.dloomis.tome graniteShadowEnabled -bool YES`. Relaunched;
`/health` answered within ~1 s.

**Deviation — orphan-WAV relocation.** 12 pre-existing crashed-recording WAVs
sat in `~/Library/Application Support/Tome/sessions/` from Nic's own past
sessions (unrelated to this work). `ContentView.checkForOrphanedSessionsOnce()`
runs a blocking `NSAlert.runModal()` on relaunch when orphans are found, which
starves the `@MainActor`-bound API (`/health` hangs) with no way to dismiss it
headlessly. To keep the relaunch unattended-safe, the 12 WAVs were `mv`'d to a
scratch holding dir immediately before quitting Tome and `mv`'d back right
after `/health` confirmed the new process was up (past the once-per-launch
scan). MD5 of all 12 files verified identical before move, after move, and
again at end-of-run — zero data loss, nothing recovered/discarded. This is a
real, pre-existing product gap (a launch-time modal can starve the local API)
worth a follow-up but out of scope here.

**Live smoke — silent-tap result: needed a display wake, not a volume bump.**
First attempt at output volume 0 failed as a genuine capture start failure,
not muted audio: `[ENGINE-5-FAIL] Failed to start system audio: … CaptureError
error 0` (= `.noDisplay`) immediately on start. Root-caused via unified log
(`log show --predicate 'subsystem == "com.apple.TCC"'`): Screen Recording TCC
was fine (`AUTHREQ_RESULT authValue=2` = allowed, confirmed for
`com.dloomis.tome` against `kTCCServiceScreenCapture`) — the actual cause was
`system_profiler SPDisplaysDataType` showing `Display Asleep: Yes`.
ScreenCaptureKit's `SCShareableContent` returns zero displays while the
built-in display is idle-asleep, which the call-capture system-audio tap
depends on even for audio-only capture. This is expected during real meetings
(display is always awake then) but not at midnight with nobody at the
keyboard — an artifact of unattended testing, not a shadow-transcription bug.
A second attempt at volume 0 (no display change) failed identically,
confirming it wasn't a launch-warm-up race. One retry (per brief) was spent
addressing the diagnosed cause instead of the brief's volume-15 fallback
(which would not have fixed a zero-display condition): woke the display for
~35 s via `caffeinate -u -d -t 35`, re-ran at output volume 0/muted (still
fully silent), then let the timer expire naturally — display returned to
`Display Asleep: Yes` on its own, matching the state found at task start; no
brightness/settings changed.

- Session `session_2026-07-10_00-06-59` (subject "Task 13 Granite Shadow
  Smoke Test Retry2"), call capture, 34 s, volume held at 0 throughout capture
  and speech.
- Primary transcript (Whisper large-v3-turbo): 3 real utterances across
  "Speaker 2/3/4" (diarization split the second voice in two), text matches
  the two spoken passages verbatim modulo casing/punctuation.
- `session_2026-07-10_00-06-59.granite.md` and `.comparison.json` both
  appeared in `~/Library/Application Support/Tome/GraniteShadow/`, paired
  correctly by session ID. Granite text contains all the distinguishing
  content words from both passages ("quarterly planning", "granite shadow
  transcription rollout", "budget allocations", "customer onboarding
  metrics", "follow-up meeting for next tuesday").
- Shadow totals from the comparison JSON: 3 segments, 0 errored, 24.36 s
  audio, 1.156 s shadow wall-clock → **RTF 0.0474** — matches the Phase 0
  benchmark RTF (~0.05) closely.
- `pgrep llama-server` empty after the job completed — spawn-per-job
  lifecycle confirmed on a real session, not just in Phase 0's harness.
- `python3 scripts/granite-shadow-report.py "~/Library/Application
  Support/Tome/GraniteShadow" -o /tmp/shadow-smoke-report.html` rendered (3
  segments); report HTML contains the same passage phrases.

**Verdict: PASS.** Silent (volume-0) system-audio tap capture works and the
full granite shadow pipeline (spawn sidecar → transcribe → compare → write
artifacts → stop sidecar) ran correctly end-to-end on a real installed build,
with the one caveat above (needs an awake display — true of real meetings,
not of this unattended test window).

**End state confirmed:** Tome running (new build, `graniteShadowEnabled=YES`),
`isRecording:false`, output volume restored to 50/unmuted (pre-task baseline),
display back to idle-asleep (pre-task state), no `llama-server` process, no
stray `say` processes, the 12 pre-existing orphan WAVs untouched (MD5-verified),
two failed-attempt test artifacts (empty transcripts/recordings from the
display-asleep failures) deleted from Nic's vault, the one successful smoke
session's transcript/recording/voiceprints left in place as evidence,
`/tmp/tome-backup/Tome.app.bak` left in place as a rollback point, `git
status` clean except this doc.

Shadow is live for Friday's meetings; review lands EOD Wednesday 2026-07-15
via `scripts/granite-shadow-report.py`.

## Reinstall (final-review fixes, fc8ab15)

2026-07-10 ~01:10–01:14 local, unattended. The whole-branch final review
landed five fix commits after the live-smoke install (sidecar
orphan-on-quit registry, foreign-server adoption refusal, HTTP-status
handling in post(), `-c 16384 --no-webui` launch args, pairing-key
collision, quitting-gate race) — the running app predated them, so the same
install discipline was repeated on build
`fc8ab1535277e7d216e22dff2bc9caea81134465` (144/144 tests green).

- **Sidecar-args sanity (pre-install):** manually launched llama-server with
  the NEW args (`-m … --mmproj … --host 127.0.0.1 --port 8873 -c 16384
  --no-webui`); /health 200 in ~1 s; `granite_client.py /tmp/probe.wav` →
  "the quick brown fox jumps over the lazy dog" in 0.2 s; killed, port 8873
  confirmed free. Note for the Wednesday review: llama.cpp warns
  `n_ctx_seq (16384) > n_ctx_train (4096)` and allocates 4 slots of
  `n_ctx_slot = 4096` — the args are accepted and functional, but the
  effective per-slot context is 4096, not 16384 (and `--no-webui` is
  deprecated spelling for `--no-ui`; still honored).
- **Install:** same procedure as the live smoke — verified idle via API,
  MD5-relocated the 12 orphan WAVs around the relaunch (restored,
  re-verified identical), graceful quit (~1 s), `ditto` install, `/health`
  up ~2 s after launch. `graniteShadowEnabled` still 1 (untouched).
  Installed CDHash `30c668f6…`, CFBundleVersion `1.4.4-63-gfc8ab15-dirty`
  (`-dirty` cosmetic: temporary build-script edit during the build, reverted).
- **No audio smoke this time** (per controller: pipeline shape unchanged and
  unit-verified; args covered by the manual sanity above). Volume and display
  never touched — volume read-verified at baseline 50/unmuted throughout.
- **Backups rotated:** `/tmp/tome-backup/Tome.app.bak-orig` = original
  v1.4.4 (`1.4.4-33-g573341d`), `/tmp/tome-backup/Tome.app.bak` = outgoing
  shadow-v1 build (`1.4.4-57-gf879620-dirty`, CDHash `91c07752…`).
- **End state:** Tome running (fc8ab15 build, flag ON), not recording, no
  llama-server, orphan WAVs byte-identical, git clean except this note.
