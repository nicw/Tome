# Tome ASR Model Setup — Scalability Review (model N+1 and live/post split)

Reviewed at commit `573341d` on `main`. All paths relative to `/Users/nic/programming/tome`.

## 1. Cost of model N+1 — inventory of every code site

The whisper-v3-turbo work (spec `docs/superpowers/specs/2026-07-08-whisper-v3-turbo-model-option-design.md`) left a clean seam: everything downstream of the enum is model-agnostic. Adding a third **live-capable, in-process** model touches:

### Production code (7 sites, 5 of them compiler-enforced exhaustive switches)

| # | Site | Change |
|---|------|--------|
| 1 | `Tome/Sources/Tome/Transcription/TranscriberModel.swift:3-5` | New enum case + stable raw value (persisted format — pick carefully, it's forever) |
| 2 | `TranscriberModel.swift:7-12` | `displayName` switch arm |
| 3 | `TranscriberModel.swift:15-20` | `pickerSubtitle` switch arm |
| 4 | `TranscriberModel.swift:32-37` | `isInstalled` switch arm → `NewBackend.isInstalled()` |
| 5 | `TranscriberModel.swift:41-47` | `approxDownloadSize` switch arm |
| 6 | `Tome/Sources/Tome/App/AppServices.swift:69-74` | `makeBackend` factory switch arm → `NewBackend()` |
| 7 | **New file** `Tome/Sources/Tome/Transcription/NewBackend.swift` | The real cost: an actor conforming to `ASRBackend` (Parakeet is 63 lines, Whisper 150 — the delta is download-location plumbing, `isInstalled` completeness, and result mapping) |

Sites 1–6 are exhaustive switches with no `default:` — the compiler produces the checklist. That's a deliberate, good property at N=3–5.

### Sites that need NO change (verified)

- `Tome/Sources/Tome/Views/SettingsView.swift:152` — picker iterates `TranscriberModel.allCases`; row subtitles (`:238-243`) derive from `model.isInstalled`/`approxDownloadSize`. Scales automatically.
- `ModelProvisioner.swift` — fully model-agnostic (selection/factory injected as closures).
- `ASRCoordinator.swift`, `StreamingTranscriber.swift`, `SegmentReTranscriber.swift`, `PostProcessingJob/Queue.swift`, `Recovery.swift:114`, `ContentView.swift:142-164` (onChange + boot kick), `TranscriptionEngine.swift:309` (status string uses `activeModel?.displayName`), APIServer `/health`.

### Tests (additive only)

- `Tome/Tests/TomeTests/TranscriberModelTests.swift:6-21` — add raw-value pin, displayName, and `from(persisted:)` lines for the new case.
- A new `NewBackendTests.swift` in the pattern of `WhisperBackendTests.swift` (variant/path resolution, pure functions only).
- `ModelProvisionerTests`/`ASRCoordinatorTests`/`FakeBackend` — untouched; they use the two existing cases as arbitrary distinct labels.

### The one genuinely non-scaling site: ASRBench

`Tome/Sources/ASRBench/main.swift` hand-mirrors backend config because "SwiftPM forbids importing the app executable from here" (`main.swift:14-15`, explicit "keep in sync" comment at `:16-18`). Each backend has its own ~50-line bench function (`benchParakeet` :83-121, `benchWhisper` :124-174), and the top level hardcodes exactly two runs, two reports, and a two-element JSON array (`main.swift:222-235`). Model N+1 = another hand-mirrored config block + bench function + edits at 3 places, with silent-drift risk against the real backend. Fixable by extracting a `TomeASR` library target both the app and ASRBench import (~half a day, mechanical).

**Net cost of a live-capable model N+1: ~1 day of plumbing + the backend itself + ASRBench mirror + manual smoke/bench runs.** The plumbing is not the bottleneck; validating the backend is.

## 2. Is `ASRBackend` adequate for a 2B-LLM-class backend?

The protocol (`Tome/Sources/Tome/Transcription/ASRBackend.swift:20-33`):

```swift
protocol ASRBackend: AnyObject, Sendable {
    var model: TranscriberModel { get }
    static func isInstalled() -> Bool
    func prepare(onEvent:) async throws
    func transcribe(samples:language:) async throws -> ASRResult
    func transcribe(buffer:language:) async throws -> ASRResult
    func unload() async
}
```

### (a) Out-of-process sidecar: mostly adequate

A proxy actor speaking XPC/stdio to a helper process conforms fine: `prepare` = spawn + load (both `PrepareEvent` phases map), `unload` = terminate, `AnyObject` identity gives the coordinator's `ObjectIdentifier`-keyed in-flight/retired tracking (`ASRCoordinator.swift:24-26`) a stable key. Gaps:

- **No health/restart notion.** A crashed sidecar surfaces only as thrown `transcribe` errors. Live: `StreamingTranscriber.swift:124,147` tolerates 10 consecutive errors, then kills the leg with "restart session". Batch: `SegmentReTranscriber.swift:71-78` writes `"[transcription failed]"` per segment and *keeps going* — a dead sidecar produces a transcript of failure placeholders rather than aborting the job. Nothing re-`prepare`s a backend except a user-driven provision/retry cycle or the flip-back re-assert (`ModelProvisioner.swift:130-144`).
- **`static func isInstalled()` is per-type**, so one Swift class per model. An LLM *family* (e.g. 2B vs 8B quantizations of the same runner) forces class-per-variant or breaking the static. (Whisper dodges this with device-resolved variants inside its statics, `WhisperBackend.swift:20-26` — workable once, not a pattern.)
- Protocol couples to FluidAudio's `Language` and `ASRResult` types (`ASRBackend.swift:28-29`); every non-FluidAudio backend hand-constructs `ASRResult` as WhisperBackend already does (`WhisperBackend.swift:104-110`). Acceptable, but the adapter drags FluidAudio into sidecar code.

### (b) Minutes-long batch latency: this is where it actually breaks

- **Backends are actors** ("Conformances are actors: they own mutable SDK handles… that must be serialized" — `ASRBackend.swift:13-14`; `ParakeetBackend.swift:15`, `WhisperBackend.swift:10`). The coordinator itself does NOT serialize (it suspends at the backend await — `ASRCoordinator.swift:9-13`), but the backend actor does. With one shared model, a minutes-long batch segment transcription blocks every live VAD chunk queued behind it on the same actor. Live becomes unusable during post-processing. **A slow accuracy model requires the dual-model split; it cannot ride the current single slot.**
- **No cancellation or timeout inside the batch loop**: `SegmentReTranscriber.run()` (`:43-80`) has zero `Task.isCancelled` checks; `PostProcessingJob` checks only between phases (`:106,138`). Cancellation of a minutes-per-segment job would take up to a full segment to land, and only if the SDK cooperates.
- **No progress**: `PostProcessingJob.progress` (`PostProcessingJob.swift:24`) is declared and never written anywhere. Invisible for a 30-second job; unacceptable for a 20-minute one.
- **Boot coupling**: orphan recovery awaits `modelProvisioner.awaitSettled()` (`ContentView.swift:891,1061`; poll loop `ModelProvisioner.swift:109-113`) — a multi-GB model download at launch blocks recovery for its duration.
- **Picker lock duration**: `SettingsView.swift:229-234` locks model changes while any job runs — correct policy, but with minutes-long jobs the lock window becomes very long, and it's one combined lock (can't change the live model while an accuracy job runs).

### (c) No live/streaming support: not expressible at all

- No capability flag exists on `ASRBackend` or `TranscriberModel`. The picker (`SettingsView.swift:151-165`) is one radio group whose selection drives **both** paths — the spec says so explicitly ("The selected model serves both live streaming transcription and post-processing re-transcription", spec Goals; per-task split is an explicit Non-Goal at spec line 38-39).
- `canStartRecording` (`ModelProvisioner.swift:48-50`) is `activity == .none && servingModel == selection()` — a post-processing-only model, once serving, would **enable** recording and route live audio through it.
- The F2 failure fallback (`ModelProvisioner.swift:210-215`) falls back to `lastGoodModel` with no notion of live-capability.

### How the transcribers acquire the backend (traced)

Neither ever holds a backend — this is the architecture's best asset for a split:

- `StreamingTranscriber` holds `ASRCoordinator` (`StreamingTranscriber.swift:7`), calls `asrCoordinator.transcribe(samples:source:)` per VAD segment (`:173`).
- `SegmentReTranscriber` holds `ASRCoordinator` (`SegmentReTranscriber.swift:8`), calls `transcribe(buffer:source:)` per diarized segment (`:65`), `source` hardcoded `.system`.
- The coordinator resolves `activeBackend` fresh per call (`ASRCoordinator.swift:109-112`) with per-backend in-flight counting.

So a dual-slot coordinator changes **one line at each call site** (name a role). But note: `ASRCoordinator.transcribe` accepts `source: AudioSource` and **never reads it** (`:83-107` — the parameter is dead), and it's the wrong routing axis anyway: batch re-transcription of call captures also passes `.system`, so live-system-audio and batch calls are indistinguishable at the coordinator today. The routing key a split needs is *purpose* (live vs accuracy), not source.

### Dual-model coordinator — full touch surface

- `ASRCoordinator.swift`: `activeBackend`(:15) → `[ASRRole: any ASRBackend]`; `lastInstallToken`(:22) per role; `install(backend:role:token:)`; `transcribe(…, role:)`; `isReady`/`activeModel`(:32-33) per role. `inFlight`/`retired`(:24-26) are already per-backend-identity and generalize — but a new hazard appears when both slots hold the *same* instance ("accuracy = same as live"): retiring it from one slot must not unload it while the other slot still serves it. Genuinely new state; needs its own tests.
- `ModelProvisioner.swift`: everything is single-slot — `servingModel`/`servingBackend`(:37-41), one `generation`(:70), one `lastGoodKey`(:60), the whole F1/F2/F3 ladder(:204-220). Cleanest path: **two provisioner instances** with distinct defaults keys + a "same as live" sentinel for the accuracy slot, rather than one dual-slot machine.
- `AppSettings.swift:30` + new persisted key; `ContentView.swift:142-147,164` onChange/boot kick ×2; `SettingsView` second picker with **per-slot** lock conditions (live model locked while recording; accuracy model locked while jobs/recovery run).
- `TranscriptionEngine.swift:120` (`isReady` gate) and `:309` (status string) → live slot; `Recovery.swift:114` → accuracy slot; APIServer `/health` → live slot.
- Memory: two models steady-state resident — the spec's stated invariant is one (spec Risks: "steady state is one model") and a 2B LLM + Parakeet is the point where that matters.

## 3. Refactor options

### Option 1 — ModelDescriptor registry (~0.5–1 day)
Keep `TranscriberModel` as the persistence-identity enum (raw values are pinned by `TranscriberModelTests.swift:6-10` — do not touch), move `displayName`/`pickerSubtitle`/`isInstalled`/`approxDownloadSize` + the `AppServices` factory into a descriptor struct in a static table. Kills the layering inversion (identity enum importing concrete backends, `TranscriberModel.swift:34-35`) and un-statics `isInstalled` (enables variant families). N+1 becomes: one enum case + one table row + backend file.
**Tests pinned today:** `TranscriberModelTests` (4 tests: raw stability, displayName, persisted fallback, didSet persistence) pass unchanged if the computed properties become lookups. `ModelProvisionerTests` (15 tests), `ASRCoordinatorTests` (7 tests) untouched.
**Verdict:** cheap, marginal payoff at N=3 by itself, but the prerequisite for option 2.

### Option 2 — Capability flags (`supportsLive`, latency class) (~1–2 days, on top of 1)
Descriptor gains `supportsLive: Bool` (+ optional caution copy — spec §8 already anticipated "may lag during live transcription"). Changes: picker annotates/filters (`SettingsView.swift:152`); `canStartRecording` adds `servingModel.supportsLive` (`ModelProvisioner.swift:48-50`); F2 fallback (`:210-215`) skips non-live-capable models; ControlBar/API error copy.
**Test impact:** `TranscriberModelTests` pins the flags; `ModelProvisionerTests` +~3 cases (post-only model selected → recording gated with correct message; fallback skips post-only; retry); `FakeBackend` (`Tome/Tests/TomeTests/FakeBackend.swift:8`) gains a capabilities knob.
**Verdict:** the minimum to *safely* add a post-only model — but in the single-slot world it means "recording disabled while the accuracy model is selected", which is poor UX. It's a guard rail, not the feature.

### Option 3 — Dual-slot coordinator + second provisioner (~3–5 days + state-machine re-audit)
As traced in §2. This is what a 2B accuracy model actually requires (the actor-serialization problem in §2b makes options 1–2 insufficient).
**Test impact:** `ASRCoordinatorTests` — all 7 tests gain a role parameter (mechanical) + new cross-slot tests (install into accuracy slot doesn't disturb live backend; shared-instance retire must not unload a backend the other slot still serves; per-role token ordering). `ModelProvisionerTests` — the existing 15 run unchanged against the live-slot instance if provisioners are per-slot; +~5 interplay tests. The spec's §9 "state-audit agent pass" needs re-running — the F-1/I-1 token races were hand-audited for one slot.

**Recommended sequencing:** Option 1 now (do it as part of model N+1 — same files), Option 2 when the first post-only model is real, Option 3 only with ASRBench-class evidence that a live/accuracy split pays for its complexity.

## 4. Existing tech debt that makes model N+1 riskier

1. **Dead `source` parameter** — `ASRCoordinator.swift:83,96`: accepted, never read, and the wrong axis for the routing a split needs (batch passes `.system` at `SegmentReTranscriber.swift:65`, identical to live system audio). A future author may reasonably assume it routes. Repurpose to a `purpose`/role or delete.
2. **ASRBench hand-mirroring** — `Sources/ASRBench/main.swift:14-18` ("keep in sync") and the hardcoded two-model run/report/JSON at `:222-235`. Every new model doubles down on drift risk in the very tool used to accept it.
3. **Model-specific constants in shared pipeline code** — `StreamingTranscriber.swift:129` drops sub-8000-sample segments with the rationale "Parakeet emits garbage below this threshold", applied to *all* backends; `SegmentReTranscriber.swift:41` pads to 1.5s "to clear Parakeet's 1s minimum", ditto; `StreamingTranscriber.swift:47-48` flush interval tuned "for Parakeet-TDT". A new model inherits Parakeet's tuning silently. These belong on the backend/descriptor.
4. **SDK calls in UI-render paths** — `TranscriberModel.approxDownloadSize` (`TranscriberModel.swift:41-47`) calls `WhisperBackend.resolveVariant()` → `WhisperKit.recommendedModels()`, and `isInstalled` (`:32-37`) stats the filesystem; both run per row per Settings render (`SettingsView.swift:238-243`). Fine at N=2; a pattern that accretes cost per model.
5. **`PostProcessingJob.progress` never written** (`PostProcessingJob.swift:24`) — latent now, blocking for any slow model.
6. **Blocking `awaitSettled` at boot recovery** (`ModelProvisioner.swift:109-113` polled from `ContentView.swift:891,1061`) — orphan recovery waits behind the full model download; the wait scales with model size.
7. **Layering inversion** — `TranscriberModel.swift:34-35` (identity enum → concrete backends) while backends reference the enum back (`ParakeetBackend.swift:16`, `WhisperBackend.swift:11`). Every N+1 touches the cycle; option 1 dissolves it.
8. **Error policy is model-blind** — the consecutive-error threshold of 10 (`StreamingTranscriber.swift:124,147`) and the per-segment `"[transcription failed]"` convention (`SegmentReTranscriber.swift:71-78`) were tuned for fast in-process backends; a sidecar that crashes or a slow model that times out hits them with very different user impact and no differentiated handling.

## What the TDD suite pins (summary)

- **`ASRCoordinatorTests.swift` (7 tests):** not-initialized throw; install/route; immediate unload on idle swap; deferred unload under in-flight call; token-ordered stale-install refusal; higher-token re-assert no-op; token re-validation across unload suspension. These pin exactly the invariants a dual-slot refactor must preserve per slot — the highest-value regression net for option 3.
- **`ModelProvisionerTests.swift` (15 tests):** full F1/F2/F3 ladder, generation guard (late failure/success inert), flip-back cancel + re-assert re-prepare, lastGood semantics, awaitSettled-spans-F2. Untouched by options 1–2; run-as-is against the live slot in option 3.
- **`TranscriberModelTests.swift` (4 tests):** raw-value stability (the on-disk contract), display names, unknown-persisted fallback (rollback compatibility: a future model's raw value reads as Parakeet in an older build), didSet persistence.
- **`WhisperBackendTests.swift` (4 tests):** variant resolution incl. the misnamed-variant trap, HubApi path layout — the template for any new backend's pure-function tests.

## BOTTOM LINE
Adding a third live-capable, in-process model is cheap and well-guarded: 5 compiler-enforced switch arms + one factory arm + a new backend file + additive tests (~1 day of plumbing; ASRBench's hand-mirrored config is the only non-scaling site). But the architecture is single-slot by design at every layer — one activeBackend, one install token, one servingModel/lastGood, one picker driving both live and batch — and backends are actors, so a minutes-long batch call would serialize ahead of live chunks on the same instance. A post-processing-only 2B-class model is therefore not expressible today: it needs capability flags at minimum (to stop canStartRecording from enabling live on a batch-only model) and realistically the dual-slot coordinator (~3–5 days + re-audit). The good news is the seam is right where it needs to be: neither StreamingTranscriber nor SegmentReTranscriber ever holds a backend — both route every call through ASRCoordinator — so the split is a coordinator/provisioner change, not a pipeline rewrite, and the existing 22 coordinator+provisioner tests pin exactly the invariants the refactor must preserve.

## VERIFICATION VERDICTS

- [CONFIRMED] Adding model N+1 requires exactly six production edits (five in TranscriberModel.swift, one makeBackend switch in AppServices.swift) plus one new backend file; the Settings picker iterates allCases and needs no change.
  NOTE: All six sites verified: case declarations at Tome/Sources/Tome/Transcription/TranscriberModel.swift:4-5, displayName switch :8-11, pickerSubtitle :16-19, isInstalled :33-36, approxDownloadSize :42-46, and the makeBackend factory switch at Tome/Sources/Tome/App/AppServices.swift:70-73. These are the only exhaustive switches over TranscriberModel in Sources (grep-verified); from(persisted:) at TranscriberModel.swift:24-26 has a nil-coalescing default and needs no edit; ModelProvisioner and AppSettings are fully generic. SettingsView.swift:152 uses ForEach(TranscriberModel.allCases) with generic rowSubtitle(for:) at :238-243 — no change needed. Trivial imprecision only: 'cases :3-5' is really lines 4-5 (line 3 is the enum declaration), and one of the 'five switch arms' is the case declaration itself, not a switch. Tests and ASRBench would also want updates, but neither breaks compilation and the claim scoped itself to production edits.

- [CONFIRMED] ASRCoordinator.transcribe accepts source: AudioSource but never reads it in either overload; SegmentReTranscriber passes .system for all batch calls, so live system-audio and post-processing calls are indistinguishable at the coordinator.
  NOTE: Both overloads at Tome/Sources/Tome/Transcription/ASRCoordinator.swift:83-94 and :96-107 declare `source: AudioSource` and never reference it in their bodies. SegmentReTranscriber.swift:65 passes `source: .system`. Live system-audio streaming also arrives as .system (TranscriptionEngine.swift:281 constructs its StreamingTranscriber with audioSource: .system; StreamingTranscriber.swift:173 forwards it), so even if the coordinator read the parameter, live-system and batch would collide on the same value. Minor nuance not affecting the verdict: the two call classes are incidentally distinguishable by overload today — live paths use the samples overload, batch uses the buffer overload — but that is an accident of plumbing, not a semantic routing key, and WhisperBackend.swift:113-114 immediately collapses the buffer overload into the samples one.

- [CONFIRMED] Every layer is single-slot (one activeBackend/lastInstallToken in ASRCoordinator, one servingModel/servingBackend and one lastGood key in ModelProvisioner), and canStartRecording checks only activity/selection, so a post-processing-only model once serving would enable live recording — no supportsLive capability exists.
  NOTE: All cites verified: ASRCoordinator.swift:15 (single activeBackend), :22 (single lastInstallToken); ModelProvisioner.swift:37 (servingModel), :41 (servingBackend), :60 (static lastGoodKey). canStartRecording at ModelProvisioner.swift:48-50 is actually `activity == .none && servingModel != nil && servingModel == selection()` — the claim omits the nil check, but it is logically redundant (Optional == non-Optional is never true for nil), so not material. Grep for supportsLive/capability across Sources finds nothing; ASRBackend (ASRBackend.swift:20-33) exposes only model/isInstalled/prepare/transcribe/unload. All recording gates — ControlBar.swift:149,174 record buttons, ContentView.swift:546 start guard, APIServer.swift:355,425 — consume canStartRecording with no per-model capability check, so any serving model enables live recording.

- [REFUTED] Backend conformances are actors that serialize their own transcribe calls, so a minutes-long batch segment on a shared model blocks all queued live VAD chunks; the coordinator itself does not serialize, meaning a slow accuracy model requires a dual-slot split rather than tuning.
  NOTE: The cited facts exist (final actor at ParakeetBackend.swift:15 and WhisperBackend.swift:10; 'must be serialized' comment at ASRBackend.swift:13-14; coordinator suspends at the backend await per ASRCoordinator.swift:9-13) but the mechanism is wrong: Swift actors are reentrant, and the very same doc comment says so (ASRBackend.swift:18-19: 'Swift actors are reentrant — a swap can land while a transcribe is suspended mid-call'). Both backends suspend at their SDK call (ParakeetBackend.swift:50 `await asrManager.transcribe`, WhisperBackend.swift:91 `await whisperKit.transcribe`), so a queued live chunk enters the backend actor and proceeds — it is not blocked behind a long batch call at the actor. WhisperKit is an `open class` (.build/checkouts/argmax-oss-swift/Sources/WhisperKit/Core/WhisperKit.swift:11), so nothing in app code serializes concurrent Whisper inferences at all; AsrManager is an actor (.build/checkouts/FluidAudio/.../AsrManager.swift:6) but also reentrant with internal awaits (e.g. per-window decodeWithTimings at :284,300,327), so calls interleave rather than queue whole-call FIFO. The actor comment refers to serializing access to the mutable SDK handle, not whole transcribe calls. The directionally-right residue — coordinator doesn't serialize, single shared slot, live latency degrades under batch load via compute contention (and possibly unsafe concurrent WhisperKit use) — does not rescue the stated head-of-line-blocking mechanism, which is the load-bearing evidence for 'requires a dual-slot split rather than tuning'.

- [CONFIRMED] ASRBench duplicates WhisperBackend's variant/path config by hand with an explicit 'keep in sync' comment because SwiftPM can't import the app executable (main.swift:14-18), and hardcodes a two-model run, report, and JSON array (main.swift:222-235) — the only code site that scales linearly-with-drift-risk per added model.
  NOTE: Tome/Sources/ASRBench/main.swift:14-15 has the exact comment ('Mirrors WhisperBackend ... keep in sync; SwiftPM forbids importing the app executable from here'); the duplicated variant config is at :16-18 and the duplicated download-base path at :19-20 (path config sits two lines past the cited range — immaterial). The two-model run is hardcoded at :223-226 (benchParakeet/benchWhisper), the Whisper-specific acceptance print at :228-230, and the literal `[parakeet, whisper]` JSON array at :235 (write spans :232-236). It also duplicates StreamingTranscriber's 480k/8k caps at :24-26. 'Only code site with drift risk' holds for Sources: the other per-model sites (TranscriberModel switches, makeBackend) are compiler-enforced exhaustive switches, so they can't silently drift; ASRBench's string/path copies can. Tests hardcode the variant strings too (WhisperBackendTests.swift:7-24) but exercise the app's own code, so they don't drift silently either.

- [CONFIRMED] Model-specific tuning is baked into shared pipeline code (StreamingTranscriber's sub-8000-sample drop citing Parakeet at :129, SegmentReTranscriber's 1.5s pad for Parakeet's 1s minimum at :41) that every new backend silently inherits; and PostProcessingJob.progress is declared but never written (:24), leaving long batch jobs with zero visible progress.
  NOTE: StreamingTranscriber.swift:129 logs exactly '(<8000 ≈ 0.5s, Parakeet emits garbage below this threshold)' for the drop decided at :118 (and again at :161-164 for end-of-stream remnants); SegmentReTranscriber.swift:41 is `let minSamples = Int(sampleRate * 1.5) // 1.5s to clear Parakeet's 1s minimum after resampling` with padding applied at :49-56. Both run upstream of the backend-agnostic ASRCoordinator, so any backend inherits them — e.g. Whisper (no such minimums) still gets Parakeet-tuned drops/padding. PostProcessingJob is at Tome/Sources/Tome/Transcription/PostProcessingJob.swift (not a PostProcessing/ directory); `private(set) var progress: Double = 0` at :24 is never assigned anywhere in the codebase — in fact it is never read either (fully dead). During a batch job the UI shows only the static string 'Finalizing…' (ContentView.swift:414-415), so 'zero visible progress' for a slow model is accurate; the phase enum (:12-20) transitions but no view renders per-phase or percentage detail.
