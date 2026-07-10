# ASR model ROI research — 2026-07-09

Six research tracks (five web, one repo review), each report followed by its
adversarial verification verdicts (48 total; refuted claims carry the
correction in the verdict note). Produced for the granite shadow
transcription decision — see
[the design spec](../../specs/2026-07-09-granite-shadow-transcription-design.md).

**Recommendation: granite-speech-4.1-2b** (AR base variant), llama.cpp
sidecar on IBM's official GGUFs, validated first via a week of hidden-flag
shadow transcription against the primary model on real meetings.

| File | Track | Net result |
|---|---|---|
| [research-granite.md](research-granite.md) | ibm-granite/granite-speech 4.x deep-dive | Winner. No 4.2 exists; NAR is a datacenter-batch artifact; `-plus` adds speaker tags but drops punctuation. Official GGUFs + merged llama.cpp support. |
| [research-accuracy.md](research-accuracy.md) | Leaderboard deep-dive, meeting-centric deltas | Granite ≈ halves whisper-large-v3-turbo's AMI WER (−49%), −26% vs parakeet-v3. Screenshot's average had AMI toggled off. "28%" figure reconstructed (avg vs whisper-large-v3 at launch, not an accents metric). Caveats: in-domain training splits, no long-form track entry. |
| [research-runtime.md](research-runtime.md) | Apple Silicon runtime landscape | llama.cpp mtmd (official IBM GGUFs) = most credible path; mlx-audio-swift lists Granite but only demonstrates the 1B checkpoint; FluidAudio/WhisperKit will not deliver LLM-decoder models. Leaderboard RTFx = batched H200/A100 throughput; ÷15–25 for 2B MLX-class on M2 Max. |
| [research-higgs.md](research-higgs.md) | bosonai/higgs-audio-v3-8b-stt-v2 | Rejected: rank is a clean-speech artifact; loses to granite on AMI/Earnings22; 17.8 GB; no viable Mac path; repetition-loop mitigations shipped by vendor. |
| [research-canary-qwen.md](research-canary-qwen.md) | nvidia/canary-qwen-2.5b | Rejected: worst meeting profile of the top set despite AMI oversampled to 15% of training; NeMo/CUDA-only; no port. |
| [research-repo.md](research-repo.md) | Tome model-setup scalability review | Model N+1 ≈ 1 day (compiler-enforced switches); post-processing-only model not expressible today (single-slot everywhere, no supportsLive); dual-slot is a coordinator/provisioner change, not a pipeline rewrite. NOTE: one claim refuted — backend actors do NOT head-of-line block (they're reentrant, suspending at SDK calls); the dual-slot case rests on capability/contention/UX grounds instead. |

Re-check candidates if the shadow week disappoints: granite-speech-4.1-2b-plus
(speaker attribution), bosonai/higgs-audio-v3-stt (2.68B sibling — best
overall average, AMI 7.19, fringe ggml ports), and whatever mlx-audio-swift
demonstrates for 4.1-2b by then.
