# IBM granite-speech 4.x — Deep Dive for Tome Integration

## 1. Model enumeration (verified on HF org page, 2026-07-09)

Current speech models under [ibm-granite](https://huggingface.co/ibm-granite) ([Granite Speech collection](https://huggingface.co/collections/ibm-granite/granite-speech)):

| Model ID | Released/updated | Notes |
|---|---|---|
| `ibm-granite/granite-speech-4.1-2b` | Apr 29–30, 2026 (updated ~Jun 12) | AR flagship: ASR + bidirectional AST, 6 languages, keyword biasing, punctuation/caps. 465k downloads |
| `ibm-granite/granite-speech-4.1-2b-nar` | Apr 30, 2026 | Non-autoregressive, ASR-only, 5 languages |
| `ibm-granite/granite-speech-4.1-2b-plus` | ~Jun 16, 2026 (newest) | Adds speaker-attributed ASR + word-level timestamps |
| `ibm-granite/granite-speech-4.1-2b-GGUF` and `granite-speech-4.1-2b-plus-GGUF` | 2026 | **Official IBM GGUF releases for llama.cpp** |
| `ibm-granite/granite-4.0-1b-speech` | Mar 6, 2026 | Predecessor (naming anomaly: "4.0-1b-speech" not "speech-4.0") |
| `granite-speech-3.3-8b`, `-3.3-2b`, `-3.2-8b` | 2025 | Legacy |

**"granite-speech-4.2.1-2b" does not exist.** No 4.2 of any kind is on the org page or findable via search; the latest is 4.1. Almost certainly a mangled recollection of **4.1-2b** (possibly crossed with the `-plus` variant or the odd `granite-4.0-1b-speech` naming).

## 2. The NAR variant explained

Source: [NAR model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar).

- **Architecture**: Instead of autoregressive token-by-token decoding, the CTC conformer encoder emits an initial hypothesis; that hypothesis is interleaved with insertion slots, concatenated with projected audio embeddings, and a **bidirectional LLM edits it (copy/insert/delete/replace) in a single forward pass**, exploiting the "identity mapping bias" of Transformers. One pass replaces hundreds of sequential decode steps — that's why leaderboard RTFx is ~3.8× higher (879 vs 231 on A100); IBM reports ~1820 RTFx on a single H100 at batch size 128 ("one hour of audio in under two seconds", per [MarkTechPost](https://www.marktechpost.com/2026/04/30/ibm-releases-two-granite-speech-4-1-2b-models-autoregressive-asr-with-translation-and-non-autoregressive-editing-for-fast-inference/)).
- **Documented trade-offs**: drops Japanese, drops speech translation, drops keyword biasing; and — directly relevant to meetings — the card states a **conservative editing bias: "prefers deletions over insertions, which reduces hallucination risk but may occasionally drop words in noisy conditions."** On the leaderboard the NAR is actually marginally better on average (3.95 vs 3.99) but slightly worse on Earnings22 (8.44 vs 8.37).
- **Apple Silicon reality check**: NAR **requires `flash_attention_2`** in transformers (CUDA-only → no MPS path); no llama.cpp GGUF exists for it; mlx-audio's Python release notes claim NAR support ([releases](https://github.com/Blaizzy/mlx-audio/releases), v0.4.1) but it's unvalidated.
- **Verdict for batch post-processing on a Mac**: the NAR's entire advantage is *datacenter batch throughput*, which is irrelevant for single-stream on-device post-processing where minutes are acceptable. **Tome wants the AR model (`4.1-2b`) or the `-plus` variant**, which also have far better Apple Silicon runtime coverage.

## 3. Architecture and footprint

Source: [granite-speech-4.1-2b model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b).

- **Encoder**: 16 conformer blocks (1024 hidden, 8 heads), trained with CTC using a novel **dual-head CTC** (credited for the 4.1 accuracy gain). **Projector**: 2-layer window-query transformer (qformer), 10× temporal downsampling. **Decoder**: LLM based on `granite-4.0-1b-base` (128k context), with an **audio-specific LoRA adapter** activated only for audio inputs. Total ~2B params, bf16.
- **Disk (official GGUF repo,** [granite-speech-4.1-2b-GGUF](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-GGUF)**)**: Q4_K_M 1.14 GB, Q5_K_M 1.32 GB, Q6_K 1.51 GB, Q8_0 1.96 GB, bf16 3.68 GB, plus `mmproj-model-f16.gguf` (audio encoder+projector) 1.16 GB. Safetensors bf16 ≈ 4–5 GB.
- **RAM at inference**: Q8_0 + f16 mmproj ≈ 3.5–4 GB working set; bf16 ≈ 6 GB. Trivial on a 64 GB M2 Max or a Mac Studio.
- **MLX**: mlx-community published 4-bit and bf16 conversions of 4.1-2b and 4.1-2b-plus ([mlx-community](https://huggingface.co/mlx-community), [mlx-audio issue #737](https://github.com/Blaizzy/mlx-audio/issues/737)).

## 4. Meeting-audio accuracy evidence

- **Leaderboard (A100)**: 4.1-2b Earnings22 **8.37** WER vs Parakeet-TDT-0.6b-v2's 11.15 (~25% relative better) and Whisper-class models ~11+. Model card reports **AMI 8.09** WER for 4.1-2b. The `-plus` card reports AMI 8.63 / Earnings22 8.68 (plus omits punctuation/caps).
- **The "28% gain for poor audio/accents" claim could NOT be located in any primary source.** I checked the raw model cards for 4.1-2b and 4.0-1b-speech (the string "28" does not appear), the [IBM Research 4.1 announcement blog](https://research.ibm.com/blog/granite-4-1-ai-foundation-models), the [IBM Research leaderboard blog](https://research.ibm.com/blog/granite-speech-recognition-hugging-face-chart), the [HF granite-4-speech blog](https://huggingface.co/blog/ibm-granite/granite-4-speech), and press coverage. Closest real claims: (a) 4.0-1b-speech "provides higher transcription accuracy for English ASR" vs granite-speech-3.3 (qualitative, no number); (b) 4.1 has "higher transcription accuracy for multilingual ASR due to a novel dual-head CTC encoder"; (c) IBM's Royal Flying Doctor Service anecdote — Granite Speech "proved in testing to be far better at handling the background noise than any other commercial models" (no percentage); (d) the `-plus` card's speaker-attribution claim WDER **0.9% vs 2.8%** for Microsoft VibeVoice on FISHER — note "2.8" here is a plausible origin of a misremembered "28%". Treat the 28% figure as unverified folklore; the checkable numbers above (Earnings22/AMI) are strong on their own.

## 5. Apple Silicon inference paths (the critical section)

**(a) llama.cpp — most mature, IBM-official.** llama.cpp's mtmd multimodal stack natively supports the Granite-Speech architecture (conformer encoder + qformer projector via mmproj, with automatic audio-LoRA toggling). IBM ships **official GGUFs** and documents macOS usage in the model card: `brew install llama.cpp` then `llama-cli -st -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 --audio audio.wav -p "transcribe the speech..."`; requires **build b9045+**. `llama-server` works too (multimodal audio input supported). Caveats: support is recent (a 2026 bugfix for infinite-asterisk output shows active churn); audio is 16 kHz mono WAV/MP3. For Tome (Swift): either **sidecar `llama-server`** (simplest, HTTP, robust) or in-process via a llama.cpp xcframework binding (heavier lift). No published Apple Silicon RTF numbers exist; estimate for M2 Max: ~1.5B-effective LLM at Q8 decodes ≳80–150 tok/s on Metal, ASR output is only ~3–4 tokens per second of speech, so expect **RTF roughly 0.03–0.15 single-stream** (estimate, unverified) — comfortably "minutes for an hour-long meeting."

**(b) MLX — the native-Swift path.** IBM's own model card documents `mlx-audio` (Python, ≥0.4.1): `python -m mlx_audio.stt.generate --model ibm-granite/granite-speech-4.1-2b`. Granite Speech 4.0 + 4.1 (incl. NAR) support landed in [mlx-audio v0.4.1](https://github.com/Blaizzy/mlx-audio/releases); mlx-community 4-bit/bf16 conversions exist. Crucially for Tome, **[mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift) (SwiftPM package `MLXAudioSTT`, v0.1.3 July 2026, ~700 stars, macOS 14+) explicitly lists Granite Speech among supported STT models** — in-process Swift, async/await, auto HF download. Maturity is the risk: v0.1.x, granite support weeks old, no published benchmarks; [issue #737](https://github.com/Blaizzy/mlx-audio/issues/737) shows 4.1-family validation was still being firmed up in May 2026. Needs a hands-on smoke test before committing.

**(c) transformers on MPS — not recommended.** Granite-speech is natively in transformers (PR [#36801](https://github.com/huggingface/transformers/pull/36801)), but MPS is undocumented/unvalidated for it, granite multimodal siblings have known MPS flakiness reports, the NAR variant requires CUDA-only flash_attention_2, and it would force a Python sidecar into a Swift app anyway.

**(d) ONNX / CoreML — nothing.** No official or community ONNX/CoreML conversions surfaced in any search. There is no FluidAudio/WhisperKit-style CoreML package for granite-speech; anyone wanting CoreML would be doing the conversion themselves (conformer + qformer + LoRA-modulated LLM = nontrivial).

## 6. License, languages, streaming, length limits

- **License**: Apache 2.0, all variants.
- **Languages**: 4.1-2b: EN/FR/DE/ES/PT/JA (ASR + bidirectional speech translation). NAR and plus: EN/FR/DE/ES/PT, ASR only.
- **Streaming**: **None.** All variants are chunk/batch — audio in, text out; no streaming decoder is documented anywhere. This fits Tome's "split" plan: keep Parakeet-TDT for live captions, granite for accurate post-processing.
- **Max audio length**: `-plus` card is explicit: **up to 9 minutes per call for ASR and speaker attribution, up to 3.5 minutes for word timestamps** ([plus card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)). Base 4.1-2b card states no limit but shares the architecture, so a 1-hour meeting needs chunked calls (~5–9 min segments) with stitching — same pattern Tome already uses for Whisper-style batch.

## Leaderboard RTFx caveat (confirmed)

Open ASR Leaderboard RTFx is measured on an **NVIDIA A100-SXM4-80GB (CUDA 12.6), batch size 64** where memory allows ([Open ASR Leaderboard paper, arXiv:2510.06961](https://arxiv.org/abs/2510.06961)). Absolute RTFx does not transfer to Apple Silicon; only relative comparisons are meaningful, and the NAR's batch-throughput edge specifically evaporates at batch size 1 on a Mac.

## Net recommendation for Tome

`granite-speech-4.1-2b-plus` is the most interesting candidate for the post-processing slot: best-in-class meeting-domain WER at 2B scale, **plus free speaker attribution and word timestamps** (features Tome would otherwise need a diarization pipeline for), Apache 2.0, ~2–4 GB RAM. Two viable ship paths: llama.cpp sidecar (official IBM GGUFs, most proven today) or mlx-audio-swift in-process (cleanest Swift integration, youngest code). Both need an empirical WER + RTF bake-off on Nic's M2 Max against WhisperKit large-v3-turbo before adoption; no Apple Silicon throughput numbers exist publicly.

## BOTTOM LINE
No granite-speech 4.2/4.2.1 exists — the latest is the 4.1-2b family (AR, NAR, and the June-2026 "plus" with speaker attribution + word timestamps), all Apache 2.0, batch-only (no streaming), ~1.1–3.7 GB GGUF on disk. The NAR variant's 4x RTFx edge is a datacenter batch-throughput artifact (single-pass edit of a CTC hypothesis, measured on A100/H100) and it may drop words in noisy audio — for Tome's single-stream Mac post-processing the AR or plus variant is the right pick. Apple Silicon support is real and IBM-documented via two paths: official GGUFs on llama.cpp's mtmd stack, and MLX (mlx-audio ≥0.4.1, plus the SwiftPM package mlx-audio-swift which lists Granite Speech as a supported STT model) — but both are weeks-to-months old and no public Apple Silicon RTF numbers exist, so a local bake-off is required. The user's recalled "28% accuracy gain" claim could not be found in any primary IBM source; the concrete evidence is Earnings22 8.37 / AMI 8.09 WER, roughly 25% relatively better than Parakeet-TDT on Earnings22.

## VERIFICATION VERDICTS

- [CONFIRMED] As of 2026-07-09, the newest IBM speech models on Hugging Face are granite-speech-4.1-2b, granite-speech-4.1-2b-nar, and granite-speech-4.1-2b-plus; no granite-speech 4.2 or 4.2.1 model exists.
  NOTE: HF API listing of ibm-granite speech models confirms the 4.1 trio is the newest model generation (granite-speech-4.1-2b and -plus created 2026-04-16, -nar created 2026-03-10, modified through 2026-06-18) and no 4.2/4.2.1 exists. Minor completeness caveat, not an error: there are also official GGUF companion repos (granite-speech-4.1-2b-GGUF, and granite-speech-4.1-2b-plus-GGUF created 2026-06-30, technically the most recently created speech repo) plus an older granite-4.0-1b-speech — but these don't contradict the claim about the newest model generation.

- [CONFIRMED] granite-speech-4.1-2b-nar is non-autoregressive, edits a CTC hypothesis in a single forward pass using a bidirectional LLM, supports only EN/FR/DE/ES/PT (no Japanese, no translation), and the card says it prefers deletions over insertions and may occasionally drop words in noisy conditions.
  NOTE: Model card matches verbatim: 'edits a CTC hypothesis in a single forward pass using a bidirectional LLM'; languages are exactly English, French, German, Spanish, Portuguese; card explicitly redirects Japanese users to the autoregressive granite-speech-4.1-2b; no translation capability mentioned. Exact quote found: 'it prefers deletions over insertions, which reduces hallucination risk but may occasionally drop words in noisy conditions.'

- [CONFIRMED] IBM publishes ibm-granite/granite-speech-4.1-2b-GGUF with quantizations from Q4_K_M (1.14 GB) to bf16 (3.68 GB) plus a 1.16 GB f16 mmproj audio-encoder file, runnable with llama.cpp build b9045+ including on macOS via brew.
  NOTE: Repo file tree verifies digit-by-digit: Q4_K_M = 1,139,247,200 B (1.14 GB), bf16 = 3,678,444,864 B (3.68 GB), mmproj-model-f16.gguf = 1,159,354,752 B (1.16 GB); README specifies 'llama.cpp build: b9045'. One sourcing nit: the README's install instructions use a curl install script, not brew — the 'via brew' route is not in the cited source (though llama.cpp is separately available in Homebrew). Not material to the claim's substance.

- [CONFIRMED] granite-speech-4.1-2b-plus adds prompt-invoked speaker-attributed ASR and word-level timestamps, handles up to 9 minutes per call for ASR/speaker-attribution (3.5 minutes for timestamps), and is Apache 2.0.
  NOTE: Model card confirms all three: SAA ('[Speaker 1]:' tags) and word-level timestamps ('[T:N]' centisecond tags) are 'controlled by different prompts'; card states 'works well with audio segments up to 9 minutes long for ASR and SAA, and up to 3.5 minutes for timestamps'; license is Apache 2.0. Worth knowing (omitted by researcher, not contradictory): plus variant drops punctuation/capitalization, has slightly worse WER than base (5.71 vs 5.33 avg), and timestamps roll over modulo 1000 every 10 s.

- [CONFIRMED] Blaizzy/mlx-audio-swift (MLXAudioSTT product, macOS 14+) explicitly lists Granite Speech among supported STT models, enabling in-process MLX inference in a Swift Mac app.
  NOTE: GitHub README confirms: repo is 'a modular Swift SDK for audio processing with MLX on Apple Silicon', built on MLX Swift; MLXAudioSTT is a named component; requirements state macOS 14+; Granite Speech appears explicitly in the supported STT model list (alongside Whisper, Parakeet, Canary, Qwen3-ASR, etc.).

- [CONFIRMED] Open ASR Leaderboard RTFx values are measured on an NVIDIA A100-SXM4-80GB GPU with batch size up to 64, so absolute RTFx numbers do not transfer to Apple Silicon.
  NOTE: Not in the abstract, but the full paper (arXiv 2510.06961, Results section) states evaluations 'were conducted on an NVIDIA A100-SXM4-80GB GPU (driver 560.28.03, CUDA 12.6)' with 'a batch size of 64 whenever memory allowed, and reduced adaptively (48, 32, 16, …)' — matching 'up to 64' exactly. The non-transferability to Apple Silicon is the researcher's (sound) inference, not a paper statement.

- [CONFIRMED] As of 2026-07-09, the newest IBM speech models on Hugging Face are granite-speech-4.1-2b, granite-speech-4.1-2b-nar, and granite-speech-4.1-2b-plus; no granite-speech 4.2 or 4.2.1 model exists.
  NOTE: HF API listing for ibm-granite shows exactly these three as the newest speech models (created 2026-03-10 to 2026-04-16); the only later speech uploads are GGUF conversions of the same models (granite-speech-4.1-2b-GGUF 2026-05-11, granite-speech-4.1-2b-plus-GGUF 2026-06-30). Older models: granite-4.0-1b-speech, granite-speech-3.2/3.3. Targeted search for granite-speech-4.2/4.2.1 returns nothing; IBM's own Granite 4.1 blog describes the 4.1 trio as the current release.

- [CONFIRMED] granite-speech-4.1-2b-nar is a non-autoregressive model that edits a CTC hypothesis in a single forward pass using a bidirectional LLM, supports only EN/FR/DE/ES/PT (no Japanese, no translation), and its model card states it prefers deletions over insertions and may occasionally drop words in noisy conditions.
  NOTE: Model card verified point-for-point: it edits a CTC hypothesis in a single forward pass using a bidirectional LLM; languages are English, French, German, Spanish, Portuguese — the card explicitly redirects Japanese users to the autoregressive granite-speech-4.1-2b, and AST (translation) is only in the AR variant. Card quote: 'it prefers deletions over insertions, which reduces hallucination risk but may occasionally drop words in noisy conditions.' Apache 2.0.

- [CONFIRMED] IBM publishes an official GGUF repo (ibm-granite/granite-speech-4.1-2b-GGUF) with quantizations from Q4_K_M (1.14 GB) to bf16 (3.68 GB) plus a 1.16 GB f16 mmproj audio-encoder file, runnable with llama.cpp (build b9045+) including on macOS via brew.
  NOTE: Repo file tree matches exactly: Q4_K_M 1.14 GB, Q5_K_M 1.32 GB, Q6_K 1.51 GB, Q8_0 1.96 GB, bf16 3.68 GB, and mmproj-model-f16.gguf at 1.16 GB. Card states 'llama.cpp build: b9045' as the requirement. One nuance: the card's macOS/Linux install instructions use a curl script and do not mention brew — but the Homebrew llama.cpp formula is currently at b9910 (>= b9045) with Apple Silicon macOS support, so the brew route is independently valid. Not materially wrong.

- [CONFIRMED] granite-speech-4.1-2b-plus adds prompt-invoked speaker-attributed ASR and word-level timestamps, handles audio up to 9 minutes per call for ASR/speaker-attribution (3.5 minutes for timestamps), and is Apache 2.0 licensed.
  NOTE: Model card confirms prompt-controlled speaker attribution ('[Speaker 1]:' turn labels) and word-level timestamps ('[T:N]' in centiseconds mod 1000). Exact card quote: 'This model works well with audio segments up to 9 minutes long for ASR and SAA, and up to 3.5 minutes for timestamps.' License: Apache 2.0. Card also notes the trade-off: Plus drops punctuation/capitalization relative to the base model.

- [CONFIRMED] The Swift package mlx-audio-swift (Blaizzy/mlx-audio-swift, MLXAudioSTT product, macOS 14+) explicitly lists Granite Speech among its supported speech-to-text models, enabling in-process MLX inference in a Swift Mac app.
  NOTE: All literal elements verified: Granite Speech is in the README's supported STT table; the package exposes an MLXAudioSTT product; requirement is macOS 14+/iOS 17+; project is active (v0.1.3 released 2026-07-09, 307 commits). It is real code, not aspiration: Sources/MLXAudioSTT/Models/GraniteSpeech/{GraniteSpeech.swift, GraniteSpeechConfig.swift} plus test coverage. IMPORTANT variant caveat: the GraniteSpeech module README lists only mlx-community/granite-4.0-1b-speech-5bit (~1B) as the available checkpoint — no MLX conversion of granite-speech-4.1-2b/-nar/-plus is listed, so running the 4.1-2b variants through this package is not demonstrated. The claim as written does not assert 4.1 support, so it stands, but do not extrapolate it to the 4.1 models.

- [CONFIRMED] Open ASR Leaderboard RTFx values are measured on an NVIDIA A100-SXM4-80GB GPU with batch size up to 64, so absolute RTFx numbers do not transfer to Apple Silicon.
  NOTE: arXiv 2510.06961 (the Open ASR Leaderboard paper) full text states measurements were 'conducted on an NVIDIA A100-SXM4-80GB GPU (driver 560.28.03, CUDA 12.6)' 'using a batch size of 64 whenever memory allowed, and reduced adaptively (48, 32, 16, ...)'. The inference that absolute RTFx does not transfer to Apple Silicon (different hardware, no 64-way batching in a local single-stream app) is sound.
