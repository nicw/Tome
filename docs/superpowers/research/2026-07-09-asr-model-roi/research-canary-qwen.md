# Deep-dive: nvidia/canary-qwen-2.5b for Tome

## 1. Model card facts (architecture, size, license, languages)

Source: [HF model card](https://huggingface.co/nvidia/canary-qwen-2.5b) (released to HF 2025-07-17).

- **Architecture**: SALM (Speech-Augmented Language Model) — a **FastConformer encoder + Qwen3-1.7B LLM decoder**, joined by a linear projection (1024→2048), with **LoRA applied to the LLM**. Total **2.5B parameters**.
- **License**: **CC-BY-4.0** — explicitly "ready for commercial use". No license blocker for Tome.
- **Languages**: **English only**. Encoder was pretrained on De/Fr/Es speech but the card says it is "unlikely to be reliable as a multilingual model." (Tome's current Parakeet-TDT v3 covers 25 EU languages; adopting canary-qwen would be an English-only regression for the post-processing path.)
- **Training data**: 234.5k hours across 26 English datasets; majority from Granary (YouTube-Commons 109.5k h, YODAS2 77k h, LibriLight 13.6k h). **AMI is in the training set and was oversampled to ~15% of total training data. Earnings22 and SPGISpeech are NOT in training** ([README](https://huggingface.co/nvidia/canary-qwen-2.5b/raw/main/README.md)).
- **Context limits**: max training audio duration **40 s**, max sequence 1024 tokens. Longer inputs "may technically" work but with degraded accuracy — so meeting-length audio requires VAD/chunking + stitching in the runtime layer (Tome would own that).
- **Punctuation/capitalization**: yes, trained with PnC transcripts.
- **Noise robustness** (from card): WER 9.83% at SNR 0 dB, 30.60% at SNR −5 dB — degrades steeply in noise, which matters for "imperfect audio" meetings.

## 2. Framework dependency: NeMo without CUDA

- Official inference path is **NeMo ≥ 2.5.0** via `nemo.collections.speechlm2.models.SALM` (`SALM.from_pretrained(...)`, `model.generate(...)`) — note this is the **speechlm2** collection, *not* the classic `asr` collection.
- The model card lists only NVIDIA GPU architectures (Ampere/Hopper/Blackwell, tested on A6000/A100/RTX 5090) and Linux/Windows as the runtime environment ([model card](https://huggingface.co/nvidia/canary-qwen-2.5b)).
- NeMo itself has *partial* macOS support: the install docs guarantee **only the ASR collection on MacBook**, and MPS inference requires `PYTORCH_ENABLE_MPS_FALLBACK=1` plus `allow_mps=true` because not all ops are implemented on MPS ([NeMo repo/docs](https://github.com/NVIDIA-NeMo/NeMo)). The speechlm2/SALM collection has **no documented Mac/MPS path**.
- Real-world evidence: in [HF discussion #11](https://huggingface.co/nvidia/canary-qwen-2.5b/discussions/11), a user on Ubuntu could not even get CPU-only inference working (dtype/device errors); NVIDIA staff (Piotr Żelasko) responded only with CUDA-based solutions. **No reports anywhere of canary-qwen running on Apple Silicon via NeMo.**
- Practically moot for Tome anyway: Tome is a Swift/SwiftPM app; embedding a Python NeMo stack is a non-starter. Any integration must go through a native port.

## 3. Ports & Apple Silicon precedent

**Mainstream Apple Silicon ASR runtimes do NOT ship canary-qwen:**

- **FluidAudio** (Tome's current Parakeet vendor): ships parakeet-tdt-0.6b v2/v3 CoreML, parakeet-ctc zh, and (per docs) Qwen3-ASR — **no canary of any kind**; model requests are funneled through [issue #49](https://github.com/FluidInference/FluidAudio/issues/49) ([repo](https://github.com/FluidInference/FluidAudio)).
- **Argmax (WhisperKit Pro / Argmax SDK)**: ships NVIDIA Parakeet v2/v3 on ANE and references **canary-1b-v2** (the encoder-decoder Canary, not canary-qwen) in its pro/commercial SDK, with accuracy testing so far only on Parakeet ([Argmax blog](https://www.argmaxinc.com/blog/nvidia-frontier-speech-models-on-argmax-sdk)).
- **onnx-asr** (CPU/CoreML-EP ONNX runtime pkg): supports "Parakeet v2/v3, Canary v1/v2" — **not canary-qwen** ([PyPI](https://pypi.org/project/onnx-asr/)).
- **sherpa-onnx** and **mlx-audio**: support Canary-1b-family and Qwen3-ASR, **not canary-qwen** ([sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [mlx-audio](https://github.com/Blaizzy/mlx-audio)).

**Fringe canary-qwen-specific ports DO exist (verified directly):**

- **CoreML**: [phequals/canary-qwen-2.5b-coreml-fp16](https://huggingface.co/phequals/canary-qwen-2.5b-coreml-fp16) (+int8/static variants) — a full 4-stage pipeline (encoder.mlpackage, projection.mlpackage, stateful decoder with KV cache targeting macOS 15 CoreML state, LM head). **Artifacts only**: "long-audio chunking, prompt formatting, and transcript stitching live in the runtime layer and are not included." ~35 downloads/month, unofficial, no published accuracy/perf validation. Using it means Tome writes the entire Swift runtime (audio frontend, chunking, autoregressive decode loop across 4 CoreML models, stitching).
- **GGML/GGUF**: [CrispASR](https://github.com/CrispStrobe/CrispASR) (whisper.cpp fork, C++ with C-ABI, `-DGGML_METAL=ON` for Apple Silicon) explicitly lists a `canary-qwen` backend (FastConformer + Qwen3-1.7B SALM) with GGUF weights at [cstr/canary-qwen-2.5b-GGUF](https://huggingface.co/cstr/canary-qwen-2.5b-GGUF). This is the most practical Mac path today — C++ is embeddable from Swift — but it's a hobby-scale v0.8.x project; its own docs cite ~0.3× realtime on M1+Metal for a comparable LLM-decoder ASR model (quant dequant bottleneck). On an M2 Max/M4 Ultra, a 1-hour meeting would plausibly take minutes-to-tens-of-minutes — within Tome's stated tolerance, but unproven.
- **ONNX**: [onnx-community/canary-qwen-2.5b-ONNX](https://huggingface.co/onnx-community/canary-qwen-2.5b-ONNX) exists (13.7 GB) but has no model card, 13 downloads, and the visible file listing suggests LLM/tokenizer files without a clear audio-encoder pipeline — not a usable path.

**Precedent summary**: Canary *encoder-decoder* models (canary-1b-v2) have real Apple Silicon support (Argmax Pro, mlx-audio, onnx-asr, sherpa-onnx). **canary-qwen specifically has zero mainstream support** — only two low-adoption community effort tracks. The SALM structure (CoreML/ggml encoder + autoregressive 1.7B LLM decode per chunk) is inherently harder to port and slower on ANE/Metal than parakeet-style transducers.

## 4. Accuracy profile for meetings

Leaderboard RTFx context: all Open ASR Leaderboard RTFx numbers (canary-qwen 418) are measured on an **NVIDIA A100-SXM4-80GB, CUDA 12.6, batch size up to 64** ([Open ASR Leaderboard paper](https://arxiv.org/html/2510.06961)) — they do not transfer to Apple Silicon.

Per-dataset WER ([model card README](https://huggingface.co/nvidia/canary-qwen-2.5b/raw/main/README.md)):

| Dataset | canary-qwen-2.5b | granite-speech-4.1-2b | Notes |
|---|---|---|---|
| LS Clean | 1.60 | 1.33 | clean read speech |
| LS Other | 3.10 | 2.50 | |
| TEDLIUM | 2.72 | — | prepared talks |
| SPGISpeech | **1.90** | 3.78 | pro-quality earnings-call segments, **not in canary-qwen training** |
| GigaSpeech | 9.41 | — | web/podcast audio |
| **AMI (meetings)** | **10.18** | **8.09** | AMI in BOTH models' training (canary-qwen oversampled it to ~15%) |
| **Earnings22** | **10.42** | **8.37** | accented, telephone/webcast-quality; in granite's training (105 h), NOT in canary-qwen's |

Characterization: canary-qwen **wins on clean, well-mic'd, professionally produced audio** (its SPGISpeech 1.90 is genuinely impressive since SPGI was out-of-training) and **loses precisely on Tome's target conditions** — accented speakers, far-field mics, degraded channels (AMI, Earnings22, GigaSpeech), plus steep noise degradation (9.8% WER at SNR 0 dB, 30.6% at −5 dB). Caveat: granite's Earnings22/AMI edge is partly train-set exposure ([granite-speech-4.1-2b trained on Earnings-22 105 h and AMI 100 h](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)), so the ~2-point gap overstates granite's generalization advantage somewhat — but canary-qwen had AMI oversampled to 15% of its diet and *still* posts 10.18 on AMI, which is a genuinely weak meeting-domain result for a 2.5B model. For meeting transcription, canary-qwen's headline 4.26 avg WER is carried by clean-audio test sets that don't resemble Tome's input.

## 5. ASR+LLM hybrid mode — relevant or gimmick?

The card documents two modes: ASR mode (transcribe) and **LLM mode**, activated by `model.llm.disable_adapter()` — this turns off the ASR LoRA and exposes the **original text-only Qwen3-1.7B**, which "can be used to post-process the transcript, e.g. summarize it or answer questions about it" ([model card](https://huggingface.co/nvidia/canary-qwen-2.5b)). Key facts: LLM mode takes **text in, text out — it cannot attend to the audio**; it is literally a stock 1.7B Qwen bundled in the same checkpoint. For Tome this is a **gimmick**: any summarization Tome wants can be done better by a separate, larger local LLM on Nic's 64 GB M2 Max (or Claude), with zero coupling to the ASR runtime. The hybrid mode's real value is NVIDIA's research story (one deployment does both), not accuracy or capability.

## Net assessment for Tome

- License and quality-on-clean-audio are fine, but canary-qwen is the **wrong accuracy profile** (weakest exactly on meeting-like/accented/noisy audio among the leaderboard's top models), **English-only**, has a **40 s window** requiring Tome-owned chunking, and has **no production-grade Apple Silicon runtime** — only an artifacts-only CoreML conversion and a hobby ggml fork. Integration cost is very high; expected accuracy payoff on meetings vs. current Parakeet-TDT v3 is modest and vs. granite-class models is negative.

## BOTTOM LINE
canary-qwen-2.5b is commercially licensed (CC-BY-4.0) and strong on clean audio, but it is English-only, capped at ~40 s per inference window, and — critically for Tome — its WER is worst exactly on meeting-like conditions (AMI 10.18, Earnings22 10.42 vs granite-4.1-2b's 8.09/8.37), despite AMI making up ~15% of its training data. There is no supported non-CUDA path: NeMo's speechlm2/SALM collection has no Mac/MPS story, and no mainstream Apple Silicon runtime (FluidAudio, Argmax, mlx-audio, sherpa-onnx, onnx-asr) ships canary-qwen — only an artifacts-only community CoreML conversion (~35 downloads/mo) and a hobby ggml/Metal fork (CrispASR) exist. Its LLM mode is just the bundled text-only Qwen3-1.7B post-processing the transcript — a gimmick Tome can beat with any separate local LLM. Recommend against integrating canary-qwen; models with better accented/noisy-meeting accuracy and viable Apple Silicon ports are stronger candidates.

## VERIFICATION VERDICTS

- [CONFIRMED] nvidia/canary-qwen-2.5b is a 2.5B-parameter SALM combining a FastConformer encoder with a Qwen3-1.7B decoder (linear projection + LoRA), released under CC-BY-4.0 with commercial use permitted, English-only, with max training audio duration of 40 seconds.
  NOTE: Raw README (huggingface.co/nvidia/canary-qwen-2.5b/raw/main/README.md) verified directly: '2.5 billion parameters', SALM architecture badge and description 'Speech-Augmented Language Model (SALM) with FastConformer Encoder and Transformer Decoder... built using nvidia/canary-1b-flash and Qwen/Qwen3-1.7B, a linear projection, and low-rank adaptation (LoRA) applied to the LLM'; 'license: cc-by-4.0' and 'This model is ready for commercial use'; 'English-only language support'; 'The maximum audio duration in training was 40s'. Every element matches.

- [CONFIRMED] canary-qwen-2.5b scores 10.18 WER on AMI and 10.42 on Earnings22 (vs 1.60 LS Clean, 1.90 SPGISpeech), even though AMI was oversampled to about 15% of its 234.5k-hour training data while Earnings22 and SPGISpeech were not in training.
  NOTE: README leaderboard table row matches digit-for-digit: AMI 10.18, Earnings22 10.42, LS Clean 1.60, SPGISpeech 1.90 (row: 418 | 5.63 | 10.18 | 9.41 | 1.60 | 3.10 | 10.42 | 1.90 | 2.72 | 5.66). Training section: 'English (234.5k hours)' and 'AMI was oversampled during model training to constitute about 15% of the total data observed'. Earnings22 and SPGISpeech are absent from the training dataset list (18 training datasets; both appear only under evaluation). Minor caveat, not material: the card's YAML model-index widget carries slightly different values (AMI 10.19, Earnings 10.45, LS clean 1.61) than the README table the claim cites; the claim's numbers match the cited table exactly.

- [CONFIRMED] ibm-granite/granite-speech-4.1-2b scores 8.09 WER on AMI and included both AMI (100 h) and Earnings-22 (105 h) in its training data, under an Apache 2.0 license.
  NOTE: Raw README verified: 'license: apache-2.0'; training data table rows 'AMI English | ASR | 100' and 'Earnings-22 English | ASR | 105' (hours). The 8.09 AMI WER is not in the README text (per-dataset WERs are chart images), but the rendered model page's Evaluation results widget contains exactly "task_id":"ami_wer","value":8.09 (verified in page HTML), and the card's own WER bar chart shows granite-speech-4.1-2b at ~8.1 on AMI_IHM, consistent. Mean WER 5.33 / RTFx 231.29 also on the page, corroborated by the card's Open ASR Leaderboard scatter plot.

- [CONFIRMED] The onnx-asr package supports NVIDIA Parakeet v2/v3 and Canary v1/v2 on CPU/CoreML across macOS and Arm, but does not support canary-qwen.
  NOTE: PyPI page (and the identical upstream README at github.com/istupakov/onnx-asr) states: 'Supports Parakeet v2 (En) / v3 (Multilingual), Canary v1/v2 (Multilingual) and GigaAM v2/v3 (Ru) models!' and 'Works on Windows, Linux, and macOS on x86 and Arm CPUs, with support for CUDA, TensorRT, CoreML, DirectML, ROCm, and WebGPU'. Zero occurrences of 'qwen' anywhere in the README — canary-qwen is not in the supported model list, supporting the negative claim.

- [CONFIRMED] A community CoreML conversion of canary-qwen-2.5b exists (phequals/canary-qwen-2.5b-coreml-fp16) containing encoder, projection, stateful decoder, and LM head mlpackages, but it ships model artifacts only — chunking, prompt formatting, and transcript stitching are explicitly not included.
  NOTE: Repo README verified via raw fetch: lists encoder.mlpackage, projection.mlpackage, canary_decoder_stateful.mlpackage ('FP16 stateful autoregressive decoder with KV cache'), and canary_lm_head.mlpackage; states verbatim 'This repo contains model artifacts only.' and 'Long-audio chunking, prompt formatting, and transcript stitching live in the runtime layer and are not included here.' Card also self-describes as a community (not official NVIDIA) CoreML FP16 conversion.

- [CONFIRMED] Open ASR Leaderboard RTFx figures (canary-qwen: 418) are measured on an NVIDIA A100-SXM4-80GB GPU with CUDA 12.6 and batch sizes up to 64, not on Apple Silicon.
  NOTE: arXiv 2510.06961 (Open ASR Leaderboard paper) states: 'The evaluation scripts for each model were run on an NVIDIA A100-SXM4-80GB GPU (driver 560.28.03, CUDA 12.6)' using 'a batch size of 64 whenever memory allowed, and reduced adaptively (48, 32, 16, ...)'. Table 3 lists NVIDIA Canary Qwen 2.5B at RTFx 418 (avg WER 5.63), matching the model card's own 418 RTFx figure. 'Batch sizes up to 64' is an accurate rendering, and the hardware is an NVIDIA datacenter GPU, not Apple Silicon.
