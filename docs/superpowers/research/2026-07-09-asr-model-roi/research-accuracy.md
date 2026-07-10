# Accuracy ROI for Tome meeting transcription — Open ASR Leaderboard deep-dive (data pulled 2026-07-09)

## 0. Data provenance & the screenshot discrepancy

The leaderboard space (https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) loads its English short-form table from `english_short_latest.csv` in the dataset repo **hf-audio/open-asr-leaderboard-results** (long-form from `Steveeeeeeen/leaderboard_longform`; confirmed by reading the space's `init.py`). I downloaded both CSVs today.

Two things about the user's screenshot:
- **The screenshot's "Average WER" is a 4-dataset average, not the leaderboard default.** Mean of its visible columns reproduces it exactly: granite-nar (8.44+1.28+2.77+3.33)/4 = 3.955 → "3.95"; granite-4.1-2b → 3.995 → "3.99"; higgs → 4.0225 → "4.02". AMI, Gigaspeech and VoxPopuli were toggled off, which is why AMI (the column Tome cares most about) was missing and the averages look lower than the canonical ones.
- The screenshot's per-dataset values differ slightly from the current CSV (e.g., nar Earnings22 8.44 vs 8.15 in CSV; RTFx values differ ~2x) — the results file was re-generated recently: per the leaderboard GitHub README, recent English short-form evals migrated to HF Jobs on **H200** GPUs (https://github.com/huggingface/open_asr_leaderboard). Rankings are essentially unchanged; I use the canonical CSV below. **RTFx is datacenter-GPU (H200/A100-class), not Apple Silicon.**

The CSV has both raw and "Cleaned" reference columns for AMI/Gigaspeech/VoxPopuli (cleaned = re-processed references; neither the app code nor constants.py documents them, so I report both).

## 1. Full extracted table (english_short_latest.csv, sorted by default cleaned average)

| Model | Avg (cleaned) | Avg (orig) | RTFx | AMI-Cleaned | AMI (raw) | Earnings22 | VoxPop-Cleaned | VoxPop (raw) | LS Clean | LS Other | SPGI | Params (B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bosonai/higgs-audio-v3-stt | 4.62 | 5.04 | 110 | 7.19 | 7.86 | 8.26 | 3.73 | 5.93 | 1.07 | 2.60 | 2.07 | 2.68 |
| bosonai/higgs-audio-v3-8b-stt-v2 | 4.73 | 5.25 | 139 | 8.37 | 9.35 | 8.45 | 2.94 | 5.62 | 0.95 | 2.05 | 3.23 | 8.91 |
| ibm-granite/granite-speech-4.1-2b | 4.90 | 5.18 | 547 | 7.06 | 7.72 | 8.23 | 4.18 | 5.40 | 1.02 | 2.17 | 3.47 | 2 |
| ibm-granite/granite-speech-4.1-2b-nar | 4.95 | 5.25 | 2079 | **6.96** | **7.64** | **8.15** | 4.25 | 5.63 | 1.04 | 2.40 | 3.23 | 2 |
| Qwen/Qwen3-ASR-1.7B | 5.02 | 5.59 | 394 | 8.31 | 9.26 | 9.88 | 3.01 | 5.99 | 1.24 | 2.92 | 2.58 | 2.04 |
| nvidia/canary-qwen-2.5b | 5.06 | 5.41 | 861 | 7.91 | 9.05 | 10.04 | 4.14 | 5.38 | 1.23 | 2.62 | 1.70 | 2.5 |
| ibm-granite/granite-4.0-1b-speech | 5.10 | 5.37 | 661 | 7.37 | 8.12 | 8.33 | 4.41 | 5.51 | 1.10 | 2.49 | 3.55 | 2 |
| CohereLabs/cohere-transcribe-03-2026 | 5.20 | 5.35 | 916 | 7.01 | 7.80 | 10.38 | 5.38 | 5.58 | 0.96 | 2.04 | 2.74 | 2 |
| ibm-granite/granite-speech-3.3-8b | 5.29 | 5.54 | 264 | 7.70 | 8.43 | 9.07 | 4.54 | 5.44 | 1.11 | 2.52 | 3.54 | 9 |
| **nvidia/parakeet-tdt-0.6b-v2** (Tome default option) | 5.39 | 5.86 | 6038 | 9.10 | 10.40 | 10.78 | 3.78 | 5.68 | 1.27 | 2.73 | 1.94 | 0.6 |
| **nvidia/parakeet-tdt-0.6b-v3** (FluidAudio v3) | 5.66 | 6.22 | 6098 | 9.41 | 10.58 | 10.77 | 3.19 | 5.89 | 1.51 | 3.12 | 3.63 | 0.6 |
| nyrahealth/CrisperWhisper | 5.76 | 6.56 | 33 | 7.10 | 8.04 | 12.43 | 4.27 | 8.60 | 1.99 | 3.95 | 1.94 | 2 |
| distil-whisper/distil-large-v3.5 | 6.10 | 7.03 | 874 | 12.09 | 13.36 | 10.83 | 2.52 | 7.69 | 1.93 | 4.50 | 2.62 | 0.8 |
| **openai/whisper-large-v3** | 6.55 | 7.33 | 462 | 13.63 | 14.86 | 11.59 | 4.53 | 8.70 | 1.55 | 3.52 | 2.71 | 2 |
| **openai/whisper-large-v3-turbo** (Tome accuracy option) | 7.01 | 7.80 | 783 | 13.87 | 15.16 | 11.07 | 7.02 | 11.22 | 2.13 | 3.70 | 2.79 | 0.8 |

Source: https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results (file `english_short_latest.csv`). Note: whisper-large-v3-turbo is actually *worse* than plain whisper-large-v3 across the board (avg 7.01 vs 6.55; VoxPopuli raw 11.22 vs 8.70) — Tome's "accuracy" option is the weakest large-Whisper variant on this board. Also note **CrisperWhisper** (a verbatim-focused whisper-large-v3 fine-tune) hits AMI 8.04 — evidence that most of the AMI gap is conversational fine-tuning, not architecture.

## 2. Deltas that matter for Tome (relative WER reduction, cleaned refs; raw AMI in parens)

**vs whisper-large-v3-turbo (current accuracy option):**

| Candidate | Avg | AMI | Earnings22 |
|---|---|---|---|
| granite-speech-4.1-2b | 4.90 (**−30%**) | 7.06 (**−49%**; raw −49%) | 8.23 (**−26%**) |
| granite-speech-4.1-2b-nar | 4.95 (−29%) | 6.96 (**−50%**; raw −50%) | 8.15 (−26%) |
| higgs-audio-v3-stt (2.7B) | 4.62 (−34%) | 7.19 (−48%) | 8.26 (−25%) |
| higgs-audio-v3-8b-stt-v2 | 4.73 (−33%) | 8.37 (−40%) | 8.45 (−24%) |
| canary-qwen-2.5b | 5.06 (−28%) | 7.91 (−43%) | 10.04 (−9%) |
| granite-4.0-1b-speech | 5.10 (−27%) | 7.37 (−47%) | 8.33 (−25%) |

**vs parakeet-tdt-0.6b-v3 (current default):**

| Candidate | Avg | AMI | Earnings22 |
|---|---|---|---|
| granite-speech-4.1-2b | −13% | −25% (raw −27%) | −24% |
| granite-speech-4.1-2b-nar | −13% | −26% (raw −28%) | −24% |
| higgs-audio-v3-stt | −18% | −24% | −23% |
| canary-qwen-2.5b | −11% | −16% | −7% |

Headline: on AMI (the meeting corpus), granite-4.1 roughly **halves** whisper-large-v3-turbo's errors (15.16→7.64 raw) and cuts parakeet-v3's by ~27% (10.58→7.64). Against parakeet, granite's biggest wins are exactly on the meeting-like sets (AMI, Earnings22); on VoxPopuli parakeet-v3 is actually better (3.19 vs 4.25 cleaned) — the gains are concentrated where Tome needs them.

## 3. What the delta means in practice

At meeting-like WER (8–15%), per 1,000 spoken words (~6–7 min of meeting at 150 wpm):
- whisper-large-v3-turbo on AMI-style audio: ~152 errors/1000 words ≈ 22 errors/minute-of-speech. granite-4.1: ~77/1000 ≈ 11/min. **~680 fewer errors per hour-long meeting.**
- parakeet-v3 → granite-4.1 on AMI: ~106 → ~77/1000, ~29 fewer per 1000 (~260/hour).
- A 2–4 point absolute drop in the 8–12% band = 17–40% relative = going from roughly one error every 1.5 sentences to one every 2–3 sentences.
- Distribution matters more than the count: leaderboard WER is computed after Whisper-style normalization (punctuation/case/numeral/filler forgiveness — confirmed in the space's constants.py), so remaining errors concentrate on **content words, proper nouns, and domain terms** — exactly what poisons meeting summaries and action items. Two mitigations specific to the granite 4.x family: **keyword-list biasing** (prime the model with attendee/project names; announced for granite-4.0-1b-speech, https://huggingface.co/blog/ibm-granite/granite-4-speech) and **granite-speech-4.1-2b-plus**, which adds speaker-attributed transcripts + word timings (https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus).
- Qualitative accented/noisy comparisons: (a) an independent (small: 58 clips, diverse accents) edge benchmark measured granite-speech-3.3-8b at **8.18% clean / 15.72% noisy** vs Whisper **19.96% clean / 29.80% noisy** (https://www.ionio.ai/blog/2025-edge-speech-to-text-model-benchmark-whisper-vs-competitors); (b) IBM cites a Royal Flying Doctor Service field test where granite was "far better at handling the background noise than any other commercial models available" (https://research.ibm.com/blog/granite-4-1-ai-foundation-models); (c) on the long-form board's CORAAL (African-American vernacular English — an accent-robustness proxy), parakeet-v3 15.57 beats whisper-large-v3 18.89 and canary-qwen 19.05; granite is absent. The leaderboard also now runs private accented/held-out test sets from Appen and DataOcean (repos `hf-audio/appen_shortform_results` / `dataocean_shortform_results` — access-gated, I couldn't pull them), built specifically to catch public-benchmark overfitting.

## 4. The "28% on English for poor audio/accents" quote — hunt result

**No verbatim published "28%" claim was found** in IBM marketing, IBM Research blogs, or press coverage after several targeted searches. What exists:
- IBM Research, 29 Apr 2026 (Granite 4.1 announcement): "Granite Speech 4.1 2B achieves a 5.33% word-error rate (WER), placing it among the top models" — no relative % (https://research.ibm.com/blog/granite-4-1-ai-foundation-models).
- **The arithmetic that reproduces 28%:** 5.33 (granite-4.1-2b leaderboard avg at launch) vs whisper-large-v3's then-displayed ~7.4 avg → (7.4−5.33)/7.4 = **28.0% relative WER reduction**. Third-party coverage did the same math for the 1B model and headlined "beating Whisper Large V3 by 25% on word error rate" (5.52 vs ~7.4) (https://awesomeagents.ai/news/ibm-granite-4-speech-edge-asr/). So the remembered "28%" is almost certainly **relative WER reduction vs whisper-large-v3 on the Open ASR Leaderboard average** — a mixed benchmark, *not* a specific accents/poor-audio measurement. IBM's accent/noise language is qualitative ("industry-leading transcription accuracy across accents, domains and noisy environments", https://www.ibm.com/granite). Using today's CSV, the equivalent numbers are −25% vs whisper-large-v3 and −30% vs whisper-large-v3-turbo — so the figure is real in spirit, and on AMI specifically it *understates* the gap (−49%).
- Also verified: **there is no granite-speech-4.2 / "4.2.1-2b"**. The ibm-granite collection tops out at the 4.1 family: granite-speech-4.1-2b, -2b-plus, -2b-nar (all Apache-2.0), plus granite-4.0-1b-speech (https://huggingface.co/collections/ibm-granite/granite-speech). The user's "4.2.1-2b" is a misremembering of "4.1-2b".

## 5. Leaderboard caveats

1. **In-domain training (biggest caveat):** granite-speech-4.1-2b's model card lists **AMI (100h) and Earnings-22 (105h)** — plus VoxPopuli, CommonVoice, LibriSpeech, Fisher, Switchboard — in its training data (https://huggingface.co/ibm-granite/granite-speech-4.1-2b). These are train partitions, not the test files, so it's not literal contamination, but granite's AMI/E22 edge is partly "trained on the same corpora's training splits" — expect the real-world gap on Zoom/Teams meeting audio to be smaller than the leaderboard gap (though training on meeting speech is arguably exactly the specialization Tome wants). CrisperWhisper's AMI 8.04 from a whisper fine-tune supports this read.
2. **Normalization:** WER computed after Whisper-style normalization (case/punct/numerals/fillers removed) — verbatim fidelity differences are hidden; punctuation quality of the raw output still differs between models.
3. **Short-form vs long-form:** the main table is short-segment (<~30s) evaluation. On the separate **long-form track** (full recordings: earnings21/22, TED-LIUM, CORAAL; https://huggingface.co/datasets/Steveeeeeeen/leaderboard_longform): **parakeet-tdt-0.6b-v3 is the best open model (avg 10.72, RTFx 1003), beating whisper-large-v3-turbo (11.01, RTFx 148), whisper-large-v3 (11.23) and canary-qwen-2.5b (11.20, RTFx 16 — LLM-decoder long-form is ~60x slower)**. **Granite-speech and Higgs are absent from the long-form track entirely.** Whisper has native sequential long-form decoding; granite 4.1 is built around short segments — the -plus variant explicitly supports chunked long-form via "incremental decoding with prefix passing" to keep speaker numbering consistent across chunk seams (https://www.mindstudio.ai/blog/ibm-granite-speech-41-vs-whisper-x-transcription-pipeline); higgs (frozen Whisper-Large-v3 encoder + Qwen3-8B decoder, 8.91B, Apache-2.0, https://huggingface.co/bosonai/higgs-audio-v3-8b-stt-v2) inherits Whisper's 30s window and would need a VAD/chunking pipeline. Since Tome already segments audio for streaming, chunked inference is a solved problem architecturally, but leaderboard AMI numbers are on pre-segmented audio — chunk-boundary errors are extra.
4. **RTFx hardware:** recent English short-form evals run on HF Jobs with **H200** GPUs (https://github.com/huggingface/open_asr_leaderboard); earlier numbers were A100-class. All RTFx figures are meaningless for Apple Silicon except as relative ordering within an architecture class (e.g., nar's 2079 vs 4.1-2b's 547 ≈ 3.8x speedup from non-autoregressive decoding should roughly carry over).
5. **Leaderboard churn:** per-dataset values shifted a few tenths between the user's screenshot and today's CSV (eval re-runs); rankings stable. The "Cleaned" AMI/Gigaspeech/VoxPopuli columns (re-processed references) are undocumented in the app code; I report both — conclusions are identical either way.

## 6. Net read for Tome

For a meeting recorder that can spend minutes on post-processing, **granite-speech-4.1-2b (or -nar for ~4x decode speed at equal accuracy, or -plus for built-in speaker attribution)** is the standout accuracy target: ~50% fewer errors than whisper-large-v3-turbo and ~27% fewer than parakeet-v3 on the meeting corpus, 2B params (laptop-friendly), Apache-2.0. Higgs-audio-v3-stt (2.7B) edges it on the overall average but not on AMI, and the 8B v2 is worse on AMI than its own 2.7B sibling. Canary-qwen-2.5b is dominated by granite on every meeting-relevant axis. The main open risks are (a) in-domain-training inflation of the AMI delta, and (b) no CoreML/Swift runtime today — granite/higgs ports to Apple frameworks are still open requests (e.g., MLX: https://github.com/Blaizzy/mlx-audio/issues/737) — which is the adjacent workstream's question. Keeping parakeet-v3 for live streaming remains well-supported: it's the best open model on the long-form board and top-tier RTFx.

## BOTTOM LINE
On the Open ASR Leaderboard's AMI meeting corpus, granite-speech-4.1-2b(-nar) roughly halves whisper-large-v3-turbo's WER (15.16→7.64 raw, −50%) and cuts parakeet-tdt-0.6b-v3's by ~27% (10.58→7.64) — ~680 fewer errors per hour-long meeting vs turbo — making it the best accuracy target for Tome's post-processing slot; higgs-audio-v3 wins the overall average but not AMI. The user's "28%" figure is not a published accents/poor-audio metric: it's the relative reduction of granite-4.1-2b's launch average WER (5.33) vs whisper-large-v3's ~7.4 on the leaderboard, and no granite-speech-4.2 exists (4.1 family is latest). Two caveats temper the AMI delta: granite trains on AMI/Earnings-22 training splits (in-domain advantage), and granite is absent from the long-form track, where parakeet-tdt-0.6b-v3 is the best open model (10.72 avg) — so a fast-live-parakeet + granite-post-processing split, with chunking for long audio, is well supported by the data.

## VERIFICATION VERDICTS

- [CONFIRMED] english_short_latest.csv: granite-speech-4.1-2b-nar 7.64 AMI (6.96 cleaned) vs whisper-large-v3-turbo 15.16 (13.87) and parakeet-tdt-0.6b-v3 10.58 (9.41); ~50% and ~27% relative error reduction
  NOTE: Fetched the raw CSV from hf-audio/open-asr-leaderboard-results (english_short_latest.csv). Digit-by-digit match: granite-speech-4.1-2b-nar AMI WER 7.64 / AMI-Cleaned 6.96; openai/whisper-large-v3-turbo 15.16 / 13.87; nvidia/parakeet-tdt-0.6b-v3 10.58 / 9.41. Relative reductions compute to 49.6% (raw) / 49.8% (cleaned) vs turbo and 27.8% (raw) / 26.0% (cleaned) vs parakeet — '~50%' and '~27%' are fair.

- [CONFIRMED] No granite-speech-4.2 exists as of 2026-07-09; newest IBM speech models are granite-speech-4.1-2b, -2b-plus, -2b-nar plus granite-4.0-1b-speech, all Apache-2.0
  NOTE: HF API model search for 'granite-speech-4.2' returns 0 results; ibm-granite org listing shows granite-speech-4.1-2b (created 2026-04-16), -2b-plus (2026-04-16), -2b-nar (2026-03-10), granite-4.0-1b-speech (2026-02-27), plus older 3.x models and GGUF conversions — nothing newer. All four carry license:apache-2.0 tags. Minor: the cited collection URL is a stub (real HF collection URLs need a slug-hash), but I verified via the org's model API directly; substance holds.

- [CONFIRMED] granite-speech-4.1-2b model card lists AMI (100h) and Earnings-22 (105h) in training data, so AMI/Earnings22 leaderboard scores are partly in-domain
  NOTE: Raw README.md of ibm-granite/granite-speech-4.1-2b contains a Training Data table with rows 'AMI English | ASR | 100 | edinburghcstr/ami' and 'Earnings-22 English | ASR | 105 | esb/datasets'. Training uses the train splits while the leaderboard tests on test splits, so 'in-domain rather than zero-shot' is the accurate characterization — confirmed.

- [CONFIRMED] IBM Granite 4.1 announcement (29 Apr 2026) states 5.33% average WER on Open ASR Leaderboard; no published IBM '28% better for accents/poor audio' claim; 28% matches relative reduction of 5.33 vs whisper-large-v3's ~7.4 average
  NOTE: Fetched research.ibm.com/blog/granite-4-1-ai-foundation-models: published 29 Apr 2026, states 'Granite Speech 4.1 2B achieves a 5.33% word-error rate (WER), placing it among the top models on the OpenASR Leaderboard.' The blog contains no '28%' or accent/noise percentage claim, and a web search found no IBM-published 28% figure either (only third-party MindStudio posts, none citing 28%). Arithmetic: (7.4-5.33)/7.4 = 27.97% ≈ 28%. Caveat: whisper-large-v3's current leaderboard original average is 7.33, so '~7.4 then-listed' is plausible but the exact historical snapshot value was not independently verifiable; the derivation is a reasonable inference, not an IBM statement.

- [CONFIRMED] Long-form track (earnings21, earnings22, TED-LIUM, CORAAL): parakeet-tdt-0.6b-v3 avg 10.72, beating whisper-large-v3-turbo 11.01, whisper-large-v3 11.23, canary-qwen-2.5b 11.20 (RTFx ~16); granite and higgs absent
  NOTE: Fetched longform_latest.csv from Steveeeeeeen/leaderboard_longform. Exact values: parakeet-tdt-0.6b-v3 Average 10.72 (RTFx 1002.91); whisper-large-v3-turbo 11.01; whisper-large-v3 11.2275 (rounds to 11.23); canary-qwen-2.5b 11.2025 (rounds to 11.20) with RTFx 16.05. Columns are earnings21, earnings22, tedlium, coraal_avg. No ibm-granite or bosonai/higgs rows exist in the file. Note the claim only says parakeet beats those named open models, which is true; several proprietary entries (e.g. elevenlabs/scribe_v2 7.32, assembly/universal-3-pro 8.34) score lower overall, but the claim does not assert parakeet is #1.

- [CONFIRMED] higgs-audio-v3-8b-stt-v2 is Apache-2.0, 8.91B params, frozen Whisper-Large-v3 encoder + Qwen3-8B decoder, model card reports 10.14% AMI WER (leaderboard 9.35 raw / 8.37 cleaned), worse on AMI than 2.7B sibling higgs-audio-v3-stt
  NOTE: Model card README states: license apache-2.0; 'Encoder: Whisper-Large-v3 (frozen)'; 'Decoder: Qwen3-8B (LoRA fine-tuned, merged)'; 'Total parameters: 8.91B' (safetensors API: 8,905,965,568 ≈ 8.91B); performance table lists AMI 10.14%. Leaderboard CSV confirms 9.35 AMI raw / 8.37 cleaned. Sibling bosonai/higgs-audio-v3-stt is 2,675,546,112 params (2.68B ≈ '2.7B') with AMI 7.86 raw / 7.19 cleaned — better than the 8B on AMI, as claimed.
