# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets[audio]>=3", "soundfile", "jiwer", "transformers", "numpy"]
# ///
"""Phase 0 ASR benchmark. Stages:
  uv run bench.py export  --work /tmp/asrbench --sets ami,earnings22,tedlium --max-hours 3
  (then run ASRBench manifest mode for parakeet + whisper — command is printed)
  uv run bench.py granite --work /tmp/asrbench --url http://127.0.0.1:8873
  uv run bench.py score   --work /tmp/asrbench
Reference/normalizer per Open ASR Leaderboard: WhisperTokenizer.normalize (public
method on transformers 5.x; the brief's `_normalize` was the private name on an
older transformers release and no longer exists — verified live against
transformers 5.13.0, see task-4-report.md).

Dataset sourcing (verified against github.com/huggingface/open_asr_leaderboard and
live HF Hub probes — see docs/superpowers/plans task-4-report.md for the full trail):
  - ami, earnings22: hf-audio/open-asr-leaderboard (parquet, config name == set name,
    split "test", ref column "text"). This is the successor repo behind the
    hf-audio/esb-datasets-test-only-sorted alias (which 307-redirects to it) —
    used directly here to avoid the redirect. NOTE: this repo's own README lists a
    "tedlium" config, but the tedlium/ directory does not actually exist there
    (confirmed via the Hub tree API) — a genuine gap in that repo, not a config
    typo on our part.
  - tedlium: distil-whisper/tedlium, config "default", split "test", ref column
    "text", pinned to revision "refs/convert/parquet" (the Hub's auto-generated
    parquet mirror of this legacy loading-script dataset — `datasets>=3` refuses
    to execute loading scripts at all, so the un-pinned repo is unusable; the
    parquet-convert branch collapses the script's "release3" config name down to
    "default"). Raw text retains TED-LIUM STM artifacts (`ignore_time_segment_in_scoring`,
    and similar bracket/gap tokens) that upstream's own loader would normally drop
    at generation time — export() filters those explicitly (see SKIP_MARKERS)
    and reports a skipped count.
"""
import argparse, json, pathlib, sys, time

# (dataset repo id, config name, revision-or-None) per set — see module docstring.
DATASETS = {
    "ami": ("hf-audio/open-asr-leaderboard", "ami", None),
    "earnings22": ("hf-audio/open-asr-leaderboard", "earnings22", None),
    "tedlium": ("distil-whisper/tedlium", "default", "refs/convert/parquet"),
}
# TED-LIUM STM scoring-gap / non-speech markers (mirrors open_asr_leaderboard's
# `ignore_segments` filtering, which upstream's now-unusable loading script applied
# at generation time). Matched against the *raw* (pre-normalization) reference text.
SKIP_MARKERS = {"ignore_time_segment_in_scoring", "<unk>", ""}


def export(work, sets, max_hours):
    import soundfile as sf
    from datasets import load_dataset, Audio
    for s in sets:
        repo, config, revision = DATASETS[s]
        d = work / s; (d / "wav").mkdir(parents=True, exist_ok=True)
        kwargs = {"revision": revision} if revision else {}
        ds = load_dataset(repo, config, split="test", streaming=True, **kwargs)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        refcol = next(c for c in ("text", "norm_transcript", "transcription", "sentence")
                      if c in ds.column_names)
        total, manifest, refs, skipped = 0.0, [], {}, 0
        for i, row in enumerate(ds):
            ref_raw = row[refcol]
            if ref_raw is None or ref_raw.strip() in SKIP_MARKERS:
                skipped += 1; continue
            audio = row["audio"]; dur = len(audio["array"]) / audio["sampling_rate"]
            if total + dur > max_hours * 3600: break
            total += dur
            rid = f"{s}-{i:05d}"; wav = d / "wav" / f"{rid}.wav"
            sf.write(wav, audio["array"], 16000, subtype="PCM_16")
            manifest.append({"id": rid, "wav": str(wav)}); refs[rid] = {"ref": ref_raw, "dur": dur}
        (d / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
        (d / "refs.json").write_text(json.dumps(refs))
        print(f"[{s}] {len(manifest)} utts, {total/3600:.2f} h, {skipped} skipped  (repo: {repo}, ref column: {refcol})")
    print("\nNow produce Tome-backend hypotheses (from Tome/, using the prebuilt release binary):")
    for s in sets:
        for b in ("parakeet", "whisper"):
            print(f"  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Tome/.build/release/ASRBench "
                  f"--manifest {work}/{s}/manifest.jsonl --backend {b} --out {work}/{s}/hyp_{b}.jsonl")


def granite(work, url):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from granite_client import transcribe
    for d in sorted(p for p in work.iterdir() if (p / "manifest.jsonl").exists()):
        out, wall, audio_s = [], 0.0, 0.0
        refs = json.loads((d / "refs.json").read_text())
        for line in (d / "manifest.jsonl").read_text().splitlines():
            m = json.loads(line)
            try:
                text, dt = transcribe(url, m["wav"])
            except Exception as e:                      # noqa: BLE001 — record and continue
                text, dt = "", 0.0; print(f"  ERR {m['id']}: {e}")
            out.append({"id": m["id"], "text": text}); wall += dt; audio_s += refs[m["id"]]["dur"]
            if len(out) % 50 == 0: print(f"[{d.name}] {len(out)} done, RTF so far {wall/max(audio_s,1):.3f}")
        (d / "hyp_granite.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
        print(f"[{d.name}] granite RTF (single-stream M2 Max): {wall/max(audio_s,1):.3f}")


def score(work):
    import jiwer
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    rows = []
    for d in sorted(p for p in work.iterdir() if (p / "refs.json").exists()):
        refs = json.loads((d / "refs.json").read_text())
        for hyp_file in sorted(d.glob("hyp_*.jsonl")):
            hyps = {json.loads(l)["id"]: json.loads(l)["text"] for l in hyp_file.read_text().splitlines()}
            pairs = [(tok.normalize(refs[i]["ref"]), tok.normalize(hyps.get(i, "")))
                     for i in refs if tok.normalize(refs[i]["ref"]).strip()]
            wer = jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) * 100
            rows.append((d.name, hyp_file.stem.removeprefix("hyp_"), wer, len(pairs)))
    print(f"{'set':<12}{'backend':<12}{'WER%':>8}{'utts':>7}")
    for s, b, w, n in rows: print(f"{s:<12}{b:<12}{w:>8.2f}{n:>7}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["export", "granite", "score"])
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--sets", default="ami,earnings22,tedlium"); ap.add_argument("--max-hours", type=float, default=3)
    ap.add_argument("--url", default="http://127.0.0.1:8873")
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    {"export": lambda: export(a.work, a.sets.split(","), a.max_hours),
     "granite": lambda: granite(a.work, a.url),
     "score": lambda: score(a.work)}[a.stage]()
