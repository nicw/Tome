# Granite Shadow Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 0 benchmark (granite-speech-4.1-2b via llama.cpp vs Tome's backends on public test sets with references) + hidden-flag shadow transcription of real meetings, per `docs/superpowers/specs/2026-07-09-granite-shadow-transcription-design.md`.

**Architecture:** A `llama-server` sidecar (spawn-per-job) serves granite; a Python harness (`scripts/asr-bench/`) measures WER on the leaderboard's ESB test sets against Tome's real backends (via a new ASRBench manifest mode); in Tome, a best-effort shadow phase inside `PostProcessingJob` re-transcribes the same merged diarized segments through granite and writes comparison artifacts. Nothing touches ASRCoordinator/ModelProvisioner/TranscriberModel.

**Tech Stack:** Swift 6 (SwiftPM, actors, Swift Testing suite as in existing tests), llama.cpp (`llama-server`, mtmd audio), Python 3.11+ via `uv` (datasets/jiwer/transformers for Phase 0; stdlib-only for the report script), bash + curl for setup.

**DEADLINE:** shadow must be live in Nic's installed Tome by **Friday 2026-07-10 morning** (meetings that day; review EOD Wednesday 2026-07-15). Phase 0 gates block **enabling** the flag, not building. Task 4's compute can run while Tasks 5–12 proceed.

## Global Constraints

- Branch: `granite-shadow-transcription`. Commit at the end of every task (`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`).
- `swift test` and `swift build` require `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- NEVER modify `Tome/Sources/Tome/API/**` (frozen pending Nic/Dan discussion).
- This feature adds NO `TranscriberModel` enum case, no Settings UI, no ModelProvisioner/ASRCoordinator changes.
- UserDefaults domain `com.dloomis.tome`; keys exactly: `graniteShadowEnabled` (Bool), `graniteShadowServerPath` (String), `graniteShadowModelDir` (String), `graniteShadowPort` (Int).
- Model files: IBM official GGUFs from `ibm-granite/granite-speech-4.1-2b-GGUF` (Q8_0 + `mmproj-model-f16.gguf`), stored in `~/Library/Application Support/Tome/Granite/`. llama.cpp ≥ **b9045**. Downloads use **curl** (never URLSession — known HF-CDN failure).
- Request template source of truth: `scripts/asr-bench/granite_request.md` (Task 2). Swift `GraniteRequest` must match it (golden test).
- Shadow artifacts dir: `~/Library/Application Support/Tome/GraniteShadow/`.
- Shadow phase NEVER throws out of `PostProcessingJob`; primary transcript/lifecycle behavior byte-identical when flag off AND when shadow fails.
- Existing 89 tests must stay green after every task.
- `scripts/granite-shadow-report.py` is Python-stdlib-only. `scripts/asr-bench/*` may use uv-managed deps.

---

### Task 1: Setup script + real model download

**Files:**
- Create: `scripts/setup-granite-shadow.sh`

**Interfaces:**
- Produces: `llama-server` verified ≥ b9045; `~/Library/Application Support/Tome/Granite/{granite-speech-4.1-2b-Q8_0.gguf, mmproj-model-f16.gguf}` on disk; a proven `llama-server` launch command.

- [ ] **Step 1: Confirm exact GGUF filenames** (they are constants consumed by Task 5's `ShadowConfig` — if they differ from the names below, update BOTH this script and the `ShadowConfig` filename constants in Task 5 before proceeding):

Run: `curl -s https://huggingface.co/api/models/ibm-granite/granite-speech-4.1-2b-GGUF | python3 -c "import json,sys; [print(s['rfilename']) for s in json.load(sys.stdin)['siblings']]"`
Expected: a `*Q8_0.gguf` file and `mmproj-model-f16.gguf`.

- [ ] **Step 2: Write the script**

```bash
#!/usr/bin/env bash
# Setup for granite shadow transcription (spec: docs/superpowers/specs/2026-07-09-granite-shadow-transcription-design.md)
set -euo pipefail

MODEL_DIR="$HOME/Library/Application Support/Tome/Granite"
REPO="https://huggingface.co/ibm-granite/granite-speech-4.1-2b-GGUF/resolve/main"
MODEL="granite-speech-4.1-2b-Q8_0.gguf"     # keep in sync with ShadowConfig.modelFilename
MMPROJ="mmproj-model-f16.gguf"              # keep in sync with ShadowConfig.mmprojFilename
SERVER="${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}"
PORT="${GRANITE_PORT:-8873}"

if [[ ! -x "$SERVER" ]]; then
  echo "llama-server not found at $SERVER — run: brew install llama.cpp" >&2
  exit 1
fi
# b9045+ required for granite-speech mtmd support
BUILD=$("$SERVER" --version 2>&1 | grep -oE 'b[0-9]+' | head -1 | tr -d 'b')
if [[ -z "$BUILD" || "$BUILD" -lt 9045 ]]; then
  echo "llama.cpp build b${BUILD:-unknown} < b9045 — run: brew upgrade llama.cpp" >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
for f in "$MODEL" "$MMPROJ"; do
  echo "Downloading $f (resumable)…"
  curl -L -C - --fail -o "$MODEL_DIR/$f" "$REPO/$f"
done
ls -lh "$MODEL_DIR"

cat <<EOF

Setup complete. Enable shadow mode with:
  defaults write com.dloomis.tome graniteShadowEnabled -bool YES
(Disable: defaults write com.dloomis.tome graniteShadowEnabled -bool NO)

Proof-of-life (Ctrl-C to stop the server when done):
  "$SERVER" -m "$MODEL_DIR/$MODEL" --mmproj "$MODEL_DIR/$MMPROJ" --host 127.0.0.1 --port $PORT
EOF
```

- [ ] **Step 3: Run it** — `chmod +x scripts/setup-granite-shadow.sh && ./scripts/setup-granite-shadow.sh`. Expected: both files download (~3.1 GB total), sizes match the HF repo listing (Q8_0 ≈ 1.96 GB, mmproj ≈ 1.16 GB).
- [ ] **Step 4: Proof-of-life** — launch the printed server command in the background; `curl -s http://127.0.0.1:8873/health` returns 200/`{"status":"ok"}` once loaded. Leave running for Task 2.
- [ ] **Step 5: Commit** — `git add scripts/setup-granite-shadow.sh && git commit -m "feat: granite shadow setup script (llama.cpp check + GGUF download via curl)"`

### Task 2: Pin the llama-server request template + Python granite client

**Files:**
- Create: `scripts/asr-bench/granite_request.md`
- Create: `scripts/asr-bench/granite_client.py`

**Interfaces:**
- Produces: `granite_request.md` — THE pinned request contract (endpoint, JSON shape, prompt, params) that Task 8's Swift `GraniteRequest` implements verbatim. `granite_client.py`: `build_request(wav_bytes: bytes, prompt: str) -> dict`, `transcribe(base_url: str, wav_path: str) -> tuple[str, float]` (text, latency-seconds).

- [ ] **Step 1: Discover the working request shape** against the live server from Task 1. Try the OpenAI-compatible endpoint first:

```bash
python3 - <<'EOF'
import base64, json, urllib.request
wav = open("/System/Library/Sounds/Submarine.aiff", "rb")  # placeholder — use a real 16 kHz mono WAV, see below
EOF
```
Use a real speech WAV: record 10 s (`say -o /tmp/probe.aiff "the quick brown fox jumps over the lazy dog" && afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/probe.aiff /tmp/probe.wav`). POST to `http://127.0.0.1:8873/v1/chat/completions`:

```json
{"messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": "<base64 wav>", "format": "wav"}},
    {"type": "text", "text": "can you transcribe the speech into a written format?"}]}],
 "temperature": 0, "max_tokens": 2048, "stream": false}
```
Expected: `choices[0].message.content` ≈ "the quick brown fox jumps over the lazy dog" (case/punct may vary). If `input_audio` is rejected, consult `llama-server --help` / llama.cpp `docs/multimodal.md` for the accepted audio content type and record what works. If the server path cannot transcribe at all, STOP: fall back to `llama-mtmd-cli` per the spec's risk section and record that decision in `granite_request.md` (the sidecar then shells out instead of HTTP — adjust Task 9 accordingly).

- [ ] **Step 2: Write `granite_request.md`** documenting exactly: endpoint path, full JSON body (with prompt string verbatim), required server launch flags, response extraction path (`choices[0].message.content`), and the probe transcript observed. This file is the single source of truth; both Python and Swift cite it.

- [ ] **Step 3: Write `granite_client.py`**

```python
"""Granite llama-server client. Request contract: see granite_request.md (source of truth)."""
import base64, json, time, urllib.request

PROMPT = "can you transcribe the speech into a written format?"  # granite_request.md

def build_request(wav_bytes: bytes, prompt: str = PROMPT) -> dict:
    return {
        "messages": [{"role": "user", "content": [
            {"type": "input_audio",
             "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0, "max_tokens": 2048, "stream": False,
    }

def transcribe(base_url: str, wav_path: str) -> tuple[str, float]:
    body = json.dumps(build_request(open(wav_path, "rb").read())).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    return out["choices"][0]["message"]["content"].strip(), time.monotonic() - t0

if __name__ == "__main__":
    import sys
    text, dt = transcribe(sys.argv[1] if len(sys.argv) > 2 else "http://127.0.0.1:8873", sys.argv[-1])
    print(f"[{dt:.1f}s] {text}")
```
(Adjust `build_request` to whatever Step 1 actually pinned.)

- [ ] **Step 4: Verify** — `python3 scripts/asr-bench/granite_client.py /tmp/probe.wav` prints the fox sentence. Note the latency: first M2 Max RTF datapoint.
- [ ] **Step 5: Commit** — `git add scripts/asr-bench && git commit -m "feat: pin granite llama-server request template + python client"`

### Task 3: BenchSupport manifest library + ASRBench manifest mode

**Files:**
- Create: `Tome/Sources/BenchSupport/BenchManifest.swift`
- Modify: `Tome/Package.swift` (add `BenchSupport` library target; `ASRBench` and `TomeTests` depend on it)
- Modify: `Tome/Sources/ASRBench/main.swift` (manifest mode)
- Test: `Tome/Tests/TomeTests/BenchManifestTests.swift`

**Interfaces:**
- Produces: `BenchManifest.parse(_ jsonl: String) throws -> [ManifestEntry]` where `ManifestEntry(id: String, wav: String)`; `BenchManifest.emit(_ hyps: [HypothesisEntry]) -> String` where `HypothesisEntry(id: String, text: String)`. CLI: `ASRBench --manifest in.jsonl --backend parakeet|whisper --out hyp.jsonl` (Task 4 consumes).

- [ ] **Step 1: Failing test** (`BenchManifestTests.swift`, match the existing suite's Swift Testing style):

```swift
import Testing
@testable import BenchSupport

@Suite struct BenchManifestTests {
    @Test func parsesJSONLAndSkipsBlankLines() throws {
        let jsonl = """
        {"id": "ami-0001", "wav": "/tmp/a.wav"}

        {"id": "ami-0002", "wav": "/tmp/b.wav"}
        """
        let entries = try BenchManifest.parse(jsonl)
        #expect(entries == [ManifestEntry(id: "ami-0001", wav: "/tmp/a.wav"),
                            ManifestEntry(id: "ami-0002", wav: "/tmp/b.wav")])
    }
    @Test func emitRoundTrips() throws {
        let hyps = [HypothesisEntry(id: "x", text: "hello there")]
        let out = BenchManifest.emit(hyps)
        #expect(out == #"{"id":"x","text":"hello there"}"# + "\n")
    }
    @Test func parseRejectsMalformedLine() {
        #expect(throws: (any Error).self) { try BenchManifest.parse("not json") }
    }
}
```

- [ ] **Step 2: Run** — `cd Tome && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter BenchManifestTests`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.** Package.swift: add `.target(name: "BenchSupport")`; add `"BenchSupport"` to the `ASRBench` executable target's and `TomeTests`' dependencies.

```swift
// Tome/Sources/BenchSupport/BenchManifest.swift
import Foundation

public struct ManifestEntry: Codable, Equatable, Sendable {
    public let id: String
    public let wav: String
    public init(id: String, wav: String) { self.id = id; self.wav = wav }
}

public struct HypothesisEntry: Codable, Equatable, Sendable {
    public let id: String
    public let text: String
    public init(id: String, text: String) { self.id = id; self.text = text }
}

public enum BenchManifest {
    public static func parse(_ jsonl: String) throws -> [ManifestEntry] {
        try jsonl.split(separator: "\n", omittingEmptySubsequences: true)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { try JSONDecoder().decode(ManifestEntry.self, from: Data($0.utf8)) }
    }
    public static func emit(_ hyps: [HypothesisEntry]) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return hyps.map { String(data: try! enc.encode($0), encoding: .utf8)! }
            .joined(separator: "\n") + (hyps.isEmpty ? "" : "\n")
    }
}
```

- [ ] **Step 4: Run tests** — same command. Expected: PASS. Then full suite: `swift test` — 89 + 3 green.
- [ ] **Step 5: Manifest mode in `main.swift`.** At the top of the existing top-level code, before the bench runs:

```swift
// Manifest mode: ASRBench --manifest in.jsonl --backend parakeet|whisper --out hyp.jsonl
// Reuses the same hand-mirrored model config as the bench functions below (keep in sync).
if let mi = CommandLine.arguments.firstIndex(of: "--manifest") {
    let args = CommandLine.arguments
    guard args.count > mi + 1,
          let bi = args.firstIndex(of: "--backend"), args.count > bi + 1,
          let oi = args.firstIndex(of: "--out"), args.count > oi + 1 else {
        FileHandle.standardError.write(Data("usage: ASRBench --manifest in.jsonl --backend parakeet|whisper --out hyp.jsonl\n".utf8))
        exit(2)
    }
    let entries = try BenchManifest.parse(String(contentsOfFile: args[mi + 1], encoding: .utf8))
    let backend = args[bi + 1]
    // Extract the model-loading half of benchParakeet()/benchWhisper() into
    // loadParakeet() / loadWhisper() helpers returning a `(String) async throws -> String`
    // transcribe closure (WAV path in, text out), reusing the existing sample-loading
    // code these bench functions already use for their own WAVs.
    let transcribe: (String) async throws -> String = backend == "whisper"
        ? try await loadWhisper()
        : try await loadParakeet()
    var hyps: [HypothesisEntry] = []
    for (i, e) in entries.enumerated() {
        let text = (try? await transcribe(e.wav)) ?? ""
        hyps.append(HypothesisEntry(id: e.id, text: text))
        if i % 50 == 0 { print("[\(backend)] \(i)/\(entries.count)") }
    }
    try BenchManifest.emit(hyps).write(toFile: args[oi + 1], atomically: true, encoding: .utf8)
    print("[\(backend)] wrote \(hyps.count) hypotheses → \(args[oi + 1])")
    exit(0)
}
```
Import `BenchSupport` at the top of main.swift. The `loadParakeet`/`loadWhisper` extraction must not change bench behavior — the existing `benchParakeet()`/`benchWhisper()` call the new helpers.

- [ ] **Step 6: Build + spot-check** — `swift build -c release`; run manifest mode on 2 WAVs (the Task 2 probe file twice) for each backend; verify hyp.jsonl content is sane.
- [ ] **Step 7: Commit** — `git commit -am "feat: ASRBench manifest mode + BenchSupport library"`

### Task 4: Phase 0 benchmark harness — build AND run

**Files:**
- Create: `scripts/asr-bench/bench.py`
- Create: `docs/superpowers/plans/2026-07-09-granite-phase0-results.md` (results, produced by running)

**Interfaces:**
- Consumes: `granite_client.transcribe`, `ASRBench --manifest`.
- Produces: the Phase 0 results doc with the WER table + RTF + go/no-go against the spec's three Phase 0 gates.

- [ ] **Step 1: Write `bench.py`** (uv inline-deps script; stages so granite/ASRBench runs are restartable):

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets[audio]>=3", "soundfile", "jiwer", "transformers", "torch", "numpy"]
# ///
"""Phase 0 ASR benchmark. Stages:
  uv run bench.py export  --work /tmp/asrbench --sets ami,earnings22,tedlium --max-hours 3
  (then run ASRBench manifest mode for parakeet + whisper — command is printed)
  uv run bench.py granite --work /tmp/asrbench --url http://127.0.0.1:8873
  uv run bench.py score   --work /tmp/asrbench
Reference/normalizer per Open ASR Leaderboard: WhisperTokenizer._normalize."""
import argparse, json, pathlib, sys, time

SETS = {"ami": "ami", "earnings22": "earnings22", "tedlium": "tedlium"}
ESB = "hf-audio/esb-datasets-test-only-sorted"

def export(work, sets, max_hours):
    import soundfile as sf
    from datasets import load_dataset, Audio
    for s in sets:
        d = work / s; (d / "wav").mkdir(parents=True, exist_ok=True)
        ds = load_dataset(ESB, SETS[s], split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        refcol = next(c for c in ("text", "norm_transcript", "transcription", "sentence")
                      if c in ds.column_names)
        total, manifest, refs = 0.0, [], {}
        for i, row in enumerate(ds):
            audio = row["audio"]; dur = len(audio["array"]) / audio["sampling_rate"]
            if total + dur > max_hours * 3600: break
            total += dur
            rid = f"{s}-{i:05d}"; wav = d / "wav" / f"{rid}.wav"
            sf.write(wav, audio["array"], 16000, subtype="PCM_16")
            manifest.append({"id": rid, "wav": str(wav)}); refs[rid] = {"ref": row[refcol], "dur": dur}
        (d / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
        (d / "refs.json").write_text(json.dumps(refs))
        print(f"[{s}] {len(manifest)} utts, {total/3600:.2f} h  (ref column: {refcol})")
    print("\nNow produce Tome-backend hypotheses (from Tome/):")
    for s in sets:
        for b in ("parakeet", "whisper"):
            print(f"  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift run -c release ASRBench "
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
            pairs = [(tok._normalize(refs[i]["ref"]), tok._normalize(hyps.get(i, "")))
                     for i in refs if tok._normalize(refs[i]["ref"]).strip()]
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
```
Held-out set note: **TED-LIUM stands in for CORAAL** (short-form, ungated, absent from granite's training list; CORAAL is long-form and needs its own chunking — stretch goal only if time allows). Record this substitution in the results doc.

- [ ] **Step 2: Run export** (`uv run scripts/asr-bench/bench.py export --work /tmp/asrbench`). Expected: 3 sets × ~3 h exported. If a dataset config name or ref column errors, check `open_asr_leaderboard`'s normalizer/dataset usage on GitHub and fix the constant.
- [ ] **Step 3: Run the two ASRBench manifest commands per set** (printed by export). Parakeet is minutes; Whisper tens of minutes.
- [ ] **Step 4: Run granite stage** (server from Task 1 running). Record per-set RTF lines.
- [ ] **Step 5: Score + write results doc** `docs/superpowers/plans/2026-07-09-granite-phase0-results.md`: the WER table, published leaderboard raw numbers alongside (granite AMI 7.72 / E22 8.23; parakeet-v3 AMI 10.58 / E22 10.77; whisper-turbo AMI 15.16 / E22 11.07), M2 Max RTF, and explicit pass/fail on the spec's three Phase 0 gates (fidelity ±1.5 pts on AMI+E22; RTF ≤ 0.25; granite beats both backends on AMI+E22 and ≥ matches parakeet on TED-LIUM). End with go/no-go for enabling shadow.
- [ ] **Step 6: Commit** — `git add scripts/asr-bench docs/superpowers/plans/2026-07-09-granite-phase0-results.md && git commit -m "feat: phase 0 ASR benchmark harness + results"`

### Task 5: ShadowConfig

**Files:**
- Create: `Tome/Sources/Tome/Transcription/ShadowConfig.swift`
- Test: `Tome/Tests/TomeTests/ShadowConfigTests.swift`

**Interfaces:**
- Produces: `ShadowConfig` (`serverPath: String`, `modelDir: URL`, `port: Int`; `modelGGUF/mmprojGGUF: URL`; `filesPresent() -> Bool`; `static func fromDefaults(_ defaults: UserDefaults) -> ShadowConfig?`). Consumed by Tasks 9–11.

- [ ] **Step 1: Failing tests**

```swift
import Foundation
import Testing
@testable import Tome

@Suite struct ShadowConfigTests {
    private func makeDefaults() -> UserDefaults {
        let d = UserDefaults(suiteName: "ShadowConfigTests-\(UUID().uuidString)")!
        d.removePersistentDomain(forName: d.description)
        return d
    }
    @Test func disabledByDefault() {
        #expect(ShadowConfig.fromDefaults(makeDefaults()) == nil)
    }
    @Test func enabledUsesDefaults() {
        let d = makeDefaults(); d.set(true, forKey: "graniteShadowEnabled")
        let c = try! #require(ShadowConfig.fromDefaults(d))
        #expect(c.serverPath == "/opt/homebrew/bin/llama-server")
        #expect(c.port == 8873)
        #expect(c.modelDir.path.hasSuffix("Tome/Granite"))
        #expect(c.modelGGUF.lastPathComponent == "granite-speech-4.1-2b-Q8_0.gguf")
        #expect(c.mmprojGGUF.lastPathComponent == "mmproj-model-f16.gguf")
    }
    @Test func overridesRespectedAndTildeExpanded() {
        let d = makeDefaults()
        d.set(true, forKey: "graniteShadowEnabled")
        d.set("/usr/local/bin/llama-server", forKey: "graniteShadowServerPath")
        d.set("~/granite-models", forKey: "graniteShadowModelDir")
        d.set(9001, forKey: "graniteShadowPort")
        let c = try! #require(ShadowConfig.fromDefaults(d))
        #expect(c.serverPath == "/usr/local/bin/llama-server")
        #expect(c.port == 9001)
        #expect(!c.modelDir.path.contains("~"))
        #expect(c.modelDir.path.hasSuffix("/granite-models"))
    }
    @Test func filesPresentFalseOnEmptyDir() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let c = ShadowConfig(serverPath: "/x", modelDir: tmp, port: 1)
        #expect(!c.filesPresent())
    }
}
```

- [ ] **Step 2: Run → FAIL.** `swift test --filter ShadowConfigTests`
- [ ] **Step 3: Implement**

```swift
import Foundation

/// Hidden-flag configuration for granite shadow transcription. Read at job
/// creation (not app launch) so toggling applies from the next session.
/// Spec: docs/superpowers/specs/2026-07-09-granite-shadow-transcription-design.md
struct ShadowConfig: Sendable, Equatable {
    let serverPath: String
    let modelDir: URL
    let port: Int

    // Keep in sync with scripts/setup-granite-shadow.sh
    static let modelFilename = "granite-speech-4.1-2b-Q8_0.gguf"
    static let mmprojFilename = "mmproj-model-f16.gguf"

    var modelGGUF: URL { modelDir.appendingPathComponent(Self.modelFilename) }
    var mmprojGGUF: URL { modelDir.appendingPathComponent(Self.mmprojFilename) }
    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    func filesPresent(fileManager: FileManager = .default) -> Bool {
        fileManager.fileExists(atPath: modelGGUF.path) && fileManager.fileExists(atPath: mmprojGGUF.path)
    }

    static func fromDefaults(_ defaults: UserDefaults = .standard) -> ShadowConfig? {
        guard defaults.bool(forKey: "graniteShadowEnabled") else { return nil }
        let server = defaults.string(forKey: "graniteShadowServerPath") ?? "/opt/homebrew/bin/llama-server"
        let dir: URL
        if let override = defaults.string(forKey: "graniteShadowModelDir") {
            dir = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
        } else {
            dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Tome/Granite")
        }
        let port = (defaults.object(forKey: "graniteShadowPort") as? Int) ?? 8873
        return ShadowConfig(serverPath: server, modelDir: dir, port: port)
    }
}
```

- [ ] **Step 4: Run → PASS**, then full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: ShadowConfig (hidden granite shadow flag)"`

### Task 6: SegmentAudio extraction (merge/pad/read) + SegmentReTranscriber refactor

**Files:**
- Create: `Tome/Sources/Tome/Transcription/SegmentAudio.swift`
- Modify: `Tome/Sources/Tome/Transcription/SegmentReTranscriber.swift` (use the extracted functions; behavior byte-identical)
- Test: `Tome/Tests/TomeTests/SegmentAudioTests.swift`

**Interfaces:**
- Produces: `SegmentAudio.merge(_ segments: [DiarizedSegment], gapThreshold: Float = 0.5) -> [DiarizedSegment]`; `SegmentAudio.paddedFrameRange(startTime: Float, endTime: Float, sampleRate: Double, totalFrames: AVAudioFramePosition, minSeconds: Double = 1.5) -> (start: AVAudioFramePosition, count: AVAudioFrameCount)?`; `SegmentAudio.readSegment(file: AVAudioFile, start: AVAudioFramePosition, count: AVAudioFrameCount) -> AVAudioPCMBuffer?`. Both the primary path and Task 10's shadow runner use these — identical audio is the spec's apples-to-apples guarantee.

- [ ] **Step 1: Failing tests** pinning the EXACT current behavior (transcribed from SegmentReTranscriber.swift:25-56):

```swift
import AVFoundation
import Testing
@testable import Tome

@Suite struct SegmentAudioTests {
    @Test func mergesSameSpeakerWithinHalfSecond() {
        let segs = [
            DiarizedSegment(speakerId: "A", startTime: 0.0, endTime: 1.0),
            DiarizedSegment(speakerId: "A", startTime: 1.3, endTime: 2.0),   // gap 0.3 < 0.5 → merge
            DiarizedSegment(speakerId: "A", startTime: 2.6, endTime: 3.0),   // gap 0.6 ≥ 0.5 → new
            DiarizedSegment(speakerId: "B", startTime: 3.1, endTime: 4.0),   // speaker change → new
        ]
        let merged = SegmentAudio.merge(segs)
        #expect(merged.count == 3)
        #expect(merged[0].startTime == 0.0 && merged[0].endTime == 2.0)
        #expect(merged[1].startTime == 2.6 && merged[2].speakerId == "B")
    }
    @Test func padsShortSegmentCentered() {
        // 0.5 s segment at 16 kHz in a long file: deficit = 24000-8000 = 16000 → 8000 both sides
        let r = SegmentAudio.paddedFrameRange(startTime: 10, endTime: 10.5, sampleRate: 16000,
                                              totalFrames: 10_000_000)
        #expect(r! == (start: 152_000, count: 24_000))
    }
    @Test func padClampsAtFileStart() {
        // Segment at t=0: no room before, pad goes after
        let r = SegmentAudio.paddedFrameRange(startTime: 0, endTime: 0.5, sampleRate: 16000,
                                              totalFrames: 10_000_000)
        #expect(r! == (start: 0, count: 24_000))
    }
    @Test func zeroLengthSegmentIsNil() {
        #expect(SegmentAudio.paddedFrameRange(startTime: 5, endTime: 5, sampleRate: 16000,
                                              totalFrames: 80_000) == nil)
    }
}
```
IMPORTANT: before finalizing expected values, re-derive them from the current code (SegmentReTranscriber.swift:44-58) — the tests must encode what the code DOES today (e.g. clamp of `endFrame` to totalFrames, `frameCount > 0` guard returning nil).

- [ ] **Step 2: Run → FAIL** (`SegmentAudio` undefined).
- [ ] **Step 3: Implement** by MOVING the logic (not copying with edits):

```swift
import AVFoundation

/// Segment mechanics shared by the primary re-transcriber and the granite
/// shadow runner — both must see byte-identical audio (spec §4).
enum SegmentAudio {
    /// Merge consecutive same-speaker segments separated by < gapThreshold seconds.
    static func merge(_ segments: [DiarizedSegment], gapThreshold: Float = 0.5) -> [DiarizedSegment] {
        var merged: [DiarizedSegment] = []
        for seg in segments {
            if let last = merged.last, last.speakerId == seg.speakerId,
               seg.startTime - last.endTime < gapThreshold {
                merged[merged.count - 1] = DiarizedSegment(
                    speakerId: last.speakerId, startTime: last.startTime, endTime: seg.endTime)
            } else {
                merged.append(seg)
            }
        }
        return merged
    }

    /// Frame range for a segment, padded to minSeconds (Parakeet's floor —
    /// applied to all backends deliberately; see spec §4) and clamped to the file.
    static func paddedFrameRange(
        startTime: Float, endTime: Float, sampleRate: Double,
        totalFrames: AVAudioFramePosition, minSeconds: Double = 1.5
    ) -> (start: AVAudioFramePosition, count: AVAudioFrameCount)? {
        var startFrame = AVAudioFramePosition(Double(startTime) * sampleRate)
        var endFrame = min(AVAudioFramePosition(Double(endTime) * sampleRate), totalFrames)
        var frameCount = Int(endFrame - startFrame)
        let minSamples = Int(sampleRate * minSeconds)
        if frameCount < minSamples && frameCount > 0 {
            let deficit = minSamples - frameCount
            let padBefore = min(AVAudioFramePosition(deficit / 2), startFrame)
            let padAfter = min(deficit - Int(padBefore), Int(totalFrames - endFrame))
            startFrame -= padBefore
            endFrame += AVAudioFramePosition(padAfter)
            frameCount = Int(endFrame - startFrame)
        }
        guard frameCount > 0 else { return nil }
        return (startFrame, AVAudioFrameCount(frameCount))
    }

    /// Read one segment's PCM out of an open file. Nil on allocation/read failure.
    static func readSegment(file: AVAudioFile, start: AVAudioFramePosition,
                            count: AVAudioFrameCount) -> AVAudioPCMBuffer? {
        file.framePosition = start
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: count)
        else { return nil }
        do { try file.read(into: buffer, frameCount: count) } catch { return nil }
        return buffer
    }
}
```
Then rewrite `SegmentReTranscriber.run()`'s merge loop and frame math to call these three functions (delete the inlined versions). The `guard !text.isEmpty else continue` and `"[transcription failed]"` conventions stay exactly where they are.

- [ ] **Step 4: Run → PASS**; full suite green (any existing test touching SegmentReTranscriber must be untouched and green).
- [ ] **Step 5: Commit** — `git commit -am "refactor: extract SegmentAudio merge/pad/read (shared with granite shadow)"`

### Task 7: AudioWAVExport (buffer → 16 kHz mono PCM16 WAV bytes)

**Files:**
- Create: `Tome/Sources/Tome/Transcription/AudioWAVExport.swift`
- Test: `Tome/Tests/TomeTests/AudioWAVExportTests.swift`

**Interfaces:**
- Produces: `AudioWAVExport.wav16kMonoPCM16(from buffer: AVAudioPCMBuffer) throws -> Data`; `AudioWAVExport.riffHeader(dataByteCount: Int) -> Data`. Consumed by Task 10.

- [ ] **Step 1: Failing tests**

```swift
import AVFoundation
import Testing
@testable import Tome

@Suite struct AudioWAVExportTests {
    @Test func riffHeaderFields() {
        let h = AudioWAVExport.riffHeader(dataByteCount: 32000)
        #expect(h.count == 44)
        #expect(String(data: h[0..<4], encoding: .ascii) == "RIFF")
        #expect(String(data: h[8..<12], encoding: .ascii) == "WAVE")
        // chunk size = 36 + data
        #expect(h[4..<8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } == 32036)
        // sample rate 16000 @ offset 24, channels 1 @ 22, bits 16 @ 34
        #expect(h[24..<28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } == 16000)
        #expect(h[22..<24].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) } == 1)
        #expect(h[34..<36].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) } == 16)
    }
    @Test func convertsStereo48kToMono16k() throws {
        let fmt = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: 48000)!
        buf.frameLength = 48000  // 1 second of silence
        let data = try AudioWAVExport.wav16kMonoPCM16(from: buf)
        let samples = (data.count - 44) / 2
        #expect(abs(samples - 16000) < 64)   // ~1 s at 16 kHz (converter may prime ±)
    }
}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```swift
import AVFoundation

/// Converts arbitrary PCM buffers to the 16 kHz mono PCM16 WAV bytes the
/// granite sidecar consumes (granite_request.md pins format: "wav").
enum AudioWAVExport {
    enum ExportError: Error { case formatUnavailable, conversionFailed }

    static func wav16kMonoPCM16(from buffer: AVAudioPCMBuffer) throws -> Data {
        guard let outFmt = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000,
                                         channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: buffer.format, to: outFmt)
        else { throw ExportError.formatUnavailable }
        let ratio = 16000.0 / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 4096
        guard let out = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: capacity)
        else { throw ExportError.conversionFailed }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true; status.pointee = .haveData; return buffer
        }
        if let error { throw error }
        let byteCount = Int(out.frameLength) * 2
        var data = riffHeader(dataByteCount: byteCount)
        data.append(Data(bytes: out.int16ChannelData![0], count: byteCount))
        return data
    }

    static func riffHeader(dataByteCount: Int) -> Data {
        var d = Data()
        func le32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func le16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        d.append(contentsOf: "RIFF".utf8); le32(UInt32(36 + dataByteCount))
        d.append(contentsOf: "WAVE".utf8)
        d.append(contentsOf: "fmt ".utf8); le32(16); le16(1) /* PCM */; le16(1) /* mono */
        le32(16000); le32(16000 * 2) /* byte rate */; le16(2) /* block align */; le16(16)
        d.append(contentsOf: "data".utf8); le32(UInt32(dataByteCount))
        return d
    }
}
```

- [ ] **Step 4: Run → PASS**; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: AudioWAVExport (16 kHz mono PCM16 WAV for granite sidecar)"`

### Task 8: GraniteRequest (build/parse, golden-matched to granite_request.md)

**Files:**
- Create: `Tome/Sources/Tome/Transcription/GraniteRequest.swift`
- Test: `Tome/Tests/TomeTests/GraniteRequestTests.swift`

**Interfaces:**
- Consumes: the pinned template in `scripts/asr-bench/granite_request.md` (Task 2).
- Produces: `GraniteRequest.prompt: String`; `GraniteRequest.build(wavData: Data) -> Data` (JSON body); `GraniteRequest.parseResponse(_ data: Data) throws -> String`. Consumed by Task 9.

- [ ] **Step 1: Failing tests.** The golden test decodes the built body and asserts every field the template pins (adapt to what Task 2 actually recorded — the values below assume the OpenAI-compatible shape):

```swift
import Foundation
import Testing
@testable import Tome

@Suite struct GraniteRequestTests {
    @Test func buildMatchesPinnedTemplate() throws {
        // Golden contract: scripts/asr-bench/granite_request.md
        let wav = Data([0x52, 0x49, 0x46, 0x46])  // "RIFF"
        let body = try JSONSerialization.jsonObject(with: GraniteRequest.build(wavData: wav)) as! [String: Any]
        #expect(body["temperature"] as? Double == 0)
        #expect(body["max_tokens"] as? Int == 2048)
        #expect(body["stream"] as? Bool == false)
        let msgs = body["messages"] as! [[String: Any]]
        #expect(msgs.count == 1 && msgs[0]["role"] as? String == "user")
        let content = msgs[0]["content"] as! [[String: Any]]
        let audio = content[0]["input_audio"] as! [String: Any]
        #expect(audio["format"] as? String == "wav")
        #expect(audio["data"] as? String == wav.base64EncodedString())
        #expect(content[1]["text"] as? String == GraniteRequest.prompt)
        #expect(GraniteRequest.prompt == "can you transcribe the speech into a written format?")
    }
    @Test func parseExtractsContent() throws {
        let json = #"{"choices":[{"message":{"role":"assistant","content":"  hello world \n"}}]}"#
        #expect(try GraniteRequest.parseResponse(Data(json.utf8)) == "hello world")
    }
    @Test func parseThrowsOnMalformed() {
        #expect(throws: (any Error).self) {
            try GraniteRequest.parseResponse(Data(#"{"error":"boom"}"#.utf8))
        }
    }
}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```swift
import Foundation

/// Builds/parses granite llama-server requests. The contract is pinned in
/// scripts/asr-bench/granite_request.md — Phase 0 validated it; change both
/// together or not at all.
enum GraniteRequest {
    static let prompt = "can you transcribe the speech into a written format?"
    static let endpointPath = "/v1/chat/completions"

    enum ParseError: Error { case unexpectedShape }

    static func build(wavData: Data) -> Data {
        let body: [String: Any] = [
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "input_audio",
                     "input_audio": ["data": wavData.base64EncodedString(), "format": "wav"]],
                    ["type": "text", "text": prompt],
                ],
            ]],
            "temperature": 0, "max_tokens": 2048, "stream": false,
        ]
        return try! JSONSerialization.data(withJSONObject: body)
    }

    static func parseResponse(_ data: Data) throws -> String {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String
        else { throw ParseError.unexpectedShape }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
```

- [ ] **Step 4: Run → PASS**; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: GraniteRequest pinned to granite_request.md template"`

### Task 9: GraniteSidecar actor (process lifecycle state machine)

**Files:**
- Create: `Tome/Sources/Tome/Transcription/GraniteSidecar.swift`
- Test: `Tome/Tests/TomeTests/GraniteSidecarTests.swift` (+ fakes in the test file)

**Interfaces:**
- Consumes: `ShadowConfig` (Task 5), `GraniteRequest` (Task 8).
- Produces: `actor GraniteSidecar` — `init(config: ShadowConfig, launcher: any SidecarProcessLauncher = DefaultProcessLauncher(), http: any SidecarHTTP = URLSessionSidecarHTTP(), readyTimeout: TimeInterval = 60, sleep: @Sendable (TimeInterval) async -> Void = { try? await Task.sleep(for: .seconds($0)) })`; `func start() async -> Bool`; `func transcribe(wavData: Data) async throws -> String`; `func stop() async`. Protocols: `SidecarProcess` (`isRunning: Bool`, `terminate()`, `forceKill()`), `SidecarProcessLauncher` (`launch(executable: URL, arguments: [String]) throws -> any SidecarProcess`), `SidecarHTTP` (`healthStatus(_ url: URL) async -> Int?`, `post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data`). Consumed by Task 10.

- [ ] **Step 1: Failing tests** (fakes included; follow FakeBackend's style):

```swift
import Foundation
import Testing
@testable import Tome

final class FakeProcess: SidecarProcess, @unchecked Sendable {
    var running = true
    var terminated = false, killed = false
    var isRunning: Bool { running }
    func terminate() { terminated = true; running = false }
    func forceKill() { killed = true; running = false }
}

final class FakeLauncher: SidecarProcessLauncher, @unchecked Sendable {
    var launched: [(URL, [String])] = []
    var processes: [FakeProcess] = []
    var launchError: (any Error)?
    func launch(executable: URL, arguments: [String]) throws -> any SidecarProcess {
        if let launchError { throw launchError }
        launched.append((executable, arguments))
        let p = FakeProcess(); processes.append(p); return p
    }
}

final class FakeHTTP: SidecarHTTP, @unchecked Sendable {
    var healthResults: [Int?] = [200]
    var postResults: [Result<Data, any Error>] = []
    func healthStatus(_ url: URL) async -> Int? {
        healthResults.isEmpty ? 200 : healthResults.removeFirst()
    }
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data {
        try postResults.removeFirst().get()
    }
}

private func makeSidecar(launcher: FakeLauncher = FakeLauncher(), http: FakeHTTP = FakeHTTP())
    -> (GraniteSidecar, FakeLauncher, FakeHTTP) {
    let config = ShadowConfig(serverPath: "/fake/llama-server",
                              modelDir: URL(fileURLWithPath: "/fake/models"), port: 9999)
    let s = GraniteSidecar(config: config, launcher: launcher, http: http,
                           readyTimeout: 1, sleep: { _ in })
    return (s, launcher, http)
}

private let ok = Data(#"{"choices":[{"message":{"content":"hi"}}]}"#.utf8)
private struct ConnErr: Error {}

@Suite struct GraniteSidecarTests {
    @Test func startLaunchesWithConfigArgsAndPollsHealth() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [503, 200]
        #expect(await s.start())
        let (exe, args) = launcher.launched[0]
        #expect(exe.path == "/fake/llama-server")
        #expect(args.contains("--port") && args.contains("9999") && args.contains("127.0.0.1"))
        #expect(args.contains("/fake/models/\(ShadowConfig.modelFilename)"))
    }
    @Test func startFailsAfterTimeoutAndKills() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = Array(repeating: 503 as Int?, count: 500)
        #expect(await s.start() == false)
        #expect(launcher.processes[0].terminated || launcher.processes[0].killed)
    }
    @Test func transcribeSendsRequestAndParses() async throws {
        let (s, _, http) = makeSidecar()
        http.postResults = [.success(ok)]
        _ = await s.start()
        #expect(try await s.transcribe(wavData: Data([1])) == "hi")
    }
    @Test func connectionFailureRelaunchesOnceThenFails() async {
        let (s, launcher, http) = makeSidecar()
        http.postResults = [.failure(ConnErr()), .failure(ConnErr())]
        _ = await s.start()
        await #expect(throws: (any Error).self) { try await s.transcribe(wavData: Data([1])) }
        #expect(launcher.launched.count == 2)   // original + one relaunch
        // subsequent calls fail fast without further launches
        await #expect(throws: (any Error).self) { try await s.transcribe(wavData: Data([1])) }
        #expect(launcher.launched.count == 2)
    }
    @Test func stopTerminatesProcess() async {
        let (s, launcher, _) = makeSidecar()
        _ = await s.start()
        await s.stop()
        #expect(launcher.processes[0].terminated)
    }
}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```swift
import Foundation

protocol SidecarProcess: Sendable {
    var isRunning: Bool { get }
    func terminate()
    func forceKill()
}

protocol SidecarProcessLauncher: Sendable {
    func launch(executable: URL, arguments: [String]) throws -> any SidecarProcess
}

protocol SidecarHTTP: Sendable {
    func healthStatus(_ url: URL) async -> Int?
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data
}

/// Owns one llama-server child process, spawn-per-job (spec §3): ~4 GB of
/// model RAM stays off the machine between jobs. One relaunch on connection
/// failure; a second failure fails the phase.
actor GraniteSidecar {
    enum State: Equatable { case idle, ready, failed }
    enum SidecarError: Error { case notReady, requestFailed }

    private let config: ShadowConfig
    private let launcher: any SidecarProcessLauncher
    private let http: any SidecarHTTP
    private let readyTimeout: TimeInterval
    private let sleep: @Sendable (TimeInterval) async -> Void
    private var process: (any SidecarProcess)?
    private var didRelaunch = false
    private(set) var state: State = .idle

    init(config: ShadowConfig,
         launcher: any SidecarProcessLauncher = DefaultProcessLauncher(),
         http: any SidecarHTTP = URLSessionSidecarHTTP(),
         readyTimeout: TimeInterval = 60,
         sleep: @Sendable @escaping (TimeInterval) async -> Void = { try? await Task.sleep(for: .seconds($0)) }) {
        self.config = config
        self.launcher = launcher
        self.http = http
        self.readyTimeout = readyTimeout
        self.sleep = sleep
    }

    @discardableResult
    func start() async -> Bool {
        do {
            process = try launcher.launch(
                executable: URL(fileURLWithPath: config.serverPath),
                arguments: ["-m", config.modelGGUF.path,
                            "--mmproj", config.mmprojGGUF.path,
                            "--host", "127.0.0.1",
                            "--port", String(config.port)])
        } catch {
            diagLog("[SHADOW] sidecar launch failed: \(error)")
            state = .failed
            return false
        }
        let deadline = readyTimeout / 0.5
        for _ in 0..<Int(deadline) {
            if await http.healthStatus(config.baseURL.appendingPathComponent("health")) == 200 {
                state = .ready
                return true
            }
            await sleep(0.5)
        }
        diagLog("[SHADOW] sidecar not healthy within \(readyTimeout)s — killing")
        endProcess()
        state = .failed
        return false
    }

    func transcribe(wavData: Data) async throws -> String {
        guard state == .ready else { throw SidecarError.notReady }
        let url = config.baseURL.appendingPathComponent(
            GraniteRequest.endpointPath.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
        let body = GraniteRequest.build(wavData: wavData)
        do {
            return try GraniteRequest.parseResponse(try await http.post(url, body: body, timeout: 600))
        } catch {
            guard !didRelaunch else {
                diagLog("[SHADOW] request failed after relaunch — failing sidecar: \(error)")
                endProcess()
                state = .failed
                throw SidecarError.requestFailed
            }
            diagLog("[SHADOW] request failed (\(error)) — relaunching sidecar once")
            didRelaunch = true
            endProcess()
            guard await start() else { throw SidecarError.requestFailed }
            return try GraniteRequest.parseResponse(try await http.post(url, body: body, timeout: 600))
        }
    }

    func stop() async {
        endProcess()
        state = .idle
    }

    private func endProcess() {
        guard let p = process else { return }
        p.terminate()
        // Escalation handled synchronously in DefaultProcessLauncher's process
        // wrapper (terminate → 5 s grace in a detached task → forceKill).
        if p.isRunning { p.forceKill() }
        process = nil
    }
}

// MARK: - Real implementations

struct DefaultProcessLauncher: SidecarProcessLauncher {
    func launch(executable: URL, arguments: [String]) throws -> any SidecarProcess {
        let p = Process()
        p.executableURL = executable
        p.arguments = arguments
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        try p.run()
        return RealSidecarProcess(process: p)
    }
}

/// Wraps Process; forceKill sends SIGKILL. A leaked llama-server must not
/// outlive Tome: Process children die with the parent only if killed, so
/// terminationHandler is not enough — the shadow phase's defer + this
/// wrapper's deinit both call terminate.
final class RealSidecarProcess: SidecarProcess, @unchecked Sendable {
    private let process: Process
    init(process: Process) { self.process = process }
    var isRunning: Bool { process.isRunning }
    func terminate() { if process.isRunning { process.terminate() } }
    func forceKill() { if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
    deinit { if process.isRunning { process.terminate() } }
}

struct URLSessionSidecarHTTP: SidecarHTTP {
    func healthStatus(_ url: URL) async -> Int? {
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        guard let (_, resp) = try? await URLSession.shared.data(for: req) else { return nil }
        return (resp as? HTTPURLResponse)?.statusCode
    }
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.httpBody = body
        req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, _) = try await URLSession.shared.data(for: req)
        return data
    }
}
```
Note: localhost URLSession is fine — the known HF-CDN URLSession issue is remote-CDN-specific; downloads still use curl.

- [ ] **Step 4: Run → PASS**; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: GraniteSidecar actor (spawn-per-job llama-server lifecycle)"`

### Task 10: Shadow runner + artifacts

**Files:**
- Create: `Tome/Sources/Tome/Transcription/GraniteShadow.swift` (SegmentTranscribing, ShadowRunner, artifact builders, GraniteShadowPhase)
- Test: `Tome/Tests/TomeTests/GraniteShadowTests.swift`

**Interfaces:**
- Consumes: `SegmentAudio` (Task 6), `AudioWAVExport` (Task 7), `GraniteSidecar` (Task 9), `ShadowConfig` (Task 5), `DiarizedSegment`/`ReTranscribedSegment` (existing).
- Produces: `protocol SegmentTranscribing: Sendable { func transcribe(buffer: AVAudioPCMBuffer) async throws -> String }`; `struct GraniteSidecarTranscriber: SegmentTranscribing` (wraps sidecar via AudioWAVExport); `ShadowRunner.run(fileURL: URL, diarSegments: [DiarizedSegment], speakerNumberBase: Int) async -> ShadowRunOutput` where `ShadowRunOutput(segments: [ShadowSegment], incomplete: Bool)` and `ShadowSegment(startTime: Float, speaker: String, durationSec: Double, text: String?, error: String?, latencySec: Double)`; `ShadowArtifacts.write(session: ShadowSessionInfo, primary: [ReTranscribedSegment], shadow: ShadowRunOutput, to dir: URL) throws -> (md: URL, json: URL)` with `ShadowSessionInfo(sessionID: String, transcriptPath: String, sessionType: String, primaryModel: String, graniteModel: String)`; `GraniteShadowPhase.run(config:bufferURL:diarSegments:speakerNumberBase:primary:session:) async` (never throws). `GraniteShadowPhase.shouldRun(config: ShadowConfig?, didRebuild: Bool, primary: [ReTranscribedSegment]?) -> Bool` (pure policy). Task 11 consumes `GraniteShadowPhase`.

- [ ] **Step 1: Failing tests** — policy, runner (fake transcriber), pairing, artifacts:

```swift
import AVFoundation
import Foundation
import Testing
@testable import Tome

final class FakeSegmentTranscriber: SegmentTranscribing, @unchecked Sendable {
    var results: [Result<String, any Error>]
    init(_ results: [Result<String, any Error>]) { self.results = results }
    func transcribe(buffer: AVAudioPCMBuffer) async throws -> String {
        try results.removeFirst().get()
    }
}
private struct Boom: Error {}

@Suite struct GraniteShadowTests {
    // -- policy --
    @Test func shouldRunRequiresConfigRebuildAndResults() {
        let cfg = ShadowConfig(serverPath: "/x", modelDir: URL(fileURLWithPath: "/x"), port: 1)
        let seg = [ReTranscribedSegment(speaker: "Speaker 2", text: "hi", startTime: 0)]
        #expect(GraniteShadowPhase.shouldRun(config: cfg, didRebuild: true, primary: seg))
        #expect(!GraniteShadowPhase.shouldRun(config: nil, didRebuild: true, primary: seg))
        #expect(!GraniteShadowPhase.shouldRun(config: cfg, didRebuild: false, primary: seg))
        #expect(!GraniteShadowPhase.shouldRun(config: cfg, didRebuild: true, primary: nil))
        #expect(!GraniteShadowPhase.shouldRun(config: cfg, didRebuild: true, primary: []))
    }
    // -- runner: uses a real tiny WAV fixture so SegmentAudio paths execute --
    private func fixtureWAV() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("shadow-\(UUID().uuidString).wav")
        let fmt = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
        let file = try AVAudioFile(forWriting: url, settings: fmt.settings)
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: 16000 * 10)!
        buf.frameLength = 16000 * 10   // 10 s silence
        try file.write(from: buf)
        return url
    }
    @Test func runnerProducesResultPerMergedSegmentIncludingErrors() async throws {
        let wav = try fixtureWAV()
        let segs = [DiarizedSegment(speakerId: "S0", startTime: 0.0, endTime: 2.0),
                    DiarizedSegment(speakerId: "S1", startTime: 3.0, endTime: 5.0)]
        let runner = ShadowRunner(transcriber: FakeSegmentTranscriber([.success("hello"), .failure(Boom())]))
        let out = await runner.run(fileURL: wav, diarSegments: segs, speakerNumberBase: 2)
        #expect(out.segments.count == 2)
        #expect(out.segments[0].text == "hello" && out.segments[0].error == nil)
        #expect(out.segments[1].text == nil && out.segments[1].error != nil)
        #expect(!out.incomplete)
    }
    // -- pairing + artifacts --
    @Test func artifactsPairByStartTimeAndHandleMissingPrimary() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let session = ShadowSessionInfo(sessionID: "s1", transcriptPath: "/t.md",
                                        sessionType: "callCapture",
                                        primaryModel: "Parakeet-TDT v3",
                                        graniteModel: "granite-speech-4.1-2b-Q8_0")
        // primary skipped the 3.0 segment (empty text) — granite has it
        let primary = [ReTranscribedSegment(speaker: "Speaker 2", text: "hi there", startTime: 0.0)]
        let shadow = ShadowRunOutput(segments: [
            ShadowSegment(startTime: 0.0, speaker: "Speaker 2", durationSec: 2, text: "hi there friend", error: nil, latencySec: 0.5),
            ShadowSegment(startTime: 3.0, speaker: "Speaker 3", durationSec: 2, text: "quarterly numbers", error: nil, latencySec: 0.4),
        ], incomplete: false)
        let (md, json) = try ShadowArtifacts.write(session: session, primary: primary, shadow: shadow, to: dir)
        let comparison = try JSONDecoder().decode(ShadowComparison.self, from: Data(contentsOf: json))
        #expect(comparison.segments.count == 2)
        #expect(comparison.segments[0].primaryText == "hi there")
        #expect(comparison.segments[1].primaryText == "")          // "" for missing side (spec §4)
        #expect(comparison.segments[1].graniteText == "quarterly numbers")
        #expect(comparison.totals.segmentCount == 2 && comparison.totals.erroredCount == 0)
        let mdText = try String(contentsOf: md, encoding: .utf8)
        #expect(mdText.contains("Speaker 3: quarterly numbers"))
        #expect(mdText.contains("granite-speech-4.1-2b-Q8_0"))
    }
}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** in `GraniteShadow.swift`:

```swift
import AVFoundation
import Foundation

protocol SegmentTranscribing: Sendable {
    func transcribe(buffer: AVAudioPCMBuffer) async throws -> String
}

/// Bridges the sidecar into the per-segment loop: buffer → 16 kHz WAV → HTTP.
struct GraniteSidecarTranscriber: SegmentTranscribing {
    let sidecar: GraniteSidecar
    func transcribe(buffer: AVAudioPCMBuffer) async throws -> String {
        try await sidecar.transcribe(wavData: try AudioWAVExport.wav16kMonoPCM16(from: buffer))
    }
}

struct ShadowSegment: Codable, Sendable, Equatable {
    let startTime: Float
    let speaker: String
    let durationSec: Double
    let text: String?
    let error: String?
    let latencySec: Double
}

struct ShadowRunOutput: Sendable {
    let segments: [ShadowSegment]
    let incomplete: Bool
}

/// Runs granite over the SAME merged segments the primary path used
/// (SegmentAudio guarantees identical audio — spec §4). Unlike
/// SegmentReTranscriber, errors are recorded per segment, not placeholdered.
struct ShadowRunner: Sendable {
    let transcriber: any SegmentTranscribing

    func run(fileURL: URL, diarSegments: [DiarizedSegment], speakerNumberBase: Int) async -> ShadowRunOutput {
        let audioFile: AVAudioFile
        do { audioFile = try AVAudioFile(forReading: fileURL) } catch {
            diagLog("[SHADOW] cannot open \(fileURL.lastPathComponent): \(error)")
            return ShadowRunOutput(segments: [], incomplete: true)
        }
        let sampleRate = audioFile.processingFormat.sampleRate
        let totalFrames = AVAudioFramePosition(audioFile.length)
        let merged = SegmentAudio.merge(diarSegments)
        let speakerMap = speakerLabels(from: merged.map(\.speakerId), startingAt: speakerNumberBase)
        var results: [ShadowSegment] = []
        var incomplete = false
        let clock = ContinuousClock()
        for seg in merged {
            if Task.isCancelled { incomplete = true; break }
            let speaker = speakerMap[seg.speakerId] ?? "Speaker \(speakerNumberBase)"
            let duration = Double(seg.endTime - seg.startTime)
            guard let range = SegmentAudio.paddedFrameRange(
                      startTime: seg.startTime, endTime: seg.endTime,
                      sampleRate: sampleRate, totalFrames: totalFrames),
                  let buffer = SegmentAudio.readSegment(file: audioFile, start: range.start, count: range.count)
            else {
                results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                             durationSec: duration, text: nil,
                                             error: "segment read failed", latencySec: 0))
                continue
            }
            let t0 = clock.now
            do {
                let text = try await transcriber.transcribe(buffer: buffer)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                             durationSec: duration, text: text, error: nil,
                                             latencySec: Double(truncating: (clock.now - t0) / .seconds(1) as NSNumber)))
            } catch {
                results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                             durationSec: duration, text: nil,
                                             error: String(describing: error),
                                             latencySec: Double(truncating: (clock.now - t0) / .seconds(1) as NSNumber)))
                if error is GraniteSidecar.SidecarError, case GraniteSidecar.SidecarError.notReady = error {
                    incomplete = true; break   // sidecar dead — stop burning segments
                }
            }
        }
        return ShadowRunOutput(segments: results, incomplete: incomplete)
    }
}
```
(If `speakerLabels(from:startingAt:)` is private to SegmentReTranscriber's file, make it internal — it's already Tome-module-internal logic. `Duration`→seconds: use `Double(components.seconds) + Double(components.attoseconds) * 1e-18` if the NSNumber cast doesn't compile.)

```swift
struct ShadowSessionInfo: Codable, Sendable {
    let sessionID: String
    let transcriptPath: String
    let sessionType: String
    let primaryModel: String
    let graniteModel: String
}

struct ShadowComparisonSegment: Codable, Sendable {
    let startTime: Float
    let speaker: String
    let durationSec: Double
    let primaryText: String
    let graniteText: String
    let graniteError: String?
    let graniteLatencySec: Double
}

struct ShadowComparisonTotals: Codable, Sendable {
    let segmentCount: Int
    let erroredCount: Int
    let audioSeconds: Double
    let shadowWallClockSec: Double
    let rtf: Double
}

struct ShadowComparison: Codable, Sendable {
    let session: ShadowSessionInfo
    let incomplete: Bool
    let segments: [ShadowComparisonSegment]
    let totals: ShadowComparisonTotals
}

enum ShadowArtifacts {
    static func defaultDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Tome/GraniteShadow")
    }

    static func write(session: ShadowSessionInfo, primary: [ReTranscribedSegment],
                      shadow: ShadowRunOutput, to dir: URL) throws -> (md: URL, json: URL) {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Pair by merged-segment startTime (spec §4: primary skips empty-text
        // segments, so array positions don't line up; "" marks a missing side).
        let primaryByStart = Dictionary(primary.map { ($0.startTime, $0.text) },
                                        uniquingKeysWith: { a, _ in a })
        let segments = shadow.segments.map { s in
            ShadowComparisonSegment(startTime: s.startTime, speaker: s.speaker,
                                    durationSec: s.durationSec,
                                    primaryText: primaryByStart[s.startTime] ?? "",
                                    graniteText: s.text ?? "",
                                    graniteError: s.error,
                                    graniteLatencySec: s.latencySec)
        }
        let audioSeconds = shadow.segments.reduce(0) { $0 + $1.durationSec }
        let wall = shadow.segments.reduce(0) { $0 + $1.latencySec }
        let comparison = ShadowComparison(
            session: session, incomplete: shadow.incomplete, segments: segments,
            totals: ShadowComparisonTotals(segmentCount: segments.count,
                                           erroredCount: shadow.segments.filter { $0.error != nil }.count,
                                           audioSeconds: audioSeconds,
                                           shadowWallClockSec: wall,
                                           rtf: audioSeconds > 0 ? wall / audioSeconds : 0))
        let jsonURL = dir.appendingPathComponent("\(session.sessionID).comparison.json")
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(comparison).write(to: jsonURL, options: .atomic)

        var md = """
        # Granite shadow transcript — \(session.sessionID)
        - Primary model: \(session.primaryModel)
        - Shadow model: \(session.graniteModel)
        - Segments: \(segments.count) (\(comparison.totals.erroredCount) errored)\(shadow.incomplete ? " — INCOMPLETE" : "")
        - Shadow RTF: \(String(format: "%.3f", comparison.totals.rtf))

        """
        for s in shadow.segments where s.text?.isEmpty == false {
            md += "\(s.speaker): \(s.text!)\n\n"
        }
        let mdURL = dir.appendingPathComponent("\(session.sessionID).granite.md")
        try md.write(to: mdURL, atomically: true, encoding: .utf8)
        return (mdURL, jsonURL)
    }
}

/// Best-effort orchestration — the ONLY entry point PostProcessingJob calls.
/// Never throws; every failure is a diagLog + recorded artifact state.
enum GraniteShadowPhase {
    static func shouldRun(config: ShadowConfig?, didRebuild: Bool,
                          primary: [ReTranscribedSegment]?) -> Bool {
        guard let config, didRebuild, let primary, !primary.isEmpty else { return false }
        _ = config
        return true
    }

    static func run(config: ShadowConfig, bufferURL: URL, diarSegments: [DiarizedSegment],
                    speakerNumberBase: Int, primary: [ReTranscribedSegment],
                    session: ShadowSessionInfo,
                    outputDir: URL = ShadowArtifacts.defaultDirectory(),
                    sidecar: GraniteSidecar? = nil) async {
        guard FileManager.default.isExecutableFile(atPath: config.serverPath) else {
            diagLog("[SHADOW] llama-server missing at \(config.serverPath) — skipping (run scripts/setup-granite-shadow.sh)")
            return
        }
        guard config.filesPresent() else {
            diagLog("[SHADOW] model files missing in \(config.modelDir.path) — skipping (run scripts/setup-granite-shadow.sh)")
            return
        }
        let sc = sidecar ?? GraniteSidecar(config: config)
        diagLog("[SHADOW] starting sidecar for \(session.sessionID) (\(diarSegments.count) diar segments)")
        guard await sc.start() else {
            diagLog("[SHADOW] sidecar failed to start — skipping session \(session.sessionID)")
            return
        }
        let output = await ShadowRunner(transcriber: GraniteSidecarTranscriber(sidecar: sc))
            .run(fileURL: bufferURL, diarSegments: diarSegments, speakerNumberBase: speakerNumberBase)
        await sc.stop()
        do {
            let (md, json) = try ShadowArtifacts.write(session: session, primary: primary,
                                                       shadow: output, to: outputDir)
            diagLog("[SHADOW] wrote \(md.lastPathComponent) + \(json.lastPathComponent) (\(output.segments.count) segments, incomplete=\(output.incomplete))")
        } catch {
            diagLog("[SHADOW] artifact write failed (non-fatal): \(error)")
        }
    }
}
```

- [ ] **Step 4: Run → PASS**; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: granite shadow runner, artifacts, phase orchestration"`

### Task 11: PostProcessingJob wiring

**Files:**
- Modify: `Tome/Sources/Tome/Transcription/PostProcessingJob.swift`
- Test: extend `Tome/Tests/TomeTests/GraniteShadowTests.swift` (policy coverage already there; this task's safety is the no-behavior-change property of the full suite)

**Interfaces:**
- Consumes: `GraniteShadowPhase` (Task 10), `ShadowConfig` (Task 5).
- Produces: shadow runs on real sessions when flag on. NO signature changes visible to callers (default parameter).

- [ ] **Step 1: Modify `PostProcessingJob`:**
  1. Add stored property + init parameter with default (evaluated at job creation — exactly the spec's "read at job creation" semantics, zero call-site changes):
     ```swift
     let shadowConfig: ShadowConfig?
     init(handle: SessionHandle, clusterThreshold: Float, numberOfSpeakers: Int,
          retention: RecordingRetentionConfig? = nil, exportVoiceprints: Bool = false,
          shadowConfig: ShadowConfig? = ShadowConfig.fromDefaults()) {
         ...existing assignments...
         self.shadowConfig = shadowConfig
     }
     ```
  2. In `run(using:)`, hoist the re-transcription results so the shadow phase can see them: before the `if let bufferURL = diarBufferURL {` block add `var primaryResults: [ReTranscribedSegment]? = nil`; inside, where `let results = await TranscriptionEngine.reTranscribe(...)` is assigned, add `primaryResults = results`. Keep `didRebuildSpeakers` as the `didRebuild` signal.
  3. Insert the shadow phase AFTER the voiceprint block (after the `if exportVoiceprints { ... }` closing brace) and BEFORE the retention comment `// 3. Retain the combined recording…`:
     ```swift
     // 2c. Granite shadow transcription (hidden flag; spec 2026-07-09).
     //     Best-effort and additive: runs while the capture WAVs still exist,
     //     never throws, never touches the primary transcript or cleanup.
     if GraniteShadowPhase.shouldRun(config: shadowConfig, didRebuild: didRebuildSpeakers,
                                     primary: primaryResults),
        let bufferURL = diarBufferURL, let diar = diarOutput {
         let speakerBase = handle.sessionType == .callCapture ? 2 : 1
         await GraniteShadowPhase.run(
             config: shadowConfig!, bufferURL: bufferURL, diarSegments: diar.segments,
             speakerNumberBase: speakerBase, primary: primaryResults!,
             session: ShadowSessionInfo(
                 sessionID: id,
                 transcriptPath: savedPath.path,
                 sessionType: String(describing: handle.sessionType),
                 primaryModel: await asr.activeModel?.displayName ?? "unknown",
                 graniteModel: ShadowConfig.modelFilename))
     }
     ```
     Note `speakerBase` re-derivation must match the switch at the top of `run` (callCapture → 2, voiceMemo → 1) — or better, hoist the existing `speakerBase` local so it's in scope here (it already is: it's declared before the diarization block — verify and reuse it instead of re-deriving).
- [ ] **Step 2: Full suite** — `swift test`: everything green (no existing test constructs PostProcessingJob; the default parameter keeps call sites source-compatible — verify with `swift build`).
- [ ] **Step 3: Flag-off no-op check** — `grep -n "GraniteShadowPhase\|shadowConfig" Tome/Sources/Tome/Transcription/PostProcessingJob.swift`: the ONLY behavioral entry is guarded by `shouldRun`, which requires a non-nil config, which requires `graniteShadowEnabled=true`.
- [ ] **Step 4: Commit** — `git commit -am "feat: wire granite shadow phase into PostProcessingJob (hidden flag)"`

### Task 12: Shadow comparison report script

**Files:**
- Create: `scripts/granite-shadow-report.py` (stdlib only)
- Create: `scripts/tests/test_shadow_report.py` (stdlib unittest + fixture inline)

**Interfaces:**
- Consumes: `*.comparison.json` files (Task 10's `ShadowComparison` schema).
- Produces: `report.html` — aggregate stats + per-session side-by-side with word-diff highlighting, highest-disagreement first.

- [ ] **Step 1: Failing test**

```python
import json, pathlib, subprocess, sys, tempfile, unittest

SAMPLE = {
    "session": {"sessionID": "s1", "transcriptPath": "/t.md", "sessionType": "callCapture",
                "primaryModel": "Parakeet-TDT v3", "graniteModel": "granite-q8"},
    "incomplete": False,
    "segments": [
        {"startTime": 0.0, "speaker": "Speaker 2", "durationSec": 2.0,
         "primaryText": "the quarterly numbers look grim", "graniteText": "the quarterly numbers look green",
         "graniteError": None, "graniteLatencySec": 0.4},
        {"startTime": 3.0, "speaker": "Speaker 2", "durationSec": 1.5,
         "primaryText": "same words", "graniteText": "same words",
         "graniteError": None, "graniteLatencySec": 0.2},
    ],
    "totals": {"segmentCount": 2, "erroredCount": 0, "audioSeconds": 3.5,
               "shadowWallClockSec": 0.6, "rtf": 0.171},
}

class ReportTest(unittest.TestCase):
    def test_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "s1.comparison.json").write_text(json.dumps(SAMPLE))
            out = d / "report.html"
            script = pathlib.Path(__file__).parent.parent / "granite-shadow-report.py"
            subprocess.run([sys.executable, str(script), str(d), "-o", str(out)], check=True)
            html = out.read_text()
            self.assertIn("grim", html)            # disagreement segment present
            self.assertIn("green", html)
            self.assertIn("0.171", html)           # RTF surfaced
            # disagreeing segment sorted before identical one
            self.assertLess(html.index("grim"), html.index("same words"))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run → FAIL** (`python3 scripts/tests/test_shadow_report.py`).
- [ ] **Step 3: Implement** (stdlib: `json`, `pathlib`, `difflib`, `html`, `argparse`):

```python
#!/usr/bin/env python3
"""Render granite shadow comparison JSONs into one side-by-side HTML report.
Usage: python3 granite-shadow-report.py "~/Library/Application Support/Tome/GraniteShadow" [-o report.html]
Stdlib only (spec §7)."""
import argparse, difflib, html, json, pathlib

def word_diff(a: str, b: str) -> tuple[float, str, str]:
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw)
    left, right = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        at, bt = " ".join(aw[i1:i2]), " ".join(bw[j1:j2])
        if op == "equal":
            left.append(html.escape(at)); right.append(html.escape(bt))
        else:
            if at: left.append(f"<mark>{html.escape(at)}</mark>")
            if bt: right.append(f"<mark>{html.escape(bt)}</mark>")
    return 1 - sm.ratio(), " ".join(left), " ".join(right)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("report.html"))
    args = ap.parse_args()
    sessions = [json.loads(p.read_text())
                for p in sorted(args.dir.expanduser().glob("*.comparison.json"))]
    rows, agg = [], {"sessions": len(sessions), "segments": 0, "errored": 0,
                     "audio": 0.0, "wall": 0.0, "disagree": 0}
    for s in sessions:
        agg["segments"] += s["totals"]["segmentCount"]; agg["errored"] += s["totals"]["erroredCount"]
        agg["audio"] += s["totals"]["audioSeconds"]; agg["wall"] += s["totals"]["shadowWallClockSec"]
        for seg in s["segments"]:
            score, lh, rh = word_diff(seg["primaryText"], seg["graniteText"])
            if score > 0.05: agg["disagree"] += 1
            rows.append((score, s["session"]["sessionID"], s["session"]["primaryModel"],
                         s["totals"]["rtf"], seg, lh, rh))
    rows.sort(key=lambda r: -r[0])
    rtf = agg["wall"] / agg["audio"] if agg["audio"] else 0
    body = [f"<h1>Granite shadow report</h1>",
            f"<p>{agg['sessions']} sessions · {agg['segments']} segments · "
            f"{agg['disagree']} disagreeing (&gt;5% word diff) · {agg['errored']} errored · "
            f"aggregate shadow RTF {rtf:.3f}</p>",
            "<table border=1 cellpadding=6 style='border-collapse:collapse;font-family:sans-serif'>",
            "<tr><th>diff</th><th>session</th><th>t</th><th>primary</th><th>granite</th></tr>"]
    for score, sid, pmodel, srtf, seg, lh, rh in rows:
        body.append(f"<tr><td>{score:.2f}</td><td>{html.escape(sid)}<br><small>{html.escape(pmodel)}"
                    f" · RTF {srtf:.3f}</small></td><td>{seg['startTime']:.0f}s</td>"
                    f"<td>{lh}</td><td>{rh}</td></tr>")
    body.append("</table>")
    args.out.write_text("\n".join(body))
    print(f"wrote {args.out} ({agg['segments']} segments)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat: granite shadow HTML comparison report (stdlib)"`

### Task 13: Live smoke + install + enable (GATED on Task 4 go)

**Files:**
- None created (operational task); update `docs/superpowers/plans/2026-07-09-granite-phase0-results.md` with the live-smoke note.

- [ ] **Step 1: GATE** — read the Phase 0 results doc. Proceed to enabling ONLY on "go" (all three Phase 0 gates). On "no-go": still complete Steps 2–3 with the flag OFF (code is inert), report to Nic, stop.
- [ ] **Step 2: Full verification** — `DEVELOPER_DIR=… swift test` (all green) and `swift build -c release`.
- [ ] **Step 3: Build + install the app the same way the whisper-v3-turbo work did its smoke restoration** (check `scripts/` for the packaging script used then; reuse it exactly). Install to the location Nic's running copy lives.
- [ ] **Step 4: Enable + live smoke** — `defaults write com.dloomis.tome graniteShadowEnabled -bool YES`; launch Tome; record a ~60 s voice memo with 2 speakers (or play a meeting clip); stop; wait for post-processing; verify:
  - primary transcript identical in structure to a flag-off run (spot-check),
  - `~/Library/Application Support/Tome/GraniteShadow/<session>.granite.md` + `.comparison.json` exist and pair correctly,
  - `python3 scripts/granite-shadow-report.py "~/Library/Application Support/Tome/GraniteShadow" -o /tmp/report.html` renders,
  - no llama-server process survives app quit (`pgrep llama-server`).
- [ ] **Step 5: Record results** in the Phase 0 results doc (live-smoke section: RTF observed, artifacts OK); commit — `git commit -am "test: granite shadow live smoke on real session"`.
- [ ] **Step 6: Notify Nic**: shadow is live for Friday's meetings; review lands EOD Wednesday 2026-07-15 via the report script.

---

## Self-Review Notes

- **Spec coverage:** §0→Task 4 (+2 for template, +3 for manifest mode, TED-LIUM substitution documented in-task); §1→Task 5; §2→Task 1; §3→Task 9; §4→Tasks 6+10 (pairing-by-startTime in ShadowArtifacts); §5→Task 11 (placement after voiceprints, before retention; cancellation via Task.isCancelled in ShadowRunner; `.finalizing` retained — no new Phase case); §6→Task 10; §7→Task 12; §8 testing→each task's test steps. Spec's "WAVs still present when the phase runs" ordering test is downgraded to a live-smoke check (Task 13) — a job-level test harness would need SessionHandle fixtures that don't exist; documented deviation.
- **Deviation from spec §4 wording:** SegmentReTranscriber keeps its `ASRCoordinator` directly; the `SegmentTranscribing` seam lives on the shadow side, and identical audio is guaranteed by the shared `SegmentAudio` functions instead. Same intent (identical inputs + testable seam), less churn on the primary path, and error conventions stay cleanly separated (placeholders for primary, recorded errors for shadow).
- **Type consistency check:** `ShadowConfig` field names match between Tasks 5/9/10/11; `ShadowSegment/ShadowRunOutput/ShadowComparison*` defined once in Task 10 and consumed in 11/12 (report reads the JSON keys `primaryText/graniteText/graniteLatencySec` — matches the Codable names); `BenchManifest` names match between Tasks 3/4.
- **Known adaptation points for the executor** (not placeholders — decision recorded where discovered): Task 1 GGUF filenames (API check), Task 2 request shape (probe), Task 4 ESB ref-column fallback chain, Task 10 `speakerLabels` visibility + Duration→seconds conversion.
