// ASRBench — compares Parakeet-TDT v3 (FluidAudio) and Whisper Large v3
// Turbo (WhisperKit) on identical audio, chunked the way the live pipeline
// chunks it (spec §8): VAD-bounded speech segments, split at the 480k-sample
// (~30s) flush ceiling, segments under ~0.5s dropped — the same caps
// StreamingTranscriber applies.
//
// Usage: swift run -c release ASRBench <wav/m4a...> [--json out.json]

import AVFoundation
import BenchSupport
import Foundation
import FluidAudio
import WhisperKit

// Mirrors WhisperBackend (Sources/Tome/Transcription/WhisperBackend.swift) —
// keep in sync; SwiftPM forbids importing the app executable from here.
let whisperFamily = "openai_whisper-large-v3-v20240930"
let whisperVariant = WhisperKit.recommendedModels().supported.contains(whisperFamily)
    ? whisperFamily : whisperFamily + "_626MB"
let whisperBase = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Tome/WhisperKit", isDirectory: true)

// Live-pipeline caps (see StreamingTranscriber.swift): flush at 480k samples,
// drop sub-8k segments ("Parakeet emits garbage below this").
let sampleRate = 16_000.0
let maxChunkSamples = 480_000
let minChunkSamples = 8_000

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

var argv = Array(CommandLine.arguments.dropFirst())
var jsonOut: String?
if let i = argv.firstIndex(of: "--json"), i + 1 < argv.count {
    jsonOut = argv[i + 1]
    argv.removeSubrange(i...(i + 1))
}
let files = argv
guard !files.isEmpty else {
    print("usage: ASRBench <wav/m4a files...> [--json out.json]")
    exit(1)
}

struct ChunkTiming: Codable {
    let seconds: Double
    let latency: Double
    var rtf: Double { latency / seconds }
}

struct BackendReport: Codable {
    let name: String
    let variant: String
    let downloadSeconds: Double?     // nil when already cached
    let diskSizeMB: Double
    let loadSecondsCold: Double
    let loadSecondsWarm: Double
    let firstTranscribeSeconds: Double   // ANE warm-up shows up here
    let timings: [ChunkTiming]
    let peakRSSMB: Double
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return 0 }
    return sorted[min(Int(Double(sorted.count) * p), sorted.count - 1)]
}

func directorySizeMB(_ url: URL) -> Double {
    guard let enumerator = FileManager.default.enumerator(
        at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
    var total = 0
    for case let file as URL in enumerator {
        total += (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) ?? 0
    }
    return Double(total) / 1_048_576
}

func peakRSSMB() -> Double {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    return Double(usage.ru_maxrss) / 1_048_576.0   // ru_maxrss is bytes on macOS
}

func now() -> Double { CFAbsoluteTimeGetCurrent() }

// --- Parakeet ---

// Model-loading half of benchParakeet(): downloads (if needed), does a cold
// load then a warm load (mirroring the bench's cold/warm timing dance), and
// hands back the warm-loaded manager plus the report fields benchParakeet()
// needs (download/load timings, cache dir). Also used directly by manifest
// mode via the `transcribe` closure it derives its own wrapper from.
func loadParakeetManager() async throws -> (
    asr: AsrManager, downloadSeconds: Double?, dir: URL, loadCold: Double, loadWarm: Double
) {
    let cached = AsrModels.modelsExist(
        at: AsrModels.defaultCacheDirectory(for: .v3), version: .v3)
    let tDownload = now()
    let dir = try await AsrModels.download(version: .v3)
    let downloadSeconds: Double? = cached ? nil : now() - tDownload

    let tCold = now()
    let coldModels = try await AsrModels.load(from: dir, version: .v3)
    let asr = AsrManager(config: .default)
    try await asr.loadModels(coldModels)
    let loadCold = now() - tCold

    await asr.cleanup()
    let tWarm = now()
    let warmModels = try await AsrModels.load(from: dir, version: .v3)
    try await asr.loadModels(warmModels)
    let loadWarm = now() - tWarm

    return (asr, downloadSeconds, dir, loadCold, loadWarm)
}

// Manifest-mode helper: loads Parakeet and returns a (WAV path in, text out)
// transcribe closure, reusing loadParakeetManager()'s load path and the same
// AudioProcessor.loadAudioAsFloatArray sample-loading the top-level bench
// driver uses for its own files.
func loadParakeet() async throws -> (String) async throws -> String {
    let (asr, _, _, _, _) = try await loadParakeetManager()
    return { path in
        let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
        var state = TdtDecoderState.make()
        let result = try await asr.transcribe(samples, decoderState: &state, language: .english)
        return result.text
    }
}

func benchParakeet(chunks: [[Float]]) async throws -> BackendReport {
    let (asr, downloadSeconds, dir, loadCold, loadWarm) = try await loadParakeetManager()

    func transcribe(_ samples: [Float]) async throws -> Double {
        var state = TdtDecoderState.make()
        let t = now()
        _ = try await asr.transcribe(samples, decoderState: &state, language: .english)
        return now() - t
    }

    let tFirst = try await transcribe(chunks[0])
    var timings: [ChunkTiming] = []
    for chunk in chunks {
        timings.append(ChunkTiming(
            seconds: Double(chunk.count) / sampleRate,
            latency: try await transcribe(chunk)))
    }
    return BackendReport(
        name: "Parakeet-TDT v3", variant: "parakeet-tdt-0.6b-v3 int8",
        downloadSeconds: downloadSeconds, diskSizeMB: directorySizeMB(dir),
        loadSecondsCold: loadCold, loadSecondsWarm: loadWarm,
        firstTranscribeSeconds: tFirst, timings: timings, peakRSSMB: peakRSSMB())
}

// --- Whisper ---

// Model-loading half of benchWhisper(): resolves/downloads the model folder,
// does a cold load then a warm load (mirroring the bench's cold/warm timing
// dance), and hands back the warm-loaded kit plus the report fields
// benchWhisper() needs (download timing, folder, variant).
func loadWhisperKit(variant: String, base: URL) async throws -> (
    kit: WhisperKit, downloadSeconds: Double?, folder: URL, loadCold: Double, loadWarm: Double
) {
    let expectedFolder = base.appendingPathComponent(
        "models/argmaxinc/whisperkit-coreml/\(variant)", isDirectory: true)
    let cached = FileManager.default.fileExists(
        atPath: expectedFolder.appendingPathComponent("TextDecoder.mlmodelc").path)
    let folder: URL
    var downloadSeconds: Double? = nil
    if cached {
        folder = expectedFolder
    } else {
        let tDownload = now()
        folder = try await WhisperKit.download(
            variant: variant, downloadBase: base,
            progressCallback: { progress in
                let pct = Int(progress.fractionCompleted * 100)
                if pct % 10 == 0 { print("whisper download: \(pct)%") }
            })
        downloadSeconds = now() - tDownload
    }

    let config = WhisperKitConfig(
        model: variant, downloadBase: base,
        modelFolder: folder.path, load: true, download: false)
    let tCold = now()
    do { _ = try await WhisperKit(config) }   // cold load; instance released at scope end
    let loadCold = now() - tCold
    let tWarm = now()
    let kit = try await WhisperKit(config)
    let loadWarm = now() - tWarm

    return (kit, downloadSeconds, folder, loadCold, loadWarm)
}

// Manifest-mode helper: loads Whisper (default family/base, same as the top-
// level bench driver uses) and returns a (WAV path in, text out) transcribe
// closure, reusing loadWhisperKit()'s load path and the existing
// AudioProcessor.loadAudioAsFloatArray sample-loading code.
func loadWhisper() async throws -> (String) async throws -> String {
    let (kit, _, _, _, _) = try await loadWhisperKit(variant: whisperVariant, base: whisperBase)
    return { path in
        let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
        let results = try await kit.transcribe(
            audioArray: samples,
            decodeOptions: DecodingOptions(task: .transcribe, language: "en"))
        return results.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

func benchWhisper(chunks: [[Float]], variant: String, base: URL) async throws -> BackendReport {
    let (kit, downloadSeconds, folder, loadCold, loadWarm) = try await loadWhisperKit(variant: variant, base: base)

    func transcribe(_ samples: [Float]) async throws -> Double {
        let t = now()
        _ = try await kit.transcribe(
            audioArray: samples,
            decodeOptions: DecodingOptions(task: .transcribe, language: "en"))
        return now() - t
    }

    let tFirst = try await transcribe(chunks[0])
    var timings: [ChunkTiming] = []
    for chunk in chunks {
        timings.append(ChunkTiming(
            seconds: Double(chunk.count) / sampleRate,
            latency: try await transcribe(chunk)))
    }
    return BackendReport(
        name: "Whisper Large v3 Turbo", variant: variant,
        downloadSeconds: downloadSeconds, diskSizeMB: directorySizeMB(folder),
        loadSecondsCold: loadCold, loadSecondsWarm: loadWarm,
        firstTranscribeSeconds: tFirst, timings: timings, peakRSSMB: peakRSSMB())
}

func printReport(_ r: BackendReport) {
    let latencies = r.timings.map(\.latency)
    let rtfs = r.timings.map(\.rtf)
    print("""

    == \(r.name) (\(r.variant)) ==
    download:        \(r.downloadSeconds.map { String(format: "%.0f", $0) + "s" } ?? "cached")
    on disk:         \(String(format: "%.0f", r.diskSizeMB)) MB
    load cold/warm:  \(String(format: "%.1f", r.loadSecondsCold))s / \(String(format: "%.1f", r.loadSecondsWarm))s
    first transcribe (warm-up): \(String(format: "%.2f", r.firstTranscribeSeconds))s
    chunk latency:   p50 \(String(format: "%.2f", percentile(latencies, 0.5)))s  p95 \(String(format: "%.2f", percentile(latencies, 0.95)))s
    RTF:             p50 \(String(format: "%.3f", percentile(rtfs, 0.5)))  p95 \(String(format: "%.3f", percentile(rtfs, 0.95)))
    peak RSS:        \(String(format: "%.0f", r.peakRSSMB)) MB
    chunks:          \(r.timings.count) totaling \(String(format: "%.0f", r.timings.map(\.seconds).reduce(0, +)))s
    """)
}

// --- Top-level: load audio, VAD-segment it, run both benches ---
var allSamples: [Float] = []
for file in files {
    let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: file)
    allSamples.append(contentsOf: samples)
    print("loaded \(file): \(String(format: "%.0f", Double(samples.count) / sampleRate))s")
}

// Same VAD the app uses, same caps as StreamingTranscriber: speech segments,
// split at the ~30s flush ceiling, sub-0.5s dropped.
let vad = try await VadManager()
var chunks: [[Float]] = []
for segment in try await vad.segmentSpeechAudio(allSamples) {
    var offset = 0
    while offset < segment.count {
        let end = min(offset + maxChunkSamples, segment.count)
        if end - offset >= minChunkSamples {
            chunks.append(Array(segment[offset..<end]))
        }
        offset = end
    }
}
guard !chunks.isEmpty else {
    print("error: VAD found no speech ≥0.5s in the input — use a fixture with real speech")
    exit(1)
}
let chunkSummary = chunks.map { Double($0.count) / sampleRate }
print("VAD chunks: \(chunks.count) (\(String(format: "%.1f", chunkSummary.min()!))s–\(String(format: "%.1f", chunkSummary.max()!))s)")

// Parakeet first (already cached on any machine that has run Tome), then Whisper.
let parakeet = try await benchParakeet(chunks: chunks)
printReport(parakeet)
let whisper = try await benchWhisper(chunks: chunks, variant: whisperVariant, base: whisperBase)
printReport(whisper)

let whisperP95 = percentile(whisper.timings.map(\.rtf), 0.95)
print("\nAcceptance bar (spec §8): live use wants p95 RTF < 0.5.")
print("Whisper p95 RTF = \(String(format: "%.3f", whisperP95)) → \(whisperP95 < 0.5 ? "PASS" : "MISS — ship with 'may lag during live transcription' copy")")

if let jsonOut {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode([parakeet, whisper]).write(to: URL(fileURLWithPath: jsonOut))
    print("wrote \(jsonOut)")
}
