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
        // merge() preserves first-occurrence order of distinct speaker IDs, so
        // labels match SegmentReTranscriber's raw-array map.
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
                      sampleRate: sampleRate, totalFrames: totalFrames)
            else {
                results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                             durationSec: duration, text: nil,
                                             error: "segment read failed", latencySec: 0))
                continue
            }
            // SegmentAudio.readSegment throws on file-read failure and returns
            // nil on buffer-allocation failure — both are non-fatal here, and
            // both continue to the next segment (correction vs the brief's
            // single-nil check, since the signature now throws).
            let buffer: AVAudioPCMBuffer
            do {
                guard let b = try SegmentAudio.readSegment(file: audioFile, start: range.start, count: range.count) else {
                    results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                                 durationSec: duration, text: nil,
                                                 error: "segment allocation failed", latencySec: 0))
                    continue
                }
                buffer = b
            } catch {
                results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                             durationSec: duration, text: nil,
                                             error: "segment read failed: \(error)", latencySec: 0))
                continue
            }
            let t0 = clock.now
            do {
                let text = try await transcriber.transcribe(buffer: buffer)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                             durationSec: duration, text: text, error: nil,
                                             latencySec: secondsSince(t0, clock: clock)))
            } catch {
                results.append(ShadowSegment(startTime: seg.startTime, speaker: speaker,
                                             durationSec: duration, text: nil,
                                             error: String(describing: error),
                                             latencySec: secondsSince(t0, clock: clock)))
                if let sidecarError = error as? GraniteSidecar.SidecarError {
                    // .notReady = sidecar dead/stopped; .requestFailed =
                    // relaunch budget exhausted, process torn down. Either way
                    // the sidecar is gone — stop burning segments. (Exhaustive
                    // switch so a future recoverable case must pick its policy
                    // here at compile time.)
                    switch sidecarError {
                    case .notReady, .requestFailed:
                        incomplete = true
                    }
                    break
                }
            }
        }
        return ShadowRunOutput(segments: results, incomplete: incomplete)
    }

    /// Duration → seconds. `Double(truncating: duration / .seconds(1) as NSNumber)`
    /// doesn't compile for `Duration`; decompose into seconds + attoseconds instead.
    private func secondsSince(_ t0: ContinuousClock.Instant, clock: ContinuousClock) -> Double {
        let d = clock.now - t0
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}

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
        // Pair by merged-segment startTime + speaker (spec §4: primary skips
        // empty-text segments, so array positions don't line up; "" marks a
        // missing side). Both sides derive startTime from the same
        // SegmentAudio.merge(...) output (Float, same source segments), but
        // startTime ALONE isn't unique: overlapping-speaker diarization can
        // produce two merged segments with an identical startTime for
        // different speakers, and keying on startTime alone would collide
        // (uniquingKeysWith silently drops one primary text). A same-speaker
        // same-startTime pair can't survive merge(), so startTime+speaker is
        // unique on both sides.
        func pairKey(startTime: Float, speaker: String) -> String { "\(startTime)|\(speaker)" }
        let primaryByKey = Dictionary(primary.map { (pairKey(startTime: $0.startTime, speaker: $0.speaker), $0.text) },
                                      uniquingKeysWith: { a, _ in a })
        let segments = shadow.segments.map { s in
            ShadowComparisonSegment(startTime: s.startTime, speaker: s.speaker,
                                    durationSec: s.durationSec,
                                    primaryText: primaryByKey[pairKey(startTime: s.startTime, speaker: s.speaker)] ?? "",
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
        // Flag off is the common case — stay silent (spec: only log skips
        // when the flag is actually on).
        guard config != nil else { return false }
        guard didRebuild, let primary, !primary.isEmpty else {
            diagLog("[SHADOW] flag on but skipping shadow phase this session (didRebuild=\(didRebuild), primarySegments=\(primary?.count ?? 0))")
            return false
        }
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
        // Spawn-per-phase: a fresh sidecar instance for this run. Never call
        // start() again on an instance that has already been stop()ed.
        let sc = sidecar ?? GraniteSidecar(config: config)
        diagLog("[SHADOW] starting sidecar for \(session.sessionID) (\(diarSegments.count) diar segments)")
        guard await sc.start() else {
            diagLog("[SHADOW] sidecar failed to start — skipping session \(session.sessionID)")
            return
        }
        // From here on the sidecar process is live: every exit path — normal
        // completion, early return, or the run() call itself throwing (it
        // doesn't, but artifact writing below can) — must stop it first so no
        // llama-server outlives the phase.
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
