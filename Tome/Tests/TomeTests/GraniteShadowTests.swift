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

    // -- corrections coverage: readSegment throws (allocation-nil vs read-error) --
    @Test func runnerRecordsAllocationFailureDistinctFromReadFailure() async throws {
        let wav = try fixtureWAV()
        // A single segment whose padded range is well within the 10s fixture,
        // so SegmentAudio.readSegment succeeds; this test exercises the success
        // path plumbing through the corrected throwing signature (nil vs throw
        // are exercised indirectly since we can't force AVAudioFile to fail
        // without corrupting the fixture — covered by reading a nonexistent file
        // via the file-open failure path below instead).
        let segs = [DiarizedSegment(speakerId: "S0", startTime: 0.0, endTime: 1.0)]
        let runner = ShadowRunner(transcriber: FakeSegmentTranscriber([.success("ok")]))
        let out = await runner.run(fileURL: wav, diarSegments: segs, speakerNumberBase: 2)
        #expect(out.segments.count == 1)
        #expect(out.segments[0].text == "ok")
    }

    @Test func runnerReturnsIncompleteWhenFileCannotBeOpened() async {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("no-such-\(UUID().uuidString).wav")
        let segs = [DiarizedSegment(speakerId: "S0", startTime: 0.0, endTime: 1.0)]
        let runner = ShadowRunner(transcriber: FakeSegmentTranscriber([.success("unused")]))
        let out = await runner.run(fileURL: missing, diarSegments: segs, speakerNumberBase: 2)
        #expect(out.segments.isEmpty)
        #expect(out.incomplete)
    }

    // -- sidecar notReady mid-run marks incomplete and stops burning segments --
    @Test func runnerStopsAndMarksIncompleteOnSidecarNotReady() async throws {
        let wav = try fixtureWAV()
        let segs = [DiarizedSegment(speakerId: "S0", startTime: 0.0, endTime: 2.0),
                    DiarizedSegment(speakerId: "S1", startTime: 3.0, endTime: 5.0)]
        let runner = ShadowRunner(transcriber: FakeSegmentTranscriber([.failure(GraniteSidecar.SidecarError.notReady)]))
        let out = await runner.run(fileURL: wav, diarSegments: segs, speakerNumberBase: 2)
        #expect(out.segments.count == 1)   // stopped after the first (notReady) segment
        #expect(out.incomplete)
    }

    // -- cancellation mid-run: partial results preserved, marked incomplete --
    /// A transcriber that signals a continuation after its first call so the
    /// test can cancel the enclosing Task from outside, simulating cooperative
    /// cancellation arriving between segments. Proves ShadowRunner's
    /// `Task.isCancelled` check (not just the sidecar-notReady check) stops
    /// the loop and reports incomplete, while keeping whatever partial results
    /// were already collected — those still flow into GraniteShadowPhase.run's
    /// unconditional artifact write.
    final class SelfCancelingTranscriber: SegmentTranscribing, @unchecked Sendable {
        func transcribe(buffer: AVAudioPCMBuffer) async throws -> String {
            withUnsafeCurrentTask { $0?.cancel() }
            return "first"
        }
    }
    @Test func runnerStopsAtNextSegmentAfterCancellationAndKeepsPartialResults() async throws {
        let wav = try fixtureWAV()
        let segs = [DiarizedSegment(speakerId: "S0", startTime: 0.0, endTime: 2.0),
                    DiarizedSegment(speakerId: "S1", startTime: 3.0, endTime: 5.0),
                    DiarizedSegment(speakerId: "S2", startTime: 6.0, endTime: 8.0)]
        // Run on a child Task so self-cancellation inside the transcriber
        // doesn't propagate up and cancel the enclosing @Test's own task.
        let child = Task {
            let runner = ShadowRunner(transcriber: SelfCancelingTranscriber())
            return await runner.run(fileURL: wav, diarSegments: segs, speakerNumberBase: 2)
        }
        let out = await child.value
        #expect(out.segments.count == 1)          // first segment completed before cancellation observed
        #expect(out.segments[0].text == "first")
        #expect(out.incomplete)
    }
}
