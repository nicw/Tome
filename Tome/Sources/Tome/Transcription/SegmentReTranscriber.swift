@preconcurrency import AVFoundation
import FluidAudio

/// Offline re-transcriber: takes diarization segments and the system audio WAV,
/// extracts each segment's audio, and runs Parakeet on it individually via the
/// shared `ASRCoordinator` so it properly serializes with live streaming.
struct SegmentReTranscriber: Sendable {
    let asrCoordinator: ASRCoordinator
    let fileURL: URL
    let segments: [DiarizedSegment]
    /// First speaker number for labels: 2 for call capture (system stream; "You" is the
    /// implicit Speaker 1), 1 for mic-only in-person diarization (every speaker, including
    /// the recording user, comes from the diarizer).
    let speakerNumberBase: Int

    func run() async -> [ReTranscribedSegment]? {
        do {
            let audioFile = try AVAudioFile(forReading: fileURL)
            let sampleRate = audioFile.processingFormat.sampleRate
            let totalFrames = AVAudioFramePosition(audioFile.length)

            let speakerMap = speakerLabels(from: segments.map(\.speakerId), startingAt: speakerNumberBase)

            // Merge consecutive segments from the same speaker (< 0.5s gap)
            let merged = SegmentAudio.merge(segments)

            var output: [ReTranscribedSegment] = []

            for seg in merged {
                guard let range = SegmentAudio.paddedFrameRange(
                    startTime: seg.startTime, endTime: seg.endTime,
                    sampleRate: sampleRate, totalFrames: totalFrames
                ) else { continue }

                do {
                    // nil = buffer allocation failure → silent skip (as before); a read
                    // failure throws into the catch below → "[transcription failed]".
                    guard let buffer = try SegmentAudio.readSegment(file: audioFile, start: range.start, count: range.count) else { continue }
                    let result = try await asrCoordinator.transcribe(buffer: buffer, source: .system)
                    let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { continue }

                    let label = speakerMap[seg.speakerId] ?? "Speaker \(speakerNumberBase)"
                    output.append(ReTranscribedSegment(speaker: label, text: text, startTime: seg.startTime))
                } catch {
                    // Visible holes in the final transcript beat silent drops — the user can see
                    // which segment failed and re-record. "[transcription failed]" is the agreed
                    // placeholder convention; downstream rebuilders treat it as opaque text.
                    diagLog("[RETRANSCRIBE] Segment \(seg.startTime)-\(seg.endTime) failed: \(error.localizedDescription)")
                    let label = speakerMap[seg.speakerId] ?? "Speaker \(speakerNumberBase)"
                    output.append(ReTranscribedSegment(speaker: label, text: "[transcription failed]", startTime: seg.startTime))
                    continue
                }
            }

            diagLog("[RETRANSCRIBE] Produced \(output.count) segments from \(merged.count) merged diarization segments")
            return output
        } catch {
            diagLog("[RETRANSCRIBE] FAILED: \(error.localizedDescription)")
            return nil
        }
    }
}

/// A speaker/time triple produced by pyannote diarization.
struct DiarizedSegment: Sendable {
    let speakerId: String
    let startTime: Float
    let endTime: Float
}

/// Full diarization output: per-speaker segments plus an optional acoustic centroid
/// per raw speaker id ("SPEAKER_n"). Each centroid is the mean of that speaker's window
/// embeddings in SpeakerKit's raw embedder space (un-normalized); `VoiceprintSidecar`
/// L2-normalizes it before writing. Surfaced for downstream voiceprint enrollment. The
/// `centroids` map is empty when the diarizer produced no embeddings.
struct DiarizationOutput: Sendable {
    let segments: [DiarizedSegment]
    let centroids: [String: [Float]]
}

/// A segment after re-transcription with a speaker label.
struct ReTranscribedSegment: Sendable {
    let speaker: String
    let text: String
    let startTime: Float
}
