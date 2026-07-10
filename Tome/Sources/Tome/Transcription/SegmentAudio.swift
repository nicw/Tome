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
