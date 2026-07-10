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
