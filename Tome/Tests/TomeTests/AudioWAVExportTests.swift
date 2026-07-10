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
