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
        var totalFrames: AVAudioFrameCount = 0
        var pcmData = Data()

        // Drain loop: keep converting until converter exhausts input/output or
        // reports error. Each iteration's produced bytes are appended to
        // `pcmData` immediately, before `out` is reset for the next iteration —
        // correct regardless of whether convert() ever splits its output across
        // multiple .haveData chunks (a single iteration is the common case
        // here, since `capacity` is sized to fit the whole conversion up
        // front, but nothing below depends on that). The previous version
        // copied bytes only once, after the loop, from whatever `out` held on
        // the FINAL iteration — correct only by the single-iteration
        // assumption; a real multi-chunk conversion would have discarded the
        // earlier chunks' samples while still counting their frames.
        while true {
            var drainError: NSError?
            let status = converter.convert(to: out, error: &drainError) { _, status in
                if fed { status.pointee = .endOfStream; return nil }
                fed = true; status.pointee = .haveData; return buffer
            }
            if let drainError { throw drainError }
            if status == .error { throw ExportError.conversionFailed }
            if out.frameLength > 0 {
                pcmData.append(Data(bytes: out.int16ChannelData![0], count: Int(out.frameLength) * 2))
                totalFrames += out.frameLength
            }
            if status == .endOfStream || out.frameLength == 0 { break }
            out.frameLength = 0  // Reset for next iteration
        }

        // Sanity check: output length should be ~expected; silent truncation is an error
        let expected = Double(buffer.frameLength) * 16000.0 / buffer.format.sampleRate
        if Double(totalFrames) < expected - 4096 {
            throw ExportError.conversionFailed
        }

        var data = riffHeader(dataByteCount: pcmData.count)
        data.append(pcmData)
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
