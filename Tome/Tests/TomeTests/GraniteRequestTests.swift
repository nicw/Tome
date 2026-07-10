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
