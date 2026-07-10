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
