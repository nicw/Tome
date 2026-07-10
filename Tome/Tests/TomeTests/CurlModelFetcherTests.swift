import Foundation
import Testing
@testable import Tome

/// CI-safe: these exercise the pure URL / path / parse / progress helpers only.
/// No curl is ever spawned and no network is touched.
@Suite struct CurlModelFetcherTests {

    // MARK: - Tree JSON → file-list parsing

    @Test func fileListKeepsOnlyFileEntries() throws {
        // Directory entries (type == "directory") must be dropped; only files kept.
        let json = """
        [
          {"type": "directory", "path": "openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc"},
          {"type": "file", "path": "openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc/coremldata.bin", "size": 123},
          {"type": "file", "path": "openai_whisper-large-v3-v20240930/config.json", "size": 45},
          {"type": "directory", "path": "openai_whisper-large-v3-v20240930/TextDecoder.mlmodelc"}
        ]
        """.data(using: .utf8)!
        let files = try CurlModelFetcher.fileList(fromTreeJSON: json)
        #expect(files == [
            "openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc/coremldata.bin",
            "openai_whisper-large-v3-v20240930/config.json",
        ])
    }

    @Test func fileListEmptyForNoFiles() throws {
        let json = "[{\"type\": \"directory\", \"path\": \"foo\"}]".data(using: .utf8)!
        #expect(try CurlModelFetcher.fileList(fromTreeJSON: json).isEmpty)
    }

    @Test func fileListThrowsOnNonArrayJSON() {
        let json = "{\"error\": \"not found\"}".data(using: .utf8)!
        #expect(throws: CurlModelFetcher.FetchError.self) {
            _ = try CurlModelFetcher.fileList(fromTreeJSON: json)
        }
    }

    // MARK: - URL derivation

    @Test func treeURLIsRecursiveAgainstTheModelsAPI() {
        let url = CurlModelFetcher.treeURL(repo: "argmaxinc/whisperkit-coreml", path: "openai_whisper-large-v3-v20240930")
        #expect(url.absoluteString ==
            "https://huggingface.co/api/models/argmaxinc/whisperkit-coreml/tree/main/openai_whisper-large-v3-v20240930?recursive=true")
    }

    @Test func resolveURLForNestedFile() {
        let url = CurlModelFetcher.resolveURL(
            repo: "argmaxinc/whisperkit-coreml",
            filePath: "openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc/coremldata.bin")
        #expect(url.absoluteString ==
            "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main/openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc/coremldata.bin")
    }

    // MARK: - Destination path derivation (must match HubApi / WhisperBackend layout)

    @Test func destinationForNestedFileMatchesModelFolderLayout() {
        let root = URL(fileURLWithPath: "/tmp/base")
        let variant = "openai_whisper-large-v3-v20240930"
        let filePath = "\(variant)/AudioEncoder.mlmodelc/coremldata.bin"
        let dest = CurlModelFetcher.destination(
            destRoot: root, repo: "argmaxinc/whisperkit-coreml", filePath: filePath)
        #expect(dest.path ==
            "/tmp/base/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc/coremldata.bin")

        // The curl layout must land inside exactly the folder WhisperBackend loads
        // from — otherwise the offline load after a fallback would miss the files.
        let modelFolder = WhisperBackend.modelFolder(variant: variant)
        let expected = modelFolder.appendingPathComponent("AudioEncoder.mlmodelc/coremldata.bin")
        let viaFetcher = CurlModelFetcher.destination(
            destRoot: WhisperBackend.downloadBase, repo: "argmaxinc/whisperkit-coreml", filePath: filePath)
        #expect(viaFetcher.path == expected.path)
    }

    @Test func destinationForTokenizerMatchesTokenizerJSONLayout() {
        let dest = CurlModelFetcher.destination(
            destRoot: WhisperBackend.downloadBase,
            repo: "openai/whisper-large-v3",
            filePath: "tokenizer.json")
        #expect(dest.path == WhisperBackend.tokenizerJSON.path)
    }

    // MARK: - Progress fraction math

    @Test func progressFractionIsCompletedOverTotal() {
        #expect(CurlModelFetcher.progressFraction(completed: 0, total: 4) == 0)
        #expect(CurlModelFetcher.progressFraction(completed: 1, total: 4) == 0.25)
        #expect(CurlModelFetcher.progressFraction(completed: 4, total: 4) == 1)
    }

    @Test func progressFractionTreatsZeroTotalAsComplete() {
        #expect(CurlModelFetcher.progressFraction(completed: 0, total: 0) == 1)
    }

    @Test func progressFractionClampsToUnitInterval() {
        #expect(CurlModelFetcher.progressFraction(completed: 5, total: 4) == 1)
        #expect(CurlModelFetcher.progressFraction(completed: -1, total: 4) == 0)
    }
}
