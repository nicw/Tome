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
            CurlModelFetcher.RemoteFile(
                path: "openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc/coremldata.bin", size: 123),
            CurlModelFetcher.RemoteFile(path: "openai_whisper-large-v3-v20240930/config.json", size: 45),
        ])
    }

    @Test func fileListToleratesMissingSize() throws {
        // Some tree entries may omit `size`; that must parse as nil, not throw.
        let json = """
        [{"type": "file", "path": "config.json"}]
        """.data(using: .utf8)!
        let files = try CurlModelFetcher.fileList(fromTreeJSON: json)
        #expect(files == [CurlModelFetcher.RemoteFile(path: "config.json", size: nil)])
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

    // MARK: - Skip-vs-redownload decision (replaces `-C -` resume; avoids HTTP 416)

    @Test func shouldSkipWhenExistingSizeMatchesExpected() {
        #expect(CurlModelFetcher.shouldSkipDownload(existingFileSize: 4096, expectedSize: 4096))
    }

    @Test func shouldNotSkipWhenExistingSizeIsSmallerThanExpected() {
        // The common interrupted-download case: a partial file present.
        #expect(!CurlModelFetcher.shouldSkipDownload(existingFileSize: 1024, expectedSize: 4096))
    }

    @Test func shouldNotSkipWhenExistingSizeIsLargerThanExpected() {
        // A mismatch either way must redownload, not just a short partial.
        #expect(!CurlModelFetcher.shouldSkipDownload(existingFileSize: 8192, expectedSize: 4096))
    }

    @Test func shouldNotSkipWhenNoFileExistsYet() {
        #expect(!CurlModelFetcher.shouldSkipDownload(existingFileSize: nil, expectedSize: 4096))
    }

    @Test func shouldNotSkipWhenExpectedSizeIsUnknown() {
        // `fetchFiles` (tokenizer) has no tree-listing size to compare against —
        // must always redownload rather than trust a same-named local file.
        #expect(!CurlModelFetcher.shouldSkipDownload(existingFileSize: 4096, expectedSize: nil))
    }

    @Test func shouldNotSkipWhenNeitherSizeIsKnown() {
        #expect(!CurlModelFetcher.shouldSkipDownload(existingFileSize: nil, expectedSize: nil))
    }
}
