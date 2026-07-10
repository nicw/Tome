import Foundation

/// Curl-based fallback fetcher for HuggingFace model files.
///
/// Why this exists: on some networks `URLSession` (and Python `urllib`) cannot
/// connect to the HF Xet CDN at any timeout, while `/usr/bin/curl` connects in
/// ~30ms — a client-stack issue, NOT the SDK's 10s timeout (documented in
/// docs/superpowers/plans/2026-07-08-benchmark-results.md "Known issue"). On
/// those networks `WhisperKit.download` always fails, so Whisper could never be
/// installed in-app. This shells out to curl instead.
///
/// Shelling out is permitted because Tome is NOT sandboxed — `Tome.entitlements`
/// declares only `com.apple.security.device.audio-input` and
/// `com.apple.security.device.screen-capture`, with no
/// `com.apple.security.app-sandbox` key — so `Foundation.Process` may exec curl.
///
/// The on-disk layout it produces mirrors HubApi (and `WhisperBackend`'s path
/// helpers) exactly: every file lands at `destRoot/models/<repo>/<filePath>`, so
/// `WhisperBackend.modelFolder(variant:)` / `.tokenizerJSON` resolve to the same
/// paths the SDK download would have written.
enum CurlModelFetcher {

    enum FetchError: Error, LocalizedError {
        case listFailed(repo: String, path: String, detail: String)
        case parseFailed(String)
        case emptyListing(repo: String, path: String)
        case downloadFailed(url: String, detail: String)

        var errorDescription: String? {
            switch self {
            case .listFailed(let repo, let path, let detail):
                return "curl could not list \(repo)/\(path): \(detail)"
            case .parseFailed(let detail):
                return "could not parse HF tree listing: \(detail)"
            case .emptyListing(let repo, let path):
                return "HF tree listing for \(repo)/\(path) contained no files"
            case .downloadFailed(let url, let detail):
                return "curl failed to download \(url): \(detail)"
            }
        }
    }

    // MARK: - Pure URL / path / progress helpers (unit-tested, no network)

    /// The HF tree-API URL that lists every file under a repo path.
    static func treeURL(repo: String, path: String) -> URL {
        URL(string: "https://huggingface.co/api/models/\(repo)/tree/main/\(path)?recursive=true")!
    }

    /// The HF `resolve` URL that serves a single file's bytes.
    static func resolveURL(repo: String, filePath: String) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(filePath)")!
    }

    /// Local destination for a downloaded file. Mirrors HubApi /
    /// `WhisperBackend.modelFolder`: `destRoot/models/<repo>/<filePath>`.
    static func destination(destRoot: URL, repo: String, filePath: String) -> URL {
        destRoot.appendingPathComponent("models/\(repo)/\(filePath)")
    }

    /// Parse the HF `tree` JSON (an array of `{type, path, ...}` objects) into
    /// the list of file paths, keeping only `type == "file"` entries. Split out
    /// so the parse is unit-testable from a fixture string with no network.
    static func fileList(fromTreeJSON data: Data) throws -> [String] {
        let root: Any
        do {
            root = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw FetchError.parseFailed(error.localizedDescription)
        }
        guard let entries = root as? [[String: Any]] else {
            throw FetchError.parseFailed("expected a top-level array of objects")
        }
        return entries.compactMap { entry in
            guard (entry["type"] as? String) == "file",
                  let path = entry["path"] as? String
            else { return nil }
            return path
        }
    }

    /// Progress as completed/total, clamped to 0…1. Total == 0 reads as complete.
    static func progressFraction(completed: Int, total: Int) -> Double {
        guard total > 0 else { return 1 }
        return min(max(Double(completed) / Double(total), 0), 1)
    }

    // MARK: - Fetch entry points

    /// List every file under `repo`/`path` via the HF tree API (fetched WITH
    /// curl too — the urllib-class clients that hang on the CDN also hang on
    /// these API calls) and download each into `destRoot`. Progress is
    /// completedFiles/totalFiles as 0…1.
    static func fetchVariant(
        repo: String,
        path: String,
        into destRoot: URL,
        onProgress: @Sendable (Double) -> Void
    ) async throws {
        let treeData: Data
        do {
            treeData = try await runCurl(
                args: dataArgs(url: treeURL(repo: repo, path: path)),
                context: "list \(repo)/\(path)"
            )
        } catch {
            throw FetchError.listFailed(repo: repo, path: path, detail: error.localizedDescription)
        }
        let files = try fileList(fromTreeJSON: treeData)
        guard !files.isEmpty else { throw FetchError.emptyListing(repo: repo, path: path) }
        try await download(repo: repo, filePaths: files, into: destRoot, onProgress: onProgress)
    }

    /// Download an EXPLICIT set of files from a repo (used for the tokenizer,
    /// where listing the whole repo would pull the multi-GB PyTorch weights we
    /// don't need). Progress is completedFiles/totalFiles as 0…1.
    static func fetchFiles(
        repo: String,
        files: [String],
        into destRoot: URL,
        onProgress: @Sendable (Double) -> Void
    ) async throws {
        try await download(repo: repo, filePaths: files, into: destRoot, onProgress: onProgress)
    }

    // MARK: - Internals

    private static func download(
        repo: String,
        filePaths: [String],
        into destRoot: URL,
        onProgress: @Sendable (Double) -> Void
    ) async throws {
        onProgress(0)
        var completed = 0
        for filePath in filePaths {
            let dest = destination(destRoot: destRoot, repo: repo, filePath: filePath)
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            let url = resolveURL(repo: repo, filePath: filePath)
            do {
                _ = try await runCurl(
                    args: downloadArgs(url: url, dest: dest),
                    context: "download \(filePath)"
                )
            } catch {
                throw FetchError.downloadFailed(url: url.absoluteString, detail: error.localizedDescription)
            }
            completed += 1
            onProgress(progressFraction(completed: completed, total: filePaths.count))
        }
    }

    /// curl args to fetch a URL's bytes to stdout.
    private static func dataArgs(url: URL) -> [String] {
        ["-sSL", "--fail", "--retry", "3", url.absoluteString]
    }

    /// curl args to download a URL to a file, resuming a partial (`-C -`).
    private static func downloadArgs(url: URL, dest: URL) -> [String] {
        ["-sSL", "--fail", "--retry", "3", "-C", "-", "-o", dest.path, url.absoluteString]
    }

    /// Run `/usr/bin/curl` off the main actor and return its stdout. `Process` is
    /// blocking, so it runs on a background queue and resumes a continuation;
    /// throws if curl exits non-zero (stderr is surfaced in the message).
    private static func runCurl(args: [String], context: String) async throws -> Data {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data, Error>) in
            DispatchQueue.global(qos: .utility).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
                process.arguments = args
                let outPipe = Pipe()
                let errPipe = Pipe()
                process.standardOutput = outPipe
                process.standardError = errPipe
                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }
                // Drain stdout before waiting to avoid a full-pipe deadlock on
                // large bodies; with -sS, stderr stays tiny (errors only).
                let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                if process.terminationStatus != 0 {
                    let stderr = String(data: errData, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    let detail = stderr.isEmpty
                        ? "\(context): curl exited \(process.terminationStatus)"
                        : "\(context): \(stderr)"
                    continuation.resume(throwing: FetchError.downloadFailed(url: context, detail: detail))
                    return
                }
                continuation.resume(returning: outData)
            }
        }
    }
}
