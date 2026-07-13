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

    /// One file entry from the HF tree listing: its repo-relative path and
    /// (when the tree API reports it) its expected byte size. `size` is nil
    /// for the explicit tokenizer file list (`fetchFiles`), which has no tree
    /// listing to draw a size from.
    struct RemoteFile: Equatable {
        let path: String
        let size: Int?
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

    /// Parse the HF `tree` JSON (an array of `{type, path, size, ...}` objects)
    /// into the list of files, keeping only `type == "file"` entries. Split out
    /// so the parse is unit-testable from a fixture string with no network.
    static func fileList(fromTreeJSON data: Data) throws -> [RemoteFile] {
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
            return RemoteFile(path: path, size: entry["size"] as? Int)
        }
    }

    /// Progress as completed/total, clamped to 0…1. Total == 0 reads as complete.
    static func progressFraction(completed: Int, total: Int) -> Double {
        guard total > 0 else { return 1 }
        return min(max(Double(completed) / Double(total), 0), 1)
    }

    /// Whether an already-present destination file can stand in for a fresh
    /// download. True only when BOTH sizes are known and equal — an unknown
    /// existing size (no file yet) or unknown expected size (no tree-listing
    /// size, e.g. the explicit tokenizer fetch) always redownloads. Pure so a
    /// fixture test can cover it with no filesystem or network access.
    ///
    /// This replaces `-C -` resume: `-C -` on a `--fail` request against an
    /// already-complete file gets HTTP 416 from HF and the whole fetch fails.
    /// Comparing sizes up front avoids the 416 entirely and gives resume at
    /// file granularity (skip whole files that already match).
    static func shouldSkipDownload(existingFileSize: Int?, expectedSize: Int?) -> Bool {
        guard let existingFileSize, let expectedSize else { return false }
        return existingFileSize == expectedSize
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
        try await download(repo: repo, files: files, into: destRoot, onProgress: onProgress)
    }

    /// Download an EXPLICIT set of files from a repo (used for the tokenizer,
    /// where listing the whole repo would pull the multi-GB PyTorch weights we
    /// don't need). No tree listing means no known size, so these always
    /// redownload rather than skip — acceptable since the tokenizer files are
    /// tiny. Progress is completedFiles/totalFiles as 0…1.
    static func fetchFiles(
        repo: String,
        files: [String],
        into destRoot: URL,
        onProgress: @Sendable (Double) -> Void
    ) async throws {
        let remoteFiles = files.map { RemoteFile(path: $0, size: nil) }
        try await download(repo: repo, files: remoteFiles, into: destRoot, onProgress: onProgress)
    }

    // MARK: - Internals

    private static func download(
        repo: String,
        files: [RemoteFile],
        into destRoot: URL,
        onProgress: @Sendable (Double) -> Void
    ) async throws {
        onProgress(0)
        var completed = 0
        for file in files {
            // A cancelled provisioning cycle must not keep pulling gigabytes
            // file-by-file just because no single curl invocation is in flight
            // at the moment the cancellation lands.
            try Task.checkCancellation()

            let dest = destination(destRoot: destRoot, repo: repo, filePath: file.path)
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)

            if shouldSkipDownload(existingFileSize: fileSize(at: dest), expectedSize: file.size) {
                completed += 1
                onProgress(progressFraction(completed: completed, total: files.count))
                continue
            }
            // Not a size match — clear any partial file from a prior
            // interrupted run before downloading fresh (no `-C -`): resuming a
            // partial that's actually already-complete-but-unequal, or corrupt,
            // would otherwise mix stale and fresh bytes in one file.
            try? FileManager.default.removeItem(at: dest)

            let url = resolveURL(repo: repo, filePath: file.path)
            do {
                _ = try await runCurl(
                    args: downloadArgs(url: url, dest: dest),
                    context: "download \(file.path)"
                )
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw FetchError.downloadFailed(url: url.absoluteString, detail: error.localizedDescription)
            }
            completed += 1
            onProgress(progressFraction(completed: completed, total: files.count))
        }
    }

    private static func fileSize(at url: URL) -> Int? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) else { return nil }
        return attrs[.size] as? Int
    }

    /// curl args to fetch a URL's bytes to stdout.
    private static func dataArgs(url: URL) -> [String] {
        ["-sSL", "--fail", "--retry", "3", url.absoluteString]
    }

    /// curl args to download a URL to a file. No `-C -`: the size check above
    /// already decides skip-vs-redownload at file granularity, and `-C -`
    /// against an already-complete file gets `--fail`ed with HTTP 416.
    private static func downloadArgs(url: URL, dest: URL) -> [String] {
        ["-sSL", "--fail", "--retry", "3", "-o", dest.path, url.absoluteString]
    }

    /// Bridges `Task` cancellation into the blocking curl child. The
    /// `withTaskCancellationHandler` `onCancel` closure can fire immediately
    /// (synchronously, before the background block even runs) or concurrently
    /// from any thread once the child is running, so `process`/`cancelled` are
    /// guarded behind a lock: `register` reports back if cancellation already
    /// happened so the caller terminates the just-created process itself.
    private final class CancellableProcessBox: @unchecked Sendable {
        private let lock = NSLock()
        private var process: Process?
        private(set) var isCancelled = false

        /// Returns false if cancellation already arrived — the caller is then
        /// responsible for terminating `process` itself, since `cancel()` has
        /// nothing to terminate yet.
        func register(_ process: Process) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            if isCancelled { return false }
            self.process = process
            return true
        }

        /// `Process.terminate()` is documented safe to call from any thread.
        func cancel() {
            lock.lock()
            isCancelled = true
            let p = process
            lock.unlock()
            p?.terminate()
        }
    }

    /// Run `/usr/bin/curl` off the main actor and return its stdout. `Process` is
    /// blocking, so it runs on a background queue and resumes a continuation;
    /// throws if curl exits non-zero (stderr is surfaced in the message).
    ///
    /// Cancellation: wrapped in `withTaskCancellationHandler` so a cancelled
    /// `Task` terminates the curl child instead of leaving it downloading with
    /// the await pending forever. The continuation is resumed exactly once
    /// either way — normal exit/non-zero-exit resumes inline after
    /// `waitUntilExit()` returns; a cancel-triggered `terminate()` unblocks
    /// that same `waitUntilExit()` (no separate `terminationHandler` needed,
    /// so there's no second resume path to reason about) and the box's
    /// `isCancelled` flag turns that same resume into `CancellationError`.
    private static func runCurl(args: [String], context: String) async throws -> Data {
        let box = CancellableProcessBox()
        return try await withTaskCancellationHandler {
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
                    if !box.register(process) {
                        // onCancel already fired before we could register —
                        // terminate right away instead of downloading further.
                        process.terminate()
                    }
                    // Drain stdout before waiting to avoid a full-pipe deadlock on
                    // large bodies; with -sS, stderr stays tiny (errors only).
                    let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()
                    if box.isCancelled {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
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
        } onCancel: {
            box.cancel()
        }
    }
}
