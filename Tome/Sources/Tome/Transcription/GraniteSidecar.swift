import Foundation

protocol SidecarProcess: Sendable {
    var isRunning: Bool { get }
    func terminate()
    func forceKill()
}

protocol SidecarProcessLauncher: Sendable {
    func launch(executable: URL, arguments: [String]) throws -> any SidecarProcess
}

protocol SidecarHTTP: Sendable {
    func healthStatus(_ url: URL) async -> Int?
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data
}

/// Owns one llama-server child process, spawn-per-job (spec §3): ~4 GB of
/// model RAM stays off the machine between jobs. One relaunch on connection
/// failure; a second failure fails the phase.
///
/// Plain actor (not @MainActor) — Task 10's shadow phase drives this from
/// PostProcessingJob, which is @MainActor-bound; calls into this actor hop
/// off the main actor via the usual actor-to-actor await.
actor GraniteSidecar {
    enum State: Equatable { case idle, ready, failed }
    enum SidecarError: Error { case notReady, requestFailed }

    private let config: ShadowConfig
    private let launcher: any SidecarProcessLauncher
    private let http: any SidecarHTTP
    private let readyTimeout: TimeInterval
    private let sleep: @Sendable (TimeInterval) async -> Void
    private var process: (any SidecarProcess)?
    private var didRelaunch = false
    private(set) var state: State = .idle

    init(config: ShadowConfig,
         launcher: any SidecarProcessLauncher = DefaultProcessLauncher(),
         http: any SidecarHTTP = URLSessionSidecarHTTP(),
         readyTimeout: TimeInterval = 60,
         sleep: @Sendable @escaping (TimeInterval) async -> Void = { try? await Task.sleep(for: .seconds($0)) }) {
        self.config = config
        self.launcher = launcher
        self.http = http
        self.readyTimeout = readyTimeout
        self.sleep = sleep
    }

    @discardableResult
    func start() async -> Bool {
        // Guard against a second start() while a process is already tracked
        // (ready or mid-poll) — without this, the prior handle would be
        // silently overwritten below and its llama-server orphaned.
        if process != nil {
            await endProcess()
        }
        do {
            process = try launcher.launch(
                executable: URL(fileURLWithPath: config.serverPath),
                arguments: ["-m", config.modelGGUF.path,
                            "--mmproj", config.mmprojGGUF.path,
                            "--host", "127.0.0.1",
                            "--port", String(config.port)])
        } catch {
            diagLog("[SHADOW] sidecar launch failed: \(error)")
            state = .failed
            return false
        }
        let iterations = Int(readyTimeout / 0.5)
        for _ in 0..<iterations {
            if await http.healthStatus(config.baseURL.appendingPathComponent("health")) == 200 {
                state = .ready
                return true
            }
            await sleep(0.5)
        }
        diagLog("[SHADOW] sidecar not healthy within \(readyTimeout)s — killing")
        await endProcess()
        state = .failed
        return false
    }

    func transcribe(wavData: Data) async throws -> String {
        guard state == .ready else { throw SidecarError.notReady }
        let url = config.baseURL.appendingPathComponent(
            GraniteRequest.endpointPath.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
        let body = GraniteRequest.build(wavData: wavData)
        do {
            let data = try await http.post(url, body: body, timeout: 600)
            return try GraniteRequest.parseResponse(data)
        } catch let error as GraniteRequest.ParseError {
            // A parse error means the server responded (200) but with an
            // unexpected shape — retrying won't fix that, so it propagates
            // as-is without relaunching or touching sidecar state.
            throw error
        } catch {
            // stop() may have run while this call was suspended in http.post
            // (the actor serializes these, but the await point is a
            // interleaving opportunity). If so, honor the stop — don't
            // resurrect a process the caller already asked to tear down.
            guard state == .ready else { throw SidecarError.notReady }
            // Any other thrown error from http.post is treated as a
            // connection-level failure and triggers the single relaunch path.
            guard !didRelaunch else {
                diagLog("[SHADOW] request failed after relaunch — failing sidecar: \(error)")
                await endProcess()
                state = .failed
                throw SidecarError.requestFailed
            }
            diagLog("[SHADOW] request failed (\(error)) — relaunching sidecar once")
            didRelaunch = true
            await endProcess()
            guard await start() else { throw SidecarError.requestFailed }
            // Recurse so a failure on the retried attempt is handled by the
            // same didRelaunch-guarded catch above (fails the sidecar and
            // marks .failed instead of leaking a third http.post call).
            return try await transcribe(wavData: wavData)
        }
    }

    func stop() async {
        await endProcess()
        state = .idle
    }

    /// Terminate then escalate to SIGKILL after a bounded grace period, using
    /// the injected `sleep` so tests stay deterministic (fakes' `sleep` is a
    /// no-op and FakeProcess drops `isRunning` immediately on terminate(),
    /// so this returns after the first check in tests).
    private func endProcess() async {
        guard let p = process else { return }
        p.terminate()
        if p.isRunning {
            for _ in 0..<5 {
                await sleep(1)
                if !p.isRunning { break }
            }
        }
        if p.isRunning { p.forceKill() }
        process = nil
    }
}

// MARK: - Real implementations

struct DefaultProcessLauncher: SidecarProcessLauncher {
    func launch(executable: URL, arguments: [String]) throws -> any SidecarProcess {
        let p = Process()
        p.executableURL = executable
        p.arguments = arguments
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        try p.run()
        return RealSidecarProcess(process: p)
    }
}

/// Wraps Process; forceKill sends SIGKILL. A leaked llama-server must not
/// outlive Tome: Process children die with the parent only if killed, so
/// terminationHandler is not enough — the shadow phase's defer + this
/// wrapper's deinit both call terminate.
final class RealSidecarProcess: SidecarProcess, @unchecked Sendable {
    private let process: Process
    init(process: Process) { self.process = process }
    var isRunning: Bool { process.isRunning }
    func terminate() { if process.isRunning { process.terminate() } }
    func forceKill() { if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
    deinit { if process.isRunning { process.terminate() } }
}

struct URLSessionSidecarHTTP: SidecarHTTP {
    // localhost URLSession is fine here — the known URLSession/HF-CDN stall
    // issue (see ModelProvisioner/downloads) is remote-CDN-specific; this
    // talks only to 127.0.0.1, never leaves the loopback interface.
    func healthStatus(_ url: URL) async -> Int? {
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        guard let (_, resp) = try? await URLSession.shared.data(for: req) else { return nil }
        return (resp as? HTTPURLResponse)?.statusCode
    }
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.httpBody = body
        req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, _) = try await URLSession.shared.data(for: req)
        return data
    }
}
