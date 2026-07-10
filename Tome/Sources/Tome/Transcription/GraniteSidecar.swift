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
    /// Returns the response body alongside its HTTP status so callers can
    /// distinguish a 2xx-with-unparseable-body (ParseError, no relaunch) from
    /// a 4xx/5xx (connection-class failure, relaunch path) — see
    /// GraniteSidecar.transcribe.
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> (Data, Int)
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

    /// A non-2xx llama-server response. Thrown from transcribe()'s do-block
    /// so the outer catch treats it exactly like a dropped connection (the
    /// relaunch path) — the body here is opaque error text, not an
    /// unparseable-but-2xx transcript shape, so it must NOT be confused with
    /// GraniteRequest.ParseError.
    private struct HTTPStatusError: Error, CustomStringConvertible {
        let status: Int
        var description: String { "HTTP \(status)" }
    }

    private let config: ShadowConfig
    private let launcher: any SidecarProcessLauncher
    private let http: any SidecarHTTP
    private let readyTimeout: TimeInterval
    private let sleep: @Sendable (TimeInterval) async -> Void
    private var process: (any SidecarProcess)?
    private var didRelaunch = false
    /// Monotonic teardown generation (pattern: ASRCoordinator.lastInstallToken).
    /// stop() bumps it; an in-flight start() or relaunch that captured an older
    /// value is stale and must tear down rather than surface a running process —
    /// every await in those paths is an interleaving opportunity for stop().
    private var generation = 0
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
        let gen = generation
        // Guard against a second start() while a process is already tracked
        // (ready or mid-poll) — without this, the prior handle would be
        // silently overwritten below and its llama-server orphaned.
        if process != nil {
            await endProcess()
            // stop() may have landed during endProcess's grace suspension —
            // honor it: don't launch a replacement the caller just tore down.
            guard generation == gen else {
                state = .idle
                return false
            }
        }
        let healthURL = config.baseURL.appendingPathComponent("health")
        // Pre-launch probe: if something is already answering /health on our
        // port before we've launched anything, it's a foreign/orphaned
        // llama-server (e.g. leaked by a prior crash) — refuse to adopt it.
        // Launching on top would either fail to bind or silently hand our
        // requests to a process we don't control and can't clean up.
        if await http.healthStatus(healthURL) == 200 {
            diagLog("[SHADOW] port \(config.port) already serving /health — refusing to adopt foreign llama-server (kill it or change graniteShadowPort)")
            state = .failed
            return false
        }
        // The probe above is an interleaving opportunity for stop() too.
        guard generation == gen else {
            state = .idle
            return false
        }
        do {
            process = try launcher.launch(
                executable: URL(fileURLWithPath: config.serverPath),
                arguments: ["-m", config.modelGGUF.path,
                            "--mmproj", config.mmprojGGUF.path,
                            "--host", "127.0.0.1",
                            "--port", String(config.port),
                            // 16k context: granite supports it, and Q8 KV at
                            // this size is fine on 64 GB (verified in
                            // Phase 0). Note mtmd still internally chunks
                            // audio >30s into 30s windows regardless of
                            // context size — the resulting boundary-quality
                            // caveat is tracked in the shadow-week results
                            // doc, not addressed here.
                            "-c", "16384",
                            "--no-webui"])
        } catch {
            diagLog("[SHADOW] sidecar launch failed: \(error)")
            state = .failed
            return false
        }
        let iterations = Int(readyTimeout / 0.5)
        for _ in 0..<iterations {
            // Check the child is still alive BEFORE polling — a dead child
            // (e.g. a port-bind failure) would otherwise spin through the
            // full readyTimeout before failing.
            guard process?.isRunning == true else {
                diagLog("[SHADOW] sidecar process exited before becoming healthy — failing")
                await endProcess()
                state = (generation == gen) ? .failed : .idle
                return false
            }
            let status = await http.healthStatus(healthURL)
            // Both the healthStatus await above and the sleep below are
            // interleaving opportunities for stop(). Re-check the generation
            // BEFORE honoring a 200 — otherwise a stop() that already tore
            // down our process would be followed by state = .ready, reporting
            // a live sidecar after stop() returned (resurrection).
            guard generation == gen else {
                await endProcess()
                state = .idle
                return false
            }
            if status == 200 {
                state = .ready
                return true
            }
            await sleep(0.5)
        }
        diagLog("[SHADOW] sidecar not healthy within \(readyTimeout)s — killing")
        await endProcess()
        // A stop() during the final endProcess grace wins over .failed.
        state = (generation == gen) ? .failed : .idle
        return false
    }

    func transcribe(wavData: Data) async throws -> String {
        guard state == .ready else { throw SidecarError.notReady }
        let url = config.baseURL.appendingPathComponent(
            GraniteRequest.endpointPath.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
        let body = GraniteRequest.build(wavData: wavData)
        do {
            let (data, status) = try await http.post(url, body: body, timeout: 600)
            guard (200..<300).contains(status) else {
                // llama-server responded but with an error status — the body
                // is opaque error text, not the expected transcript shape, so
                // this is a connection-class failure (relaunch/give-up path
                // below), not a ParseError.
                throw HTTPStatusError(status: status)
            }
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
            if Task.isCancelled {
                // Don't burn the one relaunch budget respawning a sidecar for
                // a caller that's already gone — let the cancellation unwind.
                throw error
            }
            // Any other thrown error from http.post (including the
            // HTTPStatusError above) is treated as a connection-level failure
            // and triggers the single relaunch path.
            guard !didRelaunch else {
                diagLog("[SHADOW] request failed after relaunch — failing sidecar: \(error)")
                await endProcess()
                state = .failed
                throw SidecarError.requestFailed
            }
            diagLog("[SHADOW] request failed (\(error)) — relaunching sidecar once")
            didRelaunch = true
            // The endProcess/start awaits below are interleaving opportunities
            // for stop(): capture the teardown generation and re-validate after
            // each, so a stop() landing mid-relaunch is honored instead of the
            // relaunch resurrecting a fresh process after stop() returned.
            let gen = generation
            await endProcess()
            guard generation == gen else {
                state = .idle
                throw SidecarError.notReady
            }
            let started = await start()
            guard generation == gen else {
                await endProcess()
                state = .idle
                throw SidecarError.notReady
            }
            guard started else { throw SidecarError.requestFailed }
            // Recurse so a failure on the retried attempt is handled by the
            // same didRelaunch-guarded catch above (fails the sidecar and
            // marks .failed instead of leaking a third http.post call).
            return try await transcribe(wavData: wavData)
        }
    }

    func stop() async {
        // Bump first: invalidates any in-flight start()/relaunch that captured
        // an older generation. Rest state before the endProcess suspension so a
        // reentrant transcribe() sees not-ready instead of posting to a
        // process that is mid-teardown (and then relaunching it).
        generation += 1
        state = .idle
        await endProcess()
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
        // Registered process-globally (not just held by this GraniteSidecar
        // instance) so the app's quit path can kill it even if the sidecar
        // that spawned it has already gone out of scope — see SidecarRegistry.
        SidecarRegistry.register(pid: p.processIdentifier)
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
    func terminate() {
        if process.isRunning { process.terminate() }
        SidecarRegistry.unregister(pid: process.processIdentifier)
    }
    func forceKill() {
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        SidecarRegistry.unregister(pid: process.processIdentifier)
    }
    deinit {
        if process.isRunning { process.terminate() }
        SidecarRegistry.unregister(pid: process.processIdentifier)
    }
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
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> (Data, Int) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.httpBody = body
        req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, resp) = try await URLSession.shared.data(for: req)
        return (data, (resp as? HTTPURLResponse)?.statusCode ?? 0)
    }
}
