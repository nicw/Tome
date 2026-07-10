import Foundation
import Testing
@testable import Tome

final class FakeProcess: SidecarProcess, @unchecked Sendable {
    var running = true
    var terminated = false, killed = false
    /// A stubborn process ignores SIGTERM (terminate leaves it running) and
    /// only dies to forceKill — exercises endProcess's escalation branch.
    var stubborn = false
    var isRunning: Bool { running }
    func terminate() { terminated = true; if !stubborn { running = false } }
    func forceKill() { killed = true; running = false }
}

final class FakeLauncher: SidecarProcessLauncher, @unchecked Sendable {
    var launched: [(URL, [String])] = []
    var processes: [FakeProcess] = []
    var launchError: (any Error)?
    /// When true, vended processes are stubborn (see FakeProcess.stubborn).
    var makeStubborn = false
    func launch(executable: URL, arguments: [String]) throws -> any SidecarProcess {
        if let launchError { throw launchError }
        launched.append((executable, arguments))
        let p = FakeProcess(); p.stubborn = makeStubborn; processes.append(p); return p
    }
}

/// Continuation-gate parking follows FakeBackend.swift's style (see
/// ASRCoordinatorTests.installRevalidatesTokenAcrossUnloadSuspension): a test
/// arms hangNext*, waits for the *Parked flag, interleaves, then releases.
final class FakeHTTP: SidecarHTTP, @unchecked Sendable {
    var healthResults: [Int?] = [200]
    var postResults: [Result<Data, any Error>] = []
    /// When true, the next healthStatus call parks until releaseHealth().
    var hangNextHealth = false
    private(set) var healthParked = false
    private var healthGate: CheckedContinuation<Void, Never>?
    /// When true, the next post call parks until releasePost().
    var hangNextPost = false
    private(set) var postParked = false
    private var postGate: CheckedContinuation<Void, Never>?

    func releaseHealth() { healthGate?.resume(); healthGate = nil }
    func releasePost() { postGate?.resume(); postGate = nil }

    func healthStatus(_ url: URL) async -> Int? {
        if hangNextHealth {
            hangNextHealth = false
            healthParked = true
            await withCheckedContinuation { healthGate = $0 }
            healthParked = false
        }
        return healthResults.isEmpty ? 200 : healthResults.removeFirst()
    }
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data {
        if hangNextPost {
            hangNextPost = false
            postParked = true
            await withCheckedContinuation { postGate = $0 }
            postParked = false
        }
        return try postResults.removeFirst().get()
    }
}

/// Bounded poll until `condition` — the park-wait pattern from
/// ASRCoordinatorTests (100 × 10 ms; the caller asserts afterwards).
private func waitFor(_ condition: @autoclosure @escaping () -> Bool) async throws {
    for _ in 0..<100 {
        if condition() { return }
        try await Task.sleep(for: .milliseconds(10))
    }
}

private func makeSidecar(launcher: FakeLauncher = FakeLauncher(), http: FakeHTTP = FakeHTTP())
    -> (GraniteSidecar, FakeLauncher, FakeHTTP) {
    let config = ShadowConfig(serverPath: "/fake/llama-server",
                              modelDir: URL(fileURLWithPath: "/fake/models"), port: 9999)
    let s = GraniteSidecar(config: config, launcher: launcher, http: http,
                           readyTimeout: 1, sleep: { _ in })
    return (s, launcher, http)
}

private let ok = Data(#"{"choices":[{"message":{"content":"hi"}}]}"#.utf8)
private struct ConnErr: Error {}

@Suite struct GraniteSidecarTests {
    @Test func startLaunchesWithConfigArgsAndPollsHealth() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [503, 200]
        #expect(await s.start())
        let (exe, args) = launcher.launched[0]
        #expect(exe.path == "/fake/llama-server")
        #expect(args.contains("--port") && args.contains("9999") && args.contains("127.0.0.1"))
        #expect(args.contains("/fake/models/\(ShadowConfig.modelFilename)"))
    }
    @Test func startFailsAfterTimeoutAndKills() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = Array(repeating: 503 as Int?, count: 500)
        #expect(await s.start() == false)
        #expect(launcher.processes[0].terminated || launcher.processes[0].killed)
    }
    @Test func transcribeSendsRequestAndParses() async throws {
        let (s, _, http) = makeSidecar()
        http.postResults = [.success(ok)]
        _ = await s.start()
        #expect(try await s.transcribe(wavData: Data([1])) == "hi")
    }
    @Test func connectionFailureRelaunchesOnceThenFails() async {
        let (s, launcher, http) = makeSidecar()
        http.postResults = [.failure(ConnErr()), .failure(ConnErr())]
        _ = await s.start()
        await #expect(throws: (any Error).self) { try await s.transcribe(wavData: Data([1])) }
        #expect(launcher.launched.count == 2)   // original + one relaunch
        // subsequent calls fail fast without further launches
        await #expect(throws: (any Error).self) { try await s.transcribe(wavData: Data([1])) }
        #expect(launcher.launched.count == 2)
    }
    @Test func stopTerminatesProcess() async {
        let (s, launcher, _) = makeSidecar()
        _ = await s.start()
        await s.stop()
        #expect(launcher.processes[0].terminated)
    }

    // MARK: - stop() reentrancy (generation guard)

    @Test func stopDuringRelaunchHealthPollAbortsAndLeavesNoProcess() async throws {
        let (s, launcher, http) = makeSidecar()
        _ = await s.start()
        // First post fails -> relaunch; the relaunch's start() parks in its
        // health poll. The trailing .success is a sentinel: it must NOT be
        // consumed (no post may go out after stop()).
        http.postResults = [.failure(ConnErr()), .success(ok)]
        http.hangNextHealth = true
        let job = Task { try await s.transcribe(wavData: Data([1])) }
        try await waitFor(http.healthParked)
        #expect(http.healthParked)

        await s.stop()
        http.releaseHealth()

        await #expect(throws: (any Error).self) { try await job.value }
        #expect(launcher.launched.count == 2)               // original + relaunch, none after stop
        #expect(launcher.processes.allSatisfy { !$0.running })
        #expect(await s.state == .idle)
        #expect(http.postResults.count == 1)                // sentinel untouched
    }

    @Test func stopDuringInitialStartHealthPollAbortsStart() async throws {
        let (s, launcher, http) = makeSidecar()
        http.hangNextHealth = true
        let job = Task { await s.start() }
        try await waitFor(http.healthParked)
        #expect(http.healthParked)

        await s.stop()
        http.releaseHealth()

        #expect(await job.value == false)
        #expect(await s.state == .idle)
        #expect(launcher.launched.count == 1)
        #expect(launcher.processes.allSatisfy { !$0.running })
    }

    @Test func stopWhilePostInFlightFailsFastWithoutRelaunch() async throws {
        let (s, launcher, http) = makeSidecar()
        _ = await s.start()
        http.postResults = [.failure(ConnErr()), .success(ok)]  // trailing sentinel
        http.hangNextPost = true
        let job = Task { try await s.transcribe(wavData: Data([1])) }
        try await waitFor(http.postParked)
        #expect(http.postParked)

        await s.stop()
        http.releasePost()

        await #expect(throws: (any Error).self) { try await job.value }
        #expect(launcher.launched.count == 1)               // no relaunch after stop
        #expect(launcher.processes.allSatisfy { !$0.running })
        #expect(await s.state == .idle)
        #expect(http.postResults.count == 1)                // sentinel untouched
    }

    // MARK: - error taxonomy & escalation

    @Test func parseErrorPropagatesWithoutRelaunch() async throws {
        let (s, launcher, http) = makeSidecar()
        http.postResults = [.success(Data("not json".utf8))]
        _ = await s.start()
        await #expect(throws: GraniteRequest.ParseError.self) {
            try await s.transcribe(wavData: Data([1]))
        }
        #expect(launcher.launched.count == 1)               // relaunch budget not burned
        #expect(await s.state == .ready)                    // sidecar state untouched
    }

    @Test func stubbornProcessEscalatesToForceKill() async {
        let (s, launcher, _) = makeSidecar()
        launcher.makeStubborn = true
        _ = await s.start()
        await s.stop()
        let p = launcher.processes[0]
        #expect(p.terminated && p.killed && !p.running)
    }
}
