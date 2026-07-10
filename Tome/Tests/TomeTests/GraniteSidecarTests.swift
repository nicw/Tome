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
    /// Every start() call now probes /health twice: once pre-launch (must
    /// NOT be 200, or start() refuses to adopt a "foreign" server) and once
    /// in the post-launch poll loop (200 == ready). The default models the
    /// realistic case — nothing listening yet, then healthy right after
    /// launch — so tests that just need "start() succeeds once" don't have
    /// to seed this explicitly; tests with multiple start() calls (initial +
    /// relaunch) must seed enough entries to cover every probe.
    var healthResults: [Int?] = [nil, 200]
    var postResults: [Result<(Data, Int), any Error>] = []
    /// When true, the next healthStatus call parks until releaseHealth().
    var hangNextHealth = false
    /// When set, healthStatus parks on its Nth call (1-indexed) regardless
    /// of hangNextHealth — lets a test target the post-launch poll
    /// specifically without racing to flip hangNextHealth between the
    /// pre-launch probe and the first loop iteration.
    var hangHealthAtCallNumber: Int?
    private var healthCallCount = 0
    private(set) var healthParked = false
    private var healthGate: CheckedContinuation<Void, Never>?
    /// When true, the next post call parks until releasePost().
    var hangNextPost = false
    private(set) var postParked = false
    private var postGate: CheckedContinuation<Void, Never>?

    func releaseHealth() { healthGate?.resume(); healthGate = nil }
    func releasePost() { postGate?.resume(); postGate = nil }

    func healthStatus(_ url: URL) async -> Int? {
        healthCallCount += 1
        if hangNextHealth || healthCallCount == hangHealthAtCallNumber {
            hangNextHealth = false
            healthParked = true
            await withCheckedContinuation { healthGate = $0 }
            healthParked = false
        }
        return healthResults.isEmpty ? 200 : healthResults.removeFirst()
    }
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> (Data, Int) {
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

/// `isQuitting` defaults to `{ false }` here (NOT the production default of
/// `SidecarRegistry.isQuitting`): the registry's quitting gate is permanent
/// process-global state that SidecarRegistryTests flip via killAll() in this
/// same test process, and these tests must not depend on suite ordering.
private func makeSidecar(launcher: FakeLauncher = FakeLauncher(), http: FakeHTTP = FakeHTTP(),
                          readyTimeout: TimeInterval = 1,
                          sleep: @escaping @Sendable (TimeInterval) async -> Void = { _ in },
                          isQuitting: @escaping @Sendable () -> Bool = { false })
    -> (GraniteSidecar, FakeLauncher, FakeHTTP) {
    let config = ShadowConfig(serverPath: "/fake/llama-server",
                              modelDir: URL(fileURLWithPath: "/fake/models"), port: 9999)
    let s = GraniteSidecar(config: config, launcher: launcher, http: http,
                           readyTimeout: readyTimeout, sleep: sleep, isQuitting: isQuitting)
    return (s, launcher, http)
}

private let ok = Data(#"{"choices":[{"message":{"content":"hi"}}]}"#.utf8)
private struct ConnErr: Error {}

@Suite struct GraniteSidecarTests {
    @Test func startLaunchesWithConfigArgsAndPollsHealth() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [503, 200]   // pre-launch probe (not foreign), then ready
        #expect(await s.start())
        let (exe, args) = launcher.launched[0]
        #expect(exe.path == "/fake/llama-server")
        #expect(args.contains("--port") && args.contains("9999") && args.contains("127.0.0.1"))
        #expect(args.contains("/fake/models/\(ShadowConfig.modelFilename)"))
        #expect(args.contains("-c") && args.contains("16384"))
        #expect(args.contains("--no-webui"))
    }
    @Test func startFailsAfterTimeoutAndKills() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = Array(repeating: 503 as Int?, count: 500)
        #expect(await s.start() == false)
        #expect(launcher.processes[0].terminated || launcher.processes[0].killed)
    }
    @Test func transcribeSendsRequestAndParses() async throws {
        let (s, _, http) = makeSidecar()
        http.postResults = [.success((ok, 200))]
        _ = await s.start()
        #expect(try await s.transcribe(wavData: Data([1])) == "hi")
    }
    @Test func connectionFailureRelaunchesOnceThenFails() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [nil, 200, nil, 200]   // initial start + one relaunch start
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

    // MARK: - foreign-server / dead-child guards (FIX 2)

    @Test func startRefusesToAdoptForeignServerAlreadyOnPort() async {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [200]   // something already answering /health before any launch
        #expect(await s.start() == false)
        #expect(launcher.launched.count == 0)
        #expect(await s.state == .failed)
    }

    @Test func stopDuringPrelaunchProbeYieldsIdleNotFailedEvenOn200() async throws {
        // A stop() interleaving during the pre-launch probe suspension must
        // win over the probe's outcome: releasing the probe with a 200
        // (foreign server present) must NOT stomp state = .failed over the
        // .idle that stop() just established — and must not launch anything.
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [200]         // probe would report a foreign server
        http.hangHealthAtCallNumber = 1     // park the probe itself
        let job = Task { await s.start() }
        try await waitFor(http.healthParked)
        #expect(http.healthParked)

        await s.stop()
        http.releaseHealth()

        #expect(await job.value == false)
        #expect(await s.state == .idle)     // stop()'s .idle survives, not .failed
        #expect(launcher.launched.count == 0)
    }

    // MARK: - app-quit gate (kill-vs-relaunch race)

    @Test func startRefusesToLaunchWhenAppIsQuitting() async {
        let (s, launcher, _) = makeSidecar(isQuitting: { true })
        #expect(await s.start() == false)
        #expect(launcher.launched.count == 0)
        #expect(await s.state == .failed)
    }

    @Test func postFailureWhileQuittingRethrowsWithoutRelaunch() async {
        // The kill-vs-relaunch race: SidecarRegistry.killAll() terminates the
        // server from the main thread while this actor's transcribe() is
        // suspended in http.post. The dropped connection must NOT take the
        // relaunch branch (state still .ready, Task not cancelled, relaunch
        // budget unspent) — that would spawn and register a fresh child AFTER
        // killAll's victim snapshot, orphaning it.
        final class Gate: @unchecked Sendable { var quitting = false }
        let gate = Gate()
        let (s, launcher, http) = makeSidecar(isQuitting: { gate.quitting })
        http.postResults = [.failure(ConnErr()), .success((ok, 200))]  // trailing sentinel
        _ = await s.start()
        gate.quitting = true   // killAll() has run; connection then drops
        await #expect(throws: ConnErr.self) { try await s.transcribe(wavData: Data([1])) }
        #expect(launcher.launched.count == 1)   // no relaunch spawned past the kill snapshot
        #expect(http.postResults.count == 1)    // sentinel untouched — no retry post either
    }

    @Test func startFailsFastWhenChildDiesBeforeHealthy() async throws {
        let launcher = FakeLauncher()
        let http = FakeHTTP()
        final class Counter: @unchecked Sendable { var sleeps = 0 }
        let counter = Counter()
        // A large readyTimeout (many iterations) so a slow/looping failure
        // mode would be obvious in the sleep count; the dead-child guard
        // should short-circuit long before that.
        let config = ShadowConfig(serverPath: "/fake/llama-server",
                                  modelDir: URL(fileURLWithPath: "/fake/models"), port: 9999)
        let s = GraniteSidecar(config: config, launcher: launcher, http: http,
                               readyTimeout: 250, sleep: { _ in counter.sleeps += 1 })
        http.healthResults = Array(repeating: 503 as Int?, count: 1000)  // never healthy
        http.hangHealthAtCallNumber = 2   // the loop's first poll (after launch)
        let job = Task { await s.start() }
        try await waitFor(http.healthParked)
        #expect(http.healthParked)
        launcher.processes[0].running = false   // simulate a bind failure right after launch
        http.releaseHealth()
        #expect(await job.value == false)
        #expect(launcher.launched.count == 1)   // no relaunch attempted at start() level
        #expect(counter.sleeps <= 1)            // failed fast, not through ~500 timeout iterations
    }

    // MARK: - stop() reentrancy (generation guard)

    @Test func stopDuringRelaunchHealthPollAbortsAndLeavesNoProcess() async throws {
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [nil, 200]   // initial start succeeds
        _ = await s.start()
        // First post fails -> relaunch; the relaunch's start() clears its
        // pre-launch probe, then parks in its post-launch health poll. The
        // trailing .success is a sentinel: it must NOT be consumed (no post
        // may go out after stop()).
        http.postResults = [.failure(ConnErr()), .success((ok, 200))]
        http.healthResults = [nil]        // relaunch's pre-launch probe: proceed
        http.hangHealthAtCallNumber = 4    // relaunch's first loop poll parks
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
        http.healthResults = [nil]        // pre-launch probe: nothing listening yet -> proceed
        http.hangHealthAtCallNumber = 2    // loop's first poll (after launch) parks
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
        http.postResults = [.failure(ConnErr()), .success((ok, 200))]  // trailing sentinel
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
        http.postResults = [.success((Data("not json".utf8), 200))]
        _ = await s.start()
        await #expect(throws: GraniteRequest.ParseError.self) {
            try await s.transcribe(wavData: Data([1]))
        }
        #expect(launcher.launched.count == 1)               // relaunch budget not burned
        #expect(await s.state == .ready)                    // sidecar state untouched
    }

    @Test func httpErrorStatusRelaunchesOnceThenFails() async {
        // A 4xx/5xx llama-server response must be treated the same as a
        // dropped connection (FIX 3) — not surfaced as a ParseError, and not
        // silently ignored.
        let (s, launcher, http) = makeSidecar()
        http.healthResults = [nil, 200, nil, 200]   // initial start + one relaunch start
        http.postResults = [.success((Data("server error".utf8), 500)),
                             .success((Data("server error".utf8), 500))]
        _ = await s.start()
        await #expect(throws: (any Error).self) { try await s.transcribe(wavData: Data([1])) }
        #expect(launcher.launched.count == 2)   // original + one relaunch, same as connection failure
    }

    @Test func cancelledTaskSkipsRelaunchAndRethrows() async {
        let (s, launcher, http) = makeSidecar()
        http.postResults = [.failure(ConnErr())]
        _ = await s.start()
        let task = Task {
            try await s.transcribe(wavData: Data([1]))
        }
        task.cancel()
        await #expect(throws: (any Error).self) { try await task.value }
        #expect(launcher.launched.count == 1)   // no relaunch attempted once cancelled
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
