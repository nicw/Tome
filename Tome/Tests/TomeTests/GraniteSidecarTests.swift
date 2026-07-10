import Foundation
import Testing
@testable import Tome

final class FakeProcess: SidecarProcess, @unchecked Sendable {
    var running = true
    var terminated = false, killed = false
    var isRunning: Bool { running }
    func terminate() { terminated = true; running = false }
    func forceKill() { killed = true; running = false }
}

final class FakeLauncher: SidecarProcessLauncher, @unchecked Sendable {
    var launched: [(URL, [String])] = []
    var processes: [FakeProcess] = []
    var launchError: (any Error)?
    func launch(executable: URL, arguments: [String]) throws -> any SidecarProcess {
        if let launchError { throw launchError }
        launched.append((executable, arguments))
        let p = FakeProcess(); processes.append(p); return p
    }
}

final class FakeHTTP: SidecarHTTP, @unchecked Sendable {
    var healthResults: [Int?] = [200]
    var postResults: [Result<Data, any Error>] = []
    func healthStatus(_ url: URL) async -> Int? {
        healthResults.isEmpty ? 200 : healthResults.removeFirst()
    }
    func post(_ url: URL, body: Data, timeout: TimeInterval) async throws -> Data {
        try postResults.removeFirst().get()
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
}
