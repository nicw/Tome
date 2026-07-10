import Foundation
import Testing
@testable import Tome

/// SidecarRegistry is process-global mutable state, so these tests run
/// serialized (not concurrently with each other) and never invoke the real
/// `kill` — every killAll() call below injects a recording signaler instead.
/// Fake pids are large/UUID-derived to make accidental collisions between
/// tests vanishingly unlikely even so.
@Suite(.serialized) struct SidecarRegistryTests {
    private final class Recorder: @unchecked Sendable {
        var calls: [(pid: Int32, sig: Int32)] = []
    }

    /// Large pid drawn from a UUID so concurrent test runs (or leftover state
    /// from another test) can't collide with it.
    private func fakePid() -> Int32 {
        Int32.random(in: 100_000...2_000_000_000)
    }

    @Test func registerAndUnregisterUpdateMembership() {
        let pid = fakePid()
        SidecarRegistry.register(pid: pid)
        #expect(SidecarRegistry.registeredPids.contains(pid))
        SidecarRegistry.unregister(pid: pid)
        #expect(!SidecarRegistry.registeredPids.contains(pid))
    }

    @Test func unregisterOfUnknownPidIsNoOpAndLeavesOthersIntact() {
        let known = fakePid()
        let unknown = fakePid()
        SidecarRegistry.register(pid: known)
        SidecarRegistry.unregister(pid: unknown)  // never registered
        #expect(SidecarRegistry.registeredPids.contains(known))
        SidecarRegistry.unregister(pid: known)
    }

    @Test func killAllSendsSIGTERMToEveryRegisteredPidAndDrainsRegistry() {
        let a = fakePid(), b = fakePid()
        SidecarRegistry.register(pid: a)
        SidecarRegistry.register(pid: b)
        let recorder = Recorder()
        let count = SidecarRegistry.killAll { pid, sig in
            recorder.calls.append((pid, sig))
            // kill(pid, 0) semantics: 0 == still alive, -1 == no such
            // process. Reporting "gone" on the liveness poll here so the
            // grace loop exits immediately instead of waiting out the full
            // grace period.
            return sig == 0 ? -1 : 0
        }
        #expect(count == 2)
        let termed = Set(recorder.calls.filter { $0.sig == SIGTERM }.map(\.pid))
        #expect(termed == [a, b])
        #expect(!SidecarRegistry.registeredPids.contains(a))
        #expect(!SidecarRegistry.registeredPids.contains(b))
    }

    @Test func killAllEscalatesToSIGKILLWhenLivenessPollReportsStillAlive() {
        let pid = fakePid()
        SidecarRegistry.register(pid: pid)
        let recorder = Recorder()
        // graceSeconds tiny so the escalation path doesn't slow the suite —
        // production still defaults to the spec's 2s grace.
        SidecarRegistry.killAll(graceSeconds: 0.05) { p, sig in
            recorder.calls.append((p, sig))
            return sig == 0 ? 0 : 0   // kill(pid, 0) == 0 means "still alive"
        }
        let sigkills = recorder.calls.filter { $0.sig == SIGKILL }
        #expect(!sigkills.isEmpty)
        #expect(sigkills.allSatisfy { $0.pid == pid })
        #expect(!SidecarRegistry.registeredPids.contains(pid))
    }

    @Test func killAllWithNothingRegisteredIsIdempotentNoOp() {
        // Drain any leftover registrations from other tests deterministically
        // first, so this assertion doesn't depend on suite ordering.
        _ = SidecarRegistry.killAll { _, _ in 0 }
        let count = SidecarRegistry.killAll { _, _ in
            Issue.record("signaler must not be called when the registry is empty")
            return 0
        }
        #expect(count == 0)
    }

    @Test func killAllFlipsThePermanentQuittingGate() {
        // The gate is never reset (the app is exiting), so this can only
        // assert the forward direction: after killAll, isQuitting is true —
        // including on an empty registry (the flip must not be skipped by the
        // nothing-to-kill early return). GraniteSidecarTests inject their own
        // gate closure precisely because this flip is permanent process-global
        // state shared across suites.
        _ = SidecarRegistry.killAll { _, _ in -1 }
        #expect(SidecarRegistry.isQuitting)
        _ = SidecarRegistry.killAll { _, _ in -1 }   // idempotent: still true
        #expect(SidecarRegistry.isQuitting)
    }

    @Test func killAllIsSynchronousAndSafeToCallFromANonMainThread() async {
        let pid = fakePid()
        SidecarRegistry.register(pid: pid)
        let recorder = Recorder()
        await Task.detached {
            SidecarRegistry.killAll { p, sig in
                recorder.calls.append((p, sig))
                return sig == 0 ? -1 : 0   // reports gone -> no escalation wait
            }
        }.value
        #expect(recorder.calls.contains { $0.pid == pid && $0.sig == SIGTERM })
    }
}
