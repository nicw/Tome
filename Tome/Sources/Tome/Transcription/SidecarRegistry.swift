import Foundation
import os

/// Process-global record of every live sidecar (llama-server) child pid.
///
/// A `GraniteSidecar` is spawned fresh per shadow-phase run (spec §3) and may
/// already be out of scope by the time the app is asked to quit — nothing
/// outside Transcription/ knows a sidecar exists, so there's no instance to
/// ask to stop(). This registry is the process-wide backstop: the app's quit
/// path calls `killAll()` unconditionally so no llama-server ever outlives
/// Tome, independent of whatever GraniteSidecar instance (if any) is still
/// holding a reference to the process.
///
/// `enum` (no instances, process-wide) protected by an os_unfair_lock
/// (`OSAllocatedUnfairLock`, the same primitive MicCapture/SystemAudioCapture
/// already use for cross-thread state) rather than an actor, because
/// `killAll()` must be callable synchronously from
/// `applicationShouldTerminate`, which is not async.
enum SidecarRegistry {
    /// Injectable so tests can record signals instead of sending real ones.
    /// Defaults to the real libc `kill`. Used both to deliver SIGTERM/SIGKILL
    /// and, via `signaler(pid, 0)`, to poll liveness during the grace period
    /// (the standard "kill(pid, 0) == 0 means still alive" idiom).
    typealias Signaler = @Sendable (Int32, Int32) -> Int32

    private static let state = OSAllocatedUnfairLock<Set<Int32>>(uncheckedState: [])

    static func register(pid: Int32) {
        state.withLock { pids in _ = pids.insert(pid) }
    }

    static func unregister(pid: Int32) {
        state.withLock { pids in _ = pids.remove(pid) }
    }

    /// Test/inspection only.
    static var registeredPids: Set<Int32> { state.withLock { $0 } }

    /// SIGTERM every registered pid, poll for up to `graceSeconds` (50ms
    /// steps via `usleep`) for each to exit, then SIGKILL any stragglers.
    /// Synchronous and safe to call from any thread — callers include
    /// `applicationShouldTerminate`, which is not async. Idempotent: the
    /// registry is drained atomically up front, so a second call (or a
    /// concurrent one) has nothing left to act on.
    @discardableResult
    static func killAll(graceSeconds: TimeInterval = 2.0, signaler: Signaler = kill) -> Int {
        let victims = state.withLock { pids -> Set<Int32> in
            defer { pids.removeAll() }
            return pids
        }
        guard !victims.isEmpty else { return 0 }
        for pid in victims { _ = signaler(pid, SIGTERM) }

        let pollIntervalUsec: UInt32 = 50_000  // 50ms
        let deadline = Date().addingTimeInterval(graceSeconds)
        var alive = victims
        while !alive.isEmpty && Date() < deadline {
            usleep(pollIntervalUsec)
            alive = alive.filter { signaler($0, 0) == 0 }
        }
        for pid in alive { _ = signaler(pid, SIGKILL) }
        return victims.count
    }
}
