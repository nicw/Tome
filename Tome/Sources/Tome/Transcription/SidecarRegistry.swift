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

    /// pids + the quitting gate share one lock so killAll's victim snapshot
    /// and the gate flip are a single atomic step — no window where a
    /// concurrent spawn could observe "not quitting" after the snapshot was
    /// already taken.
    private struct Registry {
        var pids: Set<Int32> = []
        var quitting = false
    }

    private static let state = OSAllocatedUnfairLock<Registry>(uncheckedState: Registry())

    static func register(pid: Int32) {
        state.withLock { r in _ = r.pids.insert(pid) }
    }

    static func unregister(pid: Int32) {
        state.withLock { r in _ = r.pids.remove(pid) }
    }

    /// Test/inspection only.
    static var registeredPids: Set<Int32> { state.withLock { $0.pids } }

    /// True once killAll() has run — the app is exiting. GraniteSidecar
    /// checks this at both of its spawn points (start(), and transcribe()'s
    /// relaunch branch) so an in-flight job whose connection drops DURING
    /// the kill can't respawn a fresh llama-server after killAll's victim
    /// snapshot was taken: killAll blocks the caller's thread (main, at
    /// quit), but the sidecar actor keeps running on its own executor.
    /// Never reset — there is no un-quit.
    static var isQuitting: Bool { state.withLock { $0.quitting } }

    /// SIGTERM every registered pid, poll for up to `graceSeconds` (50ms
    /// steps via `usleep`) for each to exit, then SIGKILL any stragglers.
    /// Synchronous and safe to call from any thread — callers include
    /// `applicationShouldTerminate`, which is not async. Idempotent: the
    /// registry is drained atomically up front, so a second call (or a
    /// concurrent one) has nothing left to act on. Also flips the permanent
    /// `isQuitting` gate in the same lock acquisition as the snapshot, so no
    /// new sidecar can be spawned after the victims are chosen.
    @discardableResult
    static func killAll(graceSeconds: TimeInterval = 2.0, signaler: Signaler = kill) -> Int {
        let victims = state.withLock { r -> Set<Int32> in
            r.quitting = true
            defer { r.pids.removeAll() }
            return r.pids
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
