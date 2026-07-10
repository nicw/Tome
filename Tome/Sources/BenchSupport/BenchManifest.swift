import Foundation

public struct ManifestEntry: Codable, Equatable, Sendable {
    public let id: String
    public let wav: String
    public init(id: String, wav: String) { self.id = id; self.wav = wav }
}

public struct HypothesisEntry: Codable, Equatable, Sendable {
    public let id: String
    public let text: String
    public init(id: String, text: String) { self.id = id; self.text = text }
}

public enum BenchManifest {
    public static func parse(_ jsonl: String) throws -> [ManifestEntry] {
        try jsonl.split(separator: "\n", omittingEmptySubsequences: true)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { try JSONDecoder().decode(ManifestEntry.self, from: Data($0.utf8)) }
    }
    public static func emit(_ hyps: [HypothesisEntry]) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return hyps.map { String(data: try! enc.encode($0), encoding: .utf8)! }
            .joined(separator: "\n") + (hyps.isEmpty ? "" : "\n")
    }
}
