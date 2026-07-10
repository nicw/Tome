import Foundation

/// Hidden-flag configuration for granite shadow transcription. Read at job
/// creation (not app launch) so toggling applies from the next session.
/// Spec: docs/superpowers/specs/2026-07-09-granite-shadow-transcription-design.md
struct ShadowConfig: Sendable, Equatable {
    let serverPath: String
    let modelDir: URL
    let port: Int

    // Keep in sync with scripts/setup-granite-shadow.sh
    static let modelFilename = "granite-speech-4.1-2b-Q8_0.gguf"
    static let mmprojFilename = "mmproj-model-f16.gguf"

    var modelGGUF: URL { modelDir.appendingPathComponent(Self.modelFilename) }
    var mmprojGGUF: URL { modelDir.appendingPathComponent(Self.mmprojFilename) }
    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    func filesPresent(fileManager: FileManager = .default) -> Bool {
        fileManager.fileExists(atPath: modelGGUF.path) && fileManager.fileExists(atPath: mmprojGGUF.path)
    }

    static func fromDefaults(_ defaults: UserDefaults = .standard) -> ShadowConfig? {
        guard defaults.bool(forKey: "graniteShadowEnabled") else { return nil }
        let server = defaults.string(forKey: "graniteShadowServerPath") ?? "/opt/homebrew/bin/llama-server"
        let dir: URL
        if let override = defaults.string(forKey: "graniteShadowModelDir") {
            dir = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
        } else {
            dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Tome/Granite")
        }
        let port = (defaults.object(forKey: "graniteShadowPort") as? Int) ?? 8873
        return ShadowConfig(serverPath: server, modelDir: dir, port: port)
    }
}
