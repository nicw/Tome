import Foundation
import Testing
@testable import Tome

@Suite struct ShadowConfigTests {
    private func makeDefaults() -> UserDefaults {
        let name = "ShadowConfigTests-\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }
    @Test func disabledByDefault() {
        #expect(ShadowConfig.fromDefaults(makeDefaults()) == nil)
    }
    @Test func enabledUsesDefaults() {
        let d = makeDefaults(); d.set(true, forKey: "graniteShadowEnabled")
        let c = try! #require(ShadowConfig.fromDefaults(d))
        #expect(c.serverPath == "/opt/homebrew/bin/llama-server")
        #expect(c.port == 8873)
        #expect(c.modelDir.path.hasSuffix("Tome/Granite"))
        #expect(c.modelGGUF.lastPathComponent == "granite-speech-4.1-2b-Q8_0.gguf")
        #expect(c.mmprojGGUF.lastPathComponent == "mmproj-model-f16.gguf")
    }
    @Test func overridesRespectedAndTildeExpanded() {
        let d = makeDefaults()
        d.set(true, forKey: "graniteShadowEnabled")
        d.set("/usr/local/bin/llama-server", forKey: "graniteShadowServerPath")
        d.set("~/granite-models", forKey: "graniteShadowModelDir")
        d.set(9001, forKey: "graniteShadowPort")
        let c = try! #require(ShadowConfig.fromDefaults(d))
        #expect(c.serverPath == "/usr/local/bin/llama-server")
        #expect(c.port == 9001)
        #expect(!c.modelDir.path.contains("~"))
        #expect(c.modelDir.path.hasSuffix("/granite-models"))
    }
    @Test func filesPresentFalseOnEmptyDir() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let c = ShadowConfig(serverPath: "/x", modelDir: tmp, port: 1)
        #expect(!c.filesPresent())
    }
}
