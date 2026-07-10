import Foundation

/// Builds/parses granite llama-server requests. The contract is pinned in
/// scripts/asr-bench/granite_request.md — Phase 0 validated it; change both
/// together or not at all.
///
/// Note: The client-side request timeout used by callers is 600 s. See the
/// granite sidecar implementation for the constant definition.
enum GraniteRequest {
    static let prompt = "can you transcribe the speech into a written format?"
    static let endpointPath = "/v1/chat/completions"

    enum ParseError: Error { case unexpectedShape }

    static func build(wavData: Data) -> Data {
        let body: [String: Any] = [
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "input_audio",
                     "input_audio": ["data": wavData.base64EncodedString(), "format": "wav"]],
                    ["type": "text", "text": prompt],
                ],
            ]],
            "temperature": 0, "max_tokens": 2048, "stream": false,
        ]
        return try! JSONSerialization.data(withJSONObject: body)
    }

    static func parseResponse(_ data: Data) throws -> String {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String
        else { throw ParseError.unexpectedShape }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
