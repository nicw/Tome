"""Granite llama-server client. Request contract: see granite_request.md (source of truth)."""
import base64, json, time, urllib.request

PROMPT = "can you transcribe the speech into a written format?"  # granite_request.md


def build_request(wav_bytes: bytes, prompt: str = PROMPT) -> dict:
    return {
        "messages": [{"role": "user", "content": [
            {"type": "input_audio",
             "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0, "max_tokens": 2048, "stream": False,
    }


def transcribe(base_url: str, wav_path: str) -> tuple[str, float]:
    body = json.dumps(build_request(open(wav_path, "rb").read())).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                  headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    return out["choices"][0]["message"]["content"].strip(), time.monotonic() - t0


if __name__ == "__main__":
    import sys
    text, dt = transcribe(sys.argv[1] if len(sys.argv) > 2 else "http://127.0.0.1:8873", sys.argv[-1])
    print(f"[{dt:.1f}s] {text}")
