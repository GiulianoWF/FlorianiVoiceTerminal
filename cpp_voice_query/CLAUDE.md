# cpp_voice_query — Voice Assistant + Multi-Endpoint Voice Server

Voice-in, voice-out assistant that chains STT → LLM → TTS, plus an HTTP server that accepts text from external agents and speaks it with configurable voices and priority-based interruption.

## Architecture

```
Agent POST /speak/assistant  ──┐
Agent POST /speak/alert      ──┤──→ PriorityPlaybackQueue ──→ TtsEngine ──→ Speaker
Agent POST /speak/monitor    ──┘         ↑
Mic → STT → LLM on_sentence ────────────┘
                                  (high prio interrupts low prio)
```

Both the mic loop and HTTP endpoints feed into the same priority queue.

## Voice server endpoints

### POST /speak/:channel
Speak complete text. Body: `{"text": "Hello world"}`. Returns 202 Accepted.

### POST /speak/:channel (streaming)
Speak streamed SSE text. Body: `{"text": "...", "stream": true}`. The body contains OpenAI-format SSE lines. Sentences are split on arrival and TTS starts on the first complete sentence.

### POST /speak/:channel/stream
Alternative streaming endpoint. Body is raw SSE data (same format as llama.cpp output).

### GET /channels
Returns JSON array of configured channels with name, voice, priority, language.

### GET /health
Returns `{"status": "ok"}`.

## Priority system

```
CRITICAL (3) — system emergencies, interrupts everything
HIGH     (2) — agent alerts, permission requests
NORMAL   (1) — conversational responses (mic → LLM flow)
LOW      (0) — monitoring chatter, status updates
```

When a higher-priority request arrives while a lower-priority one is playing, the current audio is interrupted within ~10ms (256 samples at 24kHz). Same-or-lower priority queues behind.

## LLM interface (unchanged)

- `POST /v1/chat/completions` with `{"messages": [...], "stream": true}`
- SSE stream: `data: {"choices":[{"delta":{"content":"token"}}]}\n\n`
- End: `data: [DONE]\n\n`
- Default: `http://localhost:8001` (`--llm-url`)

## Internal architecture

```
main.cpp                    Wiring: init components, start server, run mic loop
SpeechRequest.h             Data: text + priority + voice + language
VoiceChannel.h              Data: channel name → voice + priority + language
SentenceSplitter.h          Stream of text fragments → stream of sentences
TtsEngine.h/.cpp            Kokoro + Tokenizer wrapper with runtime voice switching
PriorityPlaybackQueue.h/.cpp  Thread-safe priority queue → TTS → AudioPlayer
InboundServer.h/.cpp        HTTP server (cpp-httplib) exposing /speak/{channel}
LlmClient.h/.cpp            SSE streaming client (libcurl)
PortAudioSource.h           Mic input wrapper for WhisperStream
```

### Pipeline per turn (mic path)

1. **STT phase** — WhisperStream listens until silence finalizes an utterance
2. **LLM+TTS phase** — LlmClient streams tokens → SentenceSplitter → PriorityPlaybackQueue
3. **Loop** — returns to phase 1

### Pipeline (HTTP path)

1. Agent POSTs to `/speak/{channel}`
2. InboundServer creates SpeechRequest with channel's voice/priority
3. PriorityPlaybackQueue synthesizes and plays (may interrupt current audio)

### Threading

| Thread | Purpose |
|--------|---------|
| Main | Config, STT, LLM streaming |
| Queue worker | Picks highest-priority request, synthesizes, plays |
| HTTP server | Accepts /speak/ requests |
| VAD monitor | Detects user speech during playback |

### Global state
- `g_running` — app lifetime
- `g_responding` — true during LLM+TTS phase (Ctrl+C interrupts via queue)
- `g_queue` — pointer to PriorityPlaybackQueue for signal handler

## Build & run

```bash
# Build and run (auto-downloads models):
bazel run //:voice_query_run

# Test with echo server (no real LLM needed):
python3 echo_server.py &
bazel run //:voice_query_run

# Test voice server:
curl -X POST localhost:8090/speak/assistant -H 'Content-Type: application/json' \
  -d '{"text": "Hello from the agent"}'

# Test priority interruption:
curl -X POST localhost:8090/speak/alert -H 'Content-Type: application/json' \
  -d '{"text": "Alert! Build failed"}'
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-url URL` | `http://localhost:8001` | LLM server URL |
| `--server-port PORT` | `8090` | Voice server HTTP port |
| `--channel SPEC` | (see defaults) | Channel: `name:voice_path:priority:language` |
| `--whisper-model PATH` | via Bazel | Whisper GGUF model |
| `--language LANG` | `pt` | Language (pt, en) |
| `--kokoro-model PATH` | via Bazel | Kokoro ONNX model |
| `--kokoro-voice PATH` | via Bazel (pm_santa.bin) | Default voice |
| `--no-tts` | off | Disable spoken responses |
| `--text-only` | off | Transcribe only, skip LLM |
| `-gpu` | off | GPU for Whisper |
| `--step N` | `500` | Whisper inference interval (ms) |
| `--length N` | `15000` | Whisper audio window (ms) |
| `--stable N` | `3` | Passes to commit a word |
| `--silence N` | `1` | Silent steps before finalizing |

### Channel configuration

Default channels (when no `--channel` flags given):
- `assistant` — default Kokoro voice, NORMAL priority, language from `--language`
- `alert` — default Kokoro voice, HIGH priority, en-us

Custom channels via CLI:
```bash
bazel run //:voice_query_run -- \
  --channel assistant:voices/pm_santa.bin:normal:pt-br \
  --channel alert:voices/af.bin:high:en-us \
  --channel monitor:voices/pm_alex.bin:low:en-us
```

## Dependencies

### Sibling projects (Bazel modules)
- **cpp_whisper** — `@cpp_whisper//:whisper_stream` (STT), `@cpp_whisper//:selected_model`
- **kokoro_tts** — `@kokoro_tts//:kokoro_inference`, `@kokoro_tts//:tokenizer`, `@kokoro_tts//:audio_player`, model + voice downloads

### Vendored
- **cpp-httplib** — `include/httplib/httplib.h` (v0.18.7, header-only, MIT)
- **nlohmann/json** — `include/nlohmann/json.hpp` (header-only)

### System packages
```bash
sudo apt install portaudio19-dev libcurl4-openssl-dev espeak-ng libportaudio2
# For GPU: NVIDIA CUDA toolkit
```
