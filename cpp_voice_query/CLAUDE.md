# cpp_voice_query — Voice Assistant + Multi-Endpoint Voice Server

**IMPORTANT: This file must always be kept up to date when the codebase changes. Any modification to architecture, endpoints, data structures, CLI options, or behavior must be reflected here.**

Voice-in, voice-out assistant that chains STT → LLM → TTS, plus an HTTP server that accepts text from external agents and speaks it with configurable voices, priority-based interruption, and optional response capture via callbacks.

## Architecture

```
Agent POST /speak/assistant  ──┐
Agent POST /speak/alert      ──┤──→ PriorityPlaybackQueue ──→ TtsEngine ──→ Speaker
Agent POST /speak/monitor    ──┘         ↑
Mic → STT → LLM on_sentence ────────────┘
                                  (high prio interrupts low prio)
                                  (user voice interrupts entire channel)
```

Both the mic loop and HTTP endpoints feed into the same priority queue. Each request is tagged with a channel name for targeted interruption.

## Voice server endpoints

### POST /speak/:channel
Speak complete text. Body: `{"text": "Hello world"}`. Returns 202 Accepted.

Optional fields for response capture:
```json
{
  "text": "Build failed. Should I retry?",
  "wait_response": true,
  "callback_url": "http://agent:9000/response"
}
```
When `wait_response` is true, after the text finishes playing the system captures user speech via STT (30s timeout) and POSTs `{"response": "user's answer"}` to `callback_url`.

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

## Voice interruption (VAD)

User speech interrupts playback at two points in the main loop:

1. **During LLM response** (g_responding phase) — VAD monitor thread detects speech via RMS threshold, calls `interrupt_channel()` to clear the entire currently-playing channel plus all "mic" channel items, and sets `llm_interrupt` to abort LLM streaming.

2. **Pre-STT phase** — When HTTP channel items are playing before STT starts, the main thread polls mic input. Speech interrupts the current channel; if other channels have queued items, those resume. Only when the queue is fully empty does STT begin.

Key behaviors:
- Speaking during playback clears **all items from the interrupted channel**, not just the current sentence
- After interruption, lower-priority channels resume automatically
- The interrupt speech is discarded (not sent to the LLM)
- During a pending response (wait_response hold), speech is treated as the response, not an interrupt

## Wait-for-response flow

When an agent sends `wait_response: true`:

1. Request plays normally via the priority queue
2. If the item completes without interruption, the queue worker **holds** (doesn't process more items)
3. Main loop detects the hold, runs STT with 30s timeout to capture user speech
4. POSTs `{"response": "transcribed text"}` to the callback URL
5. Releases the hold — queue resumes processing remaining items

If the item is interrupted before finishing (user speaks during playback), `wait_response` is cancelled and no callback is sent.

During the hold, the LLM keeps streaming in the background — sentences queue up and play after the response is captured.

## LLM interface

- `POST /v1/chat/completions` with `{"messages": [...], "stream": true}`
- SSE stream: `data: {"choices":[{"delta":{"content":"token"}}]}\n\n`
- End: `data: [DONE]\n\n`
- Default: `http://localhost:8001` (`--llm-url`)
- LLM streaming can be aborted via `llm_interrupt` atomic flag (VAD sets it when user speaks)

## Internal architecture

```
main.cpp                    Wiring: init components, start server, run mic loop, VAD, response capture
SpeechRequest.h             Data: text + priority + voice + language + channel + wait_response + callback_url
VoiceChannel.h              Data: channel name → voice + priority + language
SentenceSplitter.h          Stream of text fragments → stream of sentences
TtsEngine.h/.cpp            Kokoro + Tokenizer wrapper with runtime voice switching
PriorityPlaybackQueue.h/.cpp  Thread-safe priority queue → TTS → AudioPlayer, channel interruption, response hold
InboundServer.h/.cpp        HTTP server (cpp-httplib) exposing /speak/{channel}
LlmClient.h/.cpp            SSE streaming client (libcurl) with interrupt support
PortAudioSource.h           Mic input wrapper for WhisperStream
```

### Pipeline per turn (mic path)

1. **Pre-STT VAD** — If queue has items (from HTTP channels), monitor mic and interrupt on speech. Loop until queue is empty.
2. **STT phase** — WhisperStream listens until silence finalizes an utterance
3. **LLM+TTS phase** — LlmClient streams tokens → SentenceSplitter → PriorityPlaybackQueue (channel="mic")
4. **Wait phase** — Wait for queue to drain or VAD interrupt. Handle any pending response captures.
5. **Loop** — returns to phase 1

### Pipeline (HTTP path)

1. Agent POSTs to `/speak/{channel}`
2. InboundServer creates SpeechRequest with channel's voice/priority + optional wait_response/callback_url
3. PriorityPlaybackQueue synthesizes and plays (may interrupt current audio)
4. If wait_response: worker holds after playback → main loop captures speech → POSTs to callback

### Threading

| Thread | Purpose |
|--------|---------|
| Main | Config, STT, LLM streaming, pre-STT VAD polling, response capture |
| Queue worker | Picks highest-priority request, synthesizes, plays; holds on wait_response |
| HTTP server | Accepts /speak/ requests |
| VAD monitor | Detects user speech during LLM+TTS phase for channel-aware interruption |

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

# Test wait-for-response:
curl -X POST localhost:8090/speak/alert -H 'Content-Type: application/json' \
  -d '{"text": "Build failed. Should I retry?", "wait_response": true, "callback_url": "http://localhost:9000/response"}'
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
