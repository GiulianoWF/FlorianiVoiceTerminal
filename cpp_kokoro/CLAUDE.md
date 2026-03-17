# Kokoro TTS — C++ Text-to-Speech Library

Kokoro TTS is a C++ text-to-speech engine powered by the Kokoro ONNX model. It converts text to natural-sounding speech audio using phoneme-based synthesis.

## Using as a Bazel dependency

### 1. Add the dependency to your `MODULE.bazel`

Point to the kokoro_tts repository using `local_path_override` (for local development) or `http_archive` / `git_override` (for remote):

```starlark
# Local development
bazel_dep(name = "kokoro_tts", version = "1.0.0")
local_path_override(
    module_name = "kokoro_tts",
    path = "/path/to/cpp_kokoro",
)
```

### 2. Add deps in your `BUILD.bazel`

The library exposes three independent targets you can depend on individually:

```starlark
cc_binary(
    name = "my_app",
    srcs = ["main.cpp"],
    deps = [
        "@kokoro_tts//:tokenizer",          # text -> phoneme token IDs
        "@kokoro_tts//:kokoro_inference",    # token IDs -> audio samples
        "@kokoro_tts//:audio_player",        # audio samples -> speaker output
    ],
)
```

Pick only what you need — they are decoupled by design.

## Available targets

| Target | Purpose |
|---|---|
| `@kokoro_tts//:tokenizer` | Converts text to phoneme token IDs. Requires `espeak-ng` installed on the system. |
| `@kokoro_tts//:kokoro_inference` | Runs the ONNX model to synthesize audio from token IDs. Returns raw float PCM samples at 24 kHz. |
| `@kokoro_tts//:audio_player` | Plays raw float audio through the default audio device via PortAudio. |

## Public API

### Tokenizer

```cpp
#include "Tokenizer.h"

Tokenizer tokenizer;

// Convert text to phoneme token IDs.
// lang: espeak-ng language code (e.g. "en-us", "pt-br", "fr-fr")
std::vector<int64_t> tokens = tokenizer.tokenize("Hello world", "en-us");
```

- Shells out to `espeak-ng` for phoneme conversion — it must be on `$PATH`.
- Output tokens are capped at 510 (the model accepts up to 512 including padding).

### KokoroInference

```cpp
#include "KokoroInference.h"

// Load the ONNX model and a voice style file.
KokoroInference tts("kokoro-v1.0.onnx", "voices/af.bin");

// Synthesize audio from token IDs.
// speed: playback speed multiplier (1.0 = normal).
// Returns PCM float samples at 24 kHz, mono.
std::vector<float> audio = tts.synthesize(tokens, /*speed=*/1.0f);
```

- The model file (`kokoro-v1.0.onnx`) and voice file (`voices/af.bin`) must be provided at runtime.
- Voice files contain style embeddings; different `.bin` files produce different speaker voices.

### AudioPlayer

```cpp
#include "AudioPlayer.h"

// Play PCM float audio through the default output device.
// sampleRate: the model outputs at 24000 Hz.
AudioPlayer::play(audio.data(), audio.size(), 24000);
```

- Blocking call — returns after all samples have been played.
- Requires PortAudio (`libportaudio2`) installed on the system.

## Minimal example

```cpp
#include "KokoroInference.h"
#include "Tokenizer.h"
#include "AudioPlayer.h"

int main() {
    Tokenizer tokenizer;
    KokoroInference tts("kokoro-v1.0.onnx", "voices/af.bin");

    auto tokens = tokenizer.tokenize("Hello from Kokoro.", "en-us");
    auto audio = tts.synthesize(tokens, 1.0f);
    AudioPlayer::play(audio.data(), audio.size(), 24000);
}
```

## System prerequisites

These must be installed on the build/run machine:

```bash
sudo apt install espeak-ng libportaudio2 portaudio19-dev
```

- **espeak-ng** — used at runtime for text-to-phoneme conversion.
- **portaudio** — used at runtime for audio playback (only needed if using `audio_player`).

## Build and run the standalone binary

```bash
bazel build //:kokoro_tts
bazel run //:kokoro_tts
```
