# cpp_whisper — Streaming Speech-to-Text with whisper.cpp

Real-time streaming speech recognition in C++17, built on [whisper.cpp](https://github.com/ggerganov/whisper.cpp). Uses a stability-based commit algorithm to produce accurate, low-latency transcriptions with committed/tentative text separation.

## Architecture

```
whisper_stream.h / .cpp    Reusable streaming library (no audio dependency)
main.cpp                   CLI app with PortAudio microphone input
BUILD.bazel                Bazel build rules (whisper_cpp, whisper_stream, whisper_stt, model downloads)
MODULE.bazel               Bazel module deps (rules_cc, rules_foreign_cc, whisper.cpp source archive)
whisper.cpp/               Upstream whisper.cpp (git clone, used by CMake build)
models/                    GGUF model files
```

### Key classes

- **`AudioSource`** — Abstract interface for audio input. Implement `start()`, `stop()`, `get_audio()`, `clear()`, `available()` to provide audio from any source.
- **`AudioRingBuffer`** — Thread-safe ring buffer for audio samples. Useful when implementing custom `AudioSource` classes.
- **`WhisperStreamConfig`** — All tuning parameters (model path, language, step/window/keep timing, stability passes, etc).
- **`WhisperStreamResult`** — Delivered to your callback each pass: `committed` (stable text), `tentative` (may change), `inference_ms`, `finalized` (utterance ended).
- **`WhisperStream`** — The streaming engine. Takes a config + any `AudioSource`, calls your callback with results.

### Stability algorithm

Words are only committed after appearing in the same position across N consecutive inference passes (`stable_passes`). This prevents mid-word cut artifacts at window boundaries. Tentative words (at the tail of the transcription window) are shown but may change. A hallucination detector skips outputs with repeating phrase patterns.

## Build

### Prerequisites

```bash
sudo apt-get install -y portaudio19-dev cmake build-essential
# For GPU: NVIDIA CUDA toolkit
# For Bazel: install Bazelisk (https://github.com/bazelbuild/bazelisk)
```

### Bazel build (recommended)

The Bazel build fetches the whisper.cpp source automatically — no manual git clone needed.

```bash
# Build and run (downloads large-v3 model by default):
bazel run //:whisper_stt_run -- -gpu --stable 3

# Use a different model variant:
bazel run //:whisper_stt_run --define whisper_model=small -- -gpu --stable 3
bazel run //:whisper_stt_run --define whisper_model=base -- -gpu --stable 3
bazel run //:whisper_stt_run --define whisper_model=large-v3-turbo -- -gpu --stable 3

# Build only (no run):
bazel build //:whisper_stt

# Download a model file without running:
bazel build //:download_ggml-large-v3
```

#### GPU support with Bazel

CUDA is enabled by default in [BUILD.bazel](BUILD.bazel). To disable it, remove `"GGML_CUDA": "ON"` and `"CMAKE_CUDA_COMPILER"` from `cache_entries`, and remove `"libggml-cuda.a"` from `out_static_libs`.

### CMake build (manual)

```bash
./rebuild_all.sh        # clones whisper.cpp, downloads model, builds
```

Or manually:

```bash
git clone https://github.com/ggerganov/whisper.cpp.git   # if not present
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
make -j$(nproc)
```

### Download models (CMake / manual)

```bash
# Pick one (larger = more accurate, slower on CPU) default is large-v3 on gpu:
curl -L -o models/ggml-base.bin          https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
curl -L -o models/ggml-small.bin         https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
curl -L -o models/ggml-large-v3.bin      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
curl -L -o models/ggml-large-v3-turbo.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

## Run the CLI

```bash
# Bazel:
bazel run //:whisper_stt_run -- -gpu --stable 3

# CMake:
./build/whisper_stt -gpu --stable 3
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | `models/ggml-large-v3.bin` | Model path |
| `-l, --language` | `pt` | Language code (`pt`, `en`, `auto`, etc) |
| `-t, --threads` | `4` | CPU threads |
| `-gpu` | off | Enable GPU inference |
| `-fa` | off | Enable flash attention |
| `--step` | `500` | Audio step in ms (how often to run inference) |
| `--length` | `5000` | Audio window in ms (context Whisper sees) |
| `--keep` | `200` | Overlap between windows in ms |
| `--stable` | `3` | Passes needed before committing a word |
| `--silence` | `3` | Silent steps before finalizing utterance |

## Using as a library in another project

`libwhisper_stream.a` has no PortAudio dependency. It only depends on `libwhisper`.

### 1. CMake integration

```cmake
add_subdirectory(path/to/cpp_whisper)
target_link_libraries(your_app whisper_stream)
```

### 1b. Bazel integration

Add `cpp_whisper` as a dependency in your `MODULE.bazel` using a local or git override:

```python
# In your MODULE.bazel:
bazel_dep(name = "cpp_whisper", version = "0.1.0")
local_path_override(
    module_name = "cpp_whisper",
    path = "../cpp_whisper",  # relative path to this project
)
```

Then depend on the `whisper_stream` target in your `BUILD.bazel`:

```python
# In your BUILD.bazel:
cc_binary(
    name = "your_app",
    srcs = ["your_app.cpp"],
    deps = ["@cpp_whisper//:whisper_stream"],
    linkopts = [
        "-lgomp",
        # For GPU support:
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-lcublas",
        "-lcublasLt",
        "-lcuda",
    ],
)
```

### 2. Implement AudioSource

```cpp
#include "whisper_stream.h"

class MyAudioSource : public AudioSource {
public:
    bool start() override { /* open your audio device/file/network */ return true; }
    void stop() override  { /* cleanup */ }
    int  get_audio(int n, std::vector<float>& out) override {
        // Fill `out` with up to `n` most recent 16kHz mono float32 samples
        // Return actual count
    }
    void clear() override { /* discard buffered samples */ }
    int  available() override { /* return buffered sample count */ }
};
```

`AudioRingBuffer` is provided for convenience — use it inside your `AudioSource` if you have a callback-based audio API.

### 3. Run the stream

```cpp
WhisperStreamConfig config;
config.model_path = "models/ggml-large-v3-turbo.bin";
config.language   = "pt";
config.use_gpu    = true;
config.step_ms    = 500;
config.stable_passes = 3;

MyAudioSource audio;
audio.start();

WhisperStream stream(config, audio);
stream.init();

stream.run([](const WhisperStreamResult& r) {
    if (r.finalized) {
        // Utterance complete — save, send to LLM, etc.
        process_utterance(r.committed);
    } else {
        // Partial result — update UI
        display(r.committed, r.tentative);
    }
});

audio.stop();
```

### 4. Stop from another thread

```cpp
// From signal handler, UI thread, etc:
stream.stop();  // run() will return
```

## Tuning guide

| Goal | Adjust |
|------|--------|
| Faster text appearance | Lower `--step` (e.g. 300) and `--stable` (e.g. 2) |
| More accurate commits | Raise `--stable` (e.g. 4) |
| Less latency, less context | Lower `--length` (e.g. 3000) |
| Longer utterances before line break | Raise `max_committed_len` in config |
| Faster silence detection | Lower `--silence` (e.g. 2) |

The commit latency is `step_ms * stable_passes`. With GPU (~70ms inference), `--step 500 --stable 3` gives 1.5s commit delay.
