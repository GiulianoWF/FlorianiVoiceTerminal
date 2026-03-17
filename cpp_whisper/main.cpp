#include "whisper_stream.h"
#include <portaudio.h>

#include <csignal>
#include <iostream>
#include <string>

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

// ─── PortAudio-based audio source ───

class PortAudioSource : public AudioSource {
public:
    explicit PortAudioSource(int buffer_samples)
        : m_ring(buffer_samples) {}

    ~PortAudioSource() override { stop(); }

    bool start() override {
        PaError err = Pa_Initialize();
        if (err != paNoError) return false;
        m_pa_initialized = true;

        err = Pa_OpenDefaultStream(
            &m_stream, 1, 0, paFloat32,
            16000, 1024,
            pa_callback, this);

        if (err != paNoError) return false;

        err = Pa_StartStream(m_stream);
        return err == paNoError;
    }

    void stop() override {
        if (m_stream) {
            Pa_StopStream(m_stream);
            Pa_CloseStream(m_stream);
            m_stream = nullptr;
        }
        if (m_pa_initialized) {
            Pa_Terminate();
            m_pa_initialized = false;
        }
    }

    int get_audio(int n, std::vector<float>& out) override {
        m_ring.get_last(n, out);
        return (int)out.size();
    }

    void clear() override { m_ring.clear(); }
    int  available() override { return m_ring.size(); }

private:
    static int pa_callback(const void* input, void* /*output*/,
                           unsigned long frameCount,
                           const PaStreamCallbackTimeInfo* /*timeInfo*/,
                           PaStreamCallbackFlags /*flags*/,
                           void* userData) {
        auto* self = static_cast<PortAudioSource*>(userData);
        self->m_ring.push(static_cast<const float*>(input),
                          static_cast<int>(frameCount));
        return paContinue;
    }

    AudioRingBuffer m_ring;
    PaStream*       m_stream = nullptr;
    bool            m_pa_initialized = false;
};

// ─── ANSI display ───

static constexpr const char* ANSI_RESET = "\033[0m";
static constexpr const char* ANSI_DIM   = "\033[2m";

// ─── Main ───

int main(int argc, char* argv[]) {
    WhisperStreamConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if ((arg == "-l" || arg == "--language") && i + 1 < argc) {
            config.language = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            config.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-gpu" || arg == "--gpu") {
            config.use_gpu = true;
        } else if (arg == "-fa" || arg == "--flash-attn") {
            config.flash_attn = true;
        } else if ((arg == "--step") && i + 1 < argc) {
            config.step_ms = std::stoi(argv[++i]);
        } else if ((arg == "--length") && i + 1 < argc) {
            config.length_ms = std::stoi(argv[++i]);
        } else if ((arg == "--keep") && i + 1 < argc) {
            config.keep_ms = std::stoi(argv[++i]);
        } else if ((arg == "--stable") && i + 1 < argc) {
            config.stable_passes = std::stoi(argv[++i]);
        } else if ((arg == "--silence") && i + 1 < argc) {
            config.silence_steps = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cerr << "Usage: " << argv[0] << " [options]\n"
                      << "  -m,   --model PATH      Model path (default: " << config.model_path << ")\n"
                      << "  -l,   --language LANG    Language (default: " << config.language << ")\n"
                      << "  -t,   --threads N        Threads (default: " << config.n_threads << ")\n"
                      << "  -gpu, --gpu              Enable GPU inference\n"
                      << "  -fa,  --flash-attn       Enable flash attention\n"
                      << "\n  Streaming tuning:\n"
                      << "  --step N                 Audio step in ms (default: " << config.step_ms << ")\n"
                      << "  --length N               Audio window in ms (default: " << config.length_ms << ")\n"
                      << "  --keep N                 Overlap in ms (default: " << config.keep_ms << ")\n"
                      << "  --stable N               Passes to commit (default: " << config.stable_passes << ")\n"
                      << "  --silence N              Silent steps to finalize (default: " << config.silence_steps << ")\n";
            return 0;
        } else {
            config.model_path = arg;
        }
    }

    signal(SIGINT, signal_handler);

    std::cout << "Loading model: " << config.model_path << std::endl;

    // Audio source (PortAudio) — created but NOT started until model is loaded
    int buffer_samples = (config.length_ms * 16000) / 1000 * 2;
    PortAudioSource audio(buffer_samples);

    // Whisper streaming engine — load model first (can take seconds)
    WhisperStream stream(config, audio);
    if (!stream.init()) {
        std::cerr << "Failed to load model: " << config.model_path << std::endl;
        return 1;
    }

    // Start audio capture AFTER model is loaded (matches old behavior)
    if (!audio.start()) {
        std::cerr << "Failed to initialize audio\n";
        return 1;
    }

    std::cout << "\nStreaming mode: step=" << config.step_ms
              << "ms, window=" << config.length_ms
              << "ms, keep=" << config.keep_ms
              << "ms, threads=" << config.n_threads << "\n";
    std::cout << "[Start speaking]\n" << std::flush;

    // Run with display callback
    stream.run([&](const WhisperStreamResult& r) {
        if (!g_running) {
            stream.stop();
            return;
        }

        if (r.finalized) {
            printf("\33[2K\r%s\n", r.committed.c_str());
        } else {
            printf("\33[2K\r");
            if (!r.committed.empty()) {
                printf("%s", r.committed.c_str());
                if (!r.tentative.empty()) printf(" ");
            }
            printf("%s%s%s", ANSI_DIM, r.tentative.c_str(), ANSI_RESET);
            printf("  [%dms]", r.inference_ms);
        }
        fflush(stdout);
    });

    printf("\n");
    audio.stop();

    return 0;
}
