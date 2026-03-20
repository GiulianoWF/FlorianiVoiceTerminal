#include "whisper_stream.h"
#include "PortAudioSource.h"
#include "AudioPlayer.h"
#include "LlmClient.h"
#include "TtsEngine.h"
#include "PriorityPlaybackQueue.h"
#include "InboundServer.h"
#include "VoiceChannel.h"
#include "SpeechRequest.h"

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <algorithm>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

// --- Global state ---
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_responding{false};
static WhisperStream*    g_stream = nullptr;
static PriorityPlaybackQueue* g_queue = nullptr;

static void signal_handler(int) {
    if (g_responding) {
        if (g_queue) g_queue->interrupt_current();
    } else {
        g_running = false;
        if (g_stream) g_stream->stop();
    }
}

static bool is_exit_word(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    size_t start = lower.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return false;
    size_t end = lower.find_last_not_of(" \t\n\r");
    lower = lower.substr(start, end - start + 1);
    return lower == "exit" || lower == "quit" || lower == "stop" || lower == "goodbye"
        || lower == "sair" || lower == "parar" || lower == "tchau";
}

static constexpr const char* ANSI_RESET = "\033[0m";
static constexpr const char* ANSI_DIM   = "\033[2m";

// --- Callback POST helper ---

static void post_json_to_url(const std::string& url, const std::string& json_body) {
    CURL* curl = curl_easy_init();
    if (!curl) return;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[Callback] POST to " << url << " failed: "
                  << curl_easy_strerror(res) << std::endl;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

// Capture user speech via STT (with 30s timeout), POST to callback, release queue hold.
static void capture_and_send_response(WhisperStream& stream, PortAudioSource& audio_source,
                                      PriorityPlaybackQueue& queue) {
    std::string callback_url = queue.get_pending_callback_url();
    if (callback_url.empty()) {
        queue.complete_response();
        return;
    }

    audio_source.clear();
    std::string response_text;
    auto start = std::chrono::steady_clock::now();

    std::cout << "[Listening for response...]\n" << std::flush;
    stream.run([&](const WhisperStreamResult& r) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > std::chrono::seconds(30)) {
            stream.stop();
            return;
        }
        if (r.finalized) {
            printf("\33[2K\r%s\n", r.committed.c_str());
            fflush(stdout);
            response_text = r.committed;
            stream.stop();
        } else {
            printf("\33[2K\r%s%s%s", ANSI_DIM, r.tentative.c_str(), ANSI_RESET);
            fflush(stdout);
        }
    });

    if (!response_text.empty()) {
        std::cout << "   Response: " << response_text << std::endl;
    } else {
        std::cout << "   [No response — timeout]" << std::endl;
    }

    nlohmann::json j;
    j["response"] = response_text;
    post_json_to_url(callback_url, j.dump());

    queue.complete_response();
    std::cerr << "[Response sent to " << callback_url << "]" << std::endl;
}

// --- Configuration ---

struct Config {
    // Whisper STT
    std::string whisper_model = "../cpp_whisper/models/ggml-base.bin";
    std::string language = "pt";
    int  n_threads     = 4;
    bool use_gpu       = false;
    bool flash_attn    = false;
    int  step_ms       = 500;
    int  length_ms     = 15000;
    int  keep_ms       = 200;
    int  stable_passes = 3;
    int  silence_steps = 1;

    // LLM
    std::string llm_url = "http://localhost:8001";

    // Kokoro TTS
    std::string kokoro_model = "../cpp_kokoro/build/kokoro-v1.0.onnx";
    std::string kokoro_voice = "../cpp_kokoro/build/voices/pm_santa.bin";

    // Voice server
    int server_port = 8090;
    std::vector<VoiceChannel> channels;  // populated from --channel flags or defaults

    // Flags
    bool no_tts = false;
    bool text_only = false;
};

// Parse "name:voice_path:priority:language" into a VoiceChannel
static VoiceChannel parse_channel(const std::string& spec) {
    VoiceChannel ch;
    size_t p1 = spec.find(':');
    size_t p2 = spec.find(':', p1 + 1);
    size_t p3 = spec.find(':', p2 + 1);
    ch.name = spec.substr(0, p1);
    ch.voice_path = spec.substr(p1 + 1, p2 - p1 - 1);
    std::string prio_str = spec.substr(p2 + 1, p3 - p2 - 1);
    ch.language = spec.substr(p3 + 1);

    if      (prio_str == "low")      ch.priority = Priority::LOW;
    else if (prio_str == "normal")   ch.priority = Priority::NORMAL;
    else if (prio_str == "high")     ch.priority = Priority::HIGH;
    else if (prio_str == "critical") ch.priority = Priority::CRITICAL;
    else                             ch.priority = Priority::NORMAL;

    return ch;
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--whisper-model" && i + 1 < argc) {
            cfg.whisper_model = argv[++i];
        } else if (arg == "--language" && i + 1 < argc) {
            cfg.language = argv[++i];
        } else if (arg == "--llm-url" && i + 1 < argc) {
            cfg.llm_url = argv[++i];
        } else if (arg == "--kokoro-model" && i + 1 < argc) {
            cfg.kokoro_model = argv[++i];
        } else if (arg == "--kokoro-voice" && i + 1 < argc) {
            cfg.kokoro_voice = argv[++i];
        } else if (arg == "--server-port" && i + 1 < argc) {
            cfg.server_port = std::stoi(argv[++i]);
        } else if (arg == "--channel" && i + 1 < argc) {
            cfg.channels.push_back(parse_channel(argv[++i]));
        } else if (arg == "--no-tts") {
            cfg.no_tts = true;
        } else if (arg == "--text-only") {
            cfg.text_only = true;
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            cfg.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-gpu" || arg == "--gpu") {
            cfg.use_gpu = true;
        } else if (arg == "-fa" || arg == "--flash-attn") {
            cfg.flash_attn = true;
        } else if (arg == "--step" && i + 1 < argc) {
            cfg.step_ms = std::stoi(argv[++i]);
        } else if (arg == "--length" && i + 1 < argc) {
            cfg.length_ms = std::stoi(argv[++i]);
        } else if (arg == "--keep" && i + 1 < argc) {
            cfg.keep_ms = std::stoi(argv[++i]);
        } else if (arg == "--stable" && i + 1 < argc) {
            cfg.stable_passes = std::stoi(argv[++i]);
        } else if (arg == "--silence" && i + 1 < argc) {
            cfg.silence_steps = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: voice_query [options]\n"
                      << "  --whisper-model PATH  Whisper GGUF model\n"
                      << "  --language LANG       Language: en, pt (default: pt)\n"
                      << "  --llm-url URL         llama.cpp server (default: http://localhost:8001)\n"
                      << "  --kokoro-model PATH   Kokoro ONNX model\n"
                      << "  --kokoro-voice PATH   Kokoro voice .bin file (default voice)\n"
                      << "  --server-port PORT    HTTP server port (default: 8090)\n"
                      << "  --channel SPEC        Channel: name:voice_path:priority:language\n"
                      << "                        e.g. alert:voices/af.bin:high:en-us\n"
                      << "  --no-tts              Disable spoken responses\n"
                      << "  --text-only           Just transcribe, skip LLM\n"
                      << "\n  Whisper streaming:\n"
                      << "  -t, --threads N       CPU threads (default: 4)\n"
                      << "  -gpu, --gpu           Enable GPU inference\n"
                      << "  -fa, --flash-attn     Enable flash attention\n"
                      << "  --step N              Audio step in ms (default: 500)\n"
                      << "  --length N            Audio window in ms (default: 5000)\n"
                      << "  --keep N              Overlap in ms (default: 200)\n"
                      << "  --stable N            Passes to commit (default: 3)\n"
                      << "  --silence N           Silent steps to finalize (default: 3)\n";
            exit(0);
        }
    }

    // Default channels if none specified
    std::string tts_lang = (cfg.language == "pt") ? "pt-br" : "en-us";
    if (cfg.channels.empty()) {
        cfg.channels.push_back({"assistant", cfg.kokoro_voice, Priority::NORMAL, tts_lang});
        cfg.channels.push_back({"alert", cfg.kokoro_voice, Priority::HIGH, "en-us"});
    }

    return cfg;
}

// --- Main ---

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);
    signal(SIGINT, signal_handler);

    std::string tts_lang = (cfg.language == "pt") ? "pt-br" : "en-us";
    std::cout << "[Config] Language: " << cfg.language << ", TTS lang: " << tts_lang << std::endl;

    // --- Init PortAudio ---
    if (Pa_Initialize() != paNoError) {
        std::cerr << "Failed to initialize PortAudio" << std::endl;
        return 1;
    }

    // --- Init WhisperStream ---
    WhisperStreamConfig whisper_cfg;
    whisper_cfg.model_path    = cfg.whisper_model;
    whisper_cfg.language      = cfg.language;
    whisper_cfg.n_threads     = cfg.n_threads;
    whisper_cfg.use_gpu       = cfg.use_gpu;
    whisper_cfg.flash_attn    = cfg.flash_attn;
    whisper_cfg.step_ms       = cfg.step_ms;
    whisper_cfg.length_ms     = cfg.length_ms;
    whisper_cfg.keep_ms       = cfg.keep_ms;
    whisper_cfg.stable_passes = cfg.stable_passes;
    whisper_cfg.silence_steps = cfg.silence_steps;

    int buffer_samples = (whisper_cfg.length_ms * 16000) / 1000 * 2;
    PortAudioSource audio_source(buffer_samples);

    std::cout << "[STT] Loading Whisper: " << cfg.whisper_model << std::endl;
    WhisperStream stream(whisper_cfg, audio_source);
    g_stream = &stream;
    if (!stream.init()) {
        std::cerr << "Failed to load Whisper model: " << cfg.whisper_model << std::endl;
        Pa_Terminate();
        return 1;
    }
    std::cout << "[STT] Ready (step=" << cfg.step_ms
              << "ms, window=" << cfg.length_ms
              << "ms, stable=" << cfg.stable_passes << ")" << std::endl;

    // --- Init TTS engine + playback queue ---
    TtsEngine* tts = nullptr;
    PriorityPlaybackQueue* queue = nullptr;
    InboundServer* server = nullptr;

    if (!cfg.no_tts && !cfg.text_only) {
        std::cout << "[TTS] Loading Kokoro: " << cfg.kokoro_model << std::endl;
        try {
            tts = new TtsEngine(cfg.kokoro_model, cfg.kokoro_voice);
            queue = new PriorityPlaybackQueue(*tts, AudioPlayer::play);
            queue->start();
            g_queue = queue;
            std::cout << "[TTS] Ready." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[TTS] Failed: " << e.what() << " — continuing without TTS" << std::endl;
        }
    }

    // --- Init HTTP server ---
    if (queue) {
        server = new InboundServer(*queue, cfg.channels, cfg.server_port);
        server->start();
        std::cout << "[Server] Channels:";
        for (const auto& ch : cfg.channels) {
            std::cout << " /speak/" << ch.name << "(prio=" << static_cast<int>(ch.priority) << ")";
        }
        std::cout << std::endl;
    }

    // --- Init LLM client ---
    LlmClient llm(cfg.llm_url);
    if (!cfg.text_only) {
        if (llm.health_check()) {
            std::cout << "[LLM] llama.cpp at " << cfg.llm_url << " — OK" << std::endl;
        } else {
            std::cout << "[LLM] WARNING: Cannot reach " << cfg.llm_url << " — continuing anyway" << std::endl;
        }
    }

    // --- VAD monitor: listens for speech during TTS to interrupt ---
    std::atomic<bool> vad_active{false};
    std::atomic<bool> llm_interrupt{false};

    auto start_vad_monitor = [&]() -> std::thread {
        vad_active = true;
        return std::thread([&]() {
            std::vector<float> vad_buf;
            while (vad_active && g_responding) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                int avail = audio_source.available();
                if (avail < 1600) continue;
                audio_source.get_audio(1600, vad_buf);

                float sum = 0;
                for (float s : vad_buf) sum += s * s;
                float rms = std::sqrt(sum / (float)vad_buf.size());

                if (rms > 0.03f) {
                    // During pending response, speech is the expected answer — don't interrupt
                    if (queue && queue->has_pending_response()) continue;

                    std::cerr << "[VAD] Speech detected (rms=" << rms << "), interrupting" << std::endl;
                    if (queue) {
                        std::string ch = queue->get_current_channel();
                        if (!ch.empty()) queue->interrupt_channel(ch);
                        queue->interrupt_channel("mic");  // Always clear LLM sentences
                    }
                    llm_interrupt = true;
                    vad_active = false;
                }
            }
        });
    };

    std::vector<Message> history;

    // Start audio capture
    if (!audio_source.start()) {
        std::cerr << "Failed to start audio capture\n";
        Pa_Terminate();
        return 1;
    }

    std::cout << "\nReady. Press Ctrl+C to exit.\n" << std::endl;

    bool keep_audio = false;
    std::vector<float> pre_vad_buf;

    // --- Main loop: STT → LLM → queue ---
    while (g_running) {
        // --- Phase 0: Handle pending channel responses ---
        if (queue && queue->has_pending_response()) {
            capture_and_send_response(stream, audio_source, *queue);
            continue;
        }

        // --- Phase 0.5: Pre-STT VAD — interrupt HTTP channels before entering STT ---
        if (queue && queue->is_active()) {
            std::cerr << "[Main] Queue active, monitoring for VAD before STT" << std::endl;
            while (queue->is_active() && g_running && !queue->has_pending_response()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                int avail = audio_source.available();
                if (avail < 1600) continue;
                audio_source.get_audio(1600, pre_vad_buf);

                float sum = 0;
                for (float s : pre_vad_buf) sum += s * s;
                float rms = std::sqrt(sum / (float)pre_vad_buf.size());

                if (rms > 0.03f) {
                    std::cerr << "[VAD-pre] Speech detected, interrupting current channel" << std::endl;
                    std::string ch = queue->get_current_channel();
                    if (!ch.empty()) queue->interrupt_channel(ch);
                    audio_source.clear();
                    // Don't break — loop back to check if other channels resume
                }
            }
            // If a pending response appeared, handle it at top of loop
            if (queue->has_pending_response()) continue;
        }

        if (!g_running) break;

        if (!keep_audio) audio_source.clear();
        keep_audio = false;

        std::string captured_text;
        std::cout << "[Listening...]\n" << std::flush;

        std::thread whisper_thread([&] {
            stream.run([&](const WhisperStreamResult& r) {
                if (!g_running) { stream.stop(); return; }
                if (r.finalized) {
                    printf("\33[2K\r%s\n", r.committed.c_str());
                    fflush(stdout);
                    captured_text = r.committed;
                    stream.stop();
                } else {
                    printf("\33[2K\r");
                    if (!r.committed.empty()) {
                        printf("%s", r.committed.c_str());
                        if (!r.tentative.empty()) printf(" ");
                    }
                    printf("%s%s%s", ANSI_DIM, r.tentative.c_str(), ANSI_RESET);
                    printf("  [%dms]", r.inference_ms);
                    fflush(stdout);
                }
            });
        });
        whisper_thread.join();

        if (!g_running || captured_text.empty()) continue;
        std::cout << "   You: " << captured_text << std::endl;
        if (is_exit_word(captured_text)) break;
        if (cfg.text_only) continue;

        // --- Phase 2: LLM + TTS via priority queue ---
        std::cout << "[LLM] ..." << std::endl;
        g_responding = true;

        // LLM on_sentence callback: submit each sentence to the priority queue
        auto on_sentence = [&](const std::string& sentence) {
            if (!queue) return;
            SpeechRequest req;
            req.text = sentence;
            req.priority = Priority::NORMAL;
            req.voice_path = cfg.kokoro_voice;
            req.language = tts_lang;
            req.channel = "mic";
            queue->submit(std::move(req));
        };

        // Start VAD monitor
        llm_interrupt = false;
        std::thread vad_thread = start_vad_monitor();

        // Stream LLM response — on_sentence fires per sentence
        std::string reply = llm.chat(captured_text, history, on_sentence, &llm_interrupt);

        // Wait for all queued audio to finish playing (or VAD interrupt)
        if (queue) {
            while (queue->is_active() && vad_active) {
                // If a channel needs a user response, capture it
                if (queue->has_pending_response()) {
                    capture_and_send_response(stream, audio_source, *queue);
                    continue;  // Keep waiting for remaining queue items
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        // Stop VAD
        vad_active = false;
        if (vad_thread.joinable()) vad_thread.join();

        bool was_interrupted = llm_interrupt.load();
        g_responding = false;

        if (!was_interrupted && (!queue || !queue->is_active())) {
            std::cout << "   Assistant: " << reply << std::endl;
        }
    }

    // --- Shutdown ---
    std::cout << "\nDone." << std::endl;

    if (server) { server->stop(); delete server; }
    if (queue)  { queue->stop();  delete queue;  }
    delete tts;

    audio_source.stop();
    g_stream = nullptr;
    g_queue = nullptr;
    Pa_Terminate();
    return 0;
}
