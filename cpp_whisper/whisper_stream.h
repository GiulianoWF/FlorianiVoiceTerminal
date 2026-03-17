#pragma once

#include "whisper.h"

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

// ─── Audio source interface (dependency injection) ───

class AudioSource {
public:
    virtual ~AudioSource() = default;
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual int  get_audio(int n, std::vector<float>& out) = 0;
    virtual void clear() = 0;
    virtual int  available() = 0;
};

// ─── Thread-safe ring buffer (reusable by audio source implementations) ───

class AudioRingBuffer {
public:
    explicit AudioRingBuffer(int max_samples);

    void push(const float* data, int frames);
    void get_last(int n, std::vector<float>& out) const;
    int  size() const;
    void clear();

private:
    std::vector<float> m_buf;
    int m_capacity;
    int m_head = 0;
    int m_tail = 0;
    int m_size = 0;
    mutable std::mutex m_mtx;
};

// ─── Configuration ───

struct WhisperStreamConfig {
    std::string model_path = "models/ggml-large-v3.bin";
    std::string language   = "pt";
    int  n_threads         = 4;
    bool use_gpu           = false;
    bool flash_attn        = false;
    int  step_ms           = 500;
    int  length_ms         = 5000;
    int  keep_ms           = 200;
    int  stable_passes     = 3;
    int  silence_steps     = 3;
    int  max_committed_len = 300;
};

// ─── Result delivered to callback ───

struct WhisperStreamResult {
    std::string committed;    // stable text (won't change)
    std::string tentative;    // may change next pass
    int         inference_ms = 0;
    bool        finalized    = false; // true = utterance ended, time for newline
};

using WhisperStreamCallback = std::function<void(const WhisperStreamResult&)>;

// ─── Streaming transcription engine ───

class WhisperStream {
public:
    WhisperStream(const WhisperStreamConfig& config, AudioSource& audio);
    ~WhisperStream();

    WhisperStream(const WhisperStream&) = delete;
    WhisperStream& operator=(const WhisperStream&) = delete;

    bool init();
    void run(WhisperStreamCallback callback);
    void stop();

private:
    struct StreamState {
        std::string              committed_text;
        std::vector<whisper_token> prompt_token_ids;
        std::string              last_tentative_text;
        std::deque<std::vector<std::string>> pass_history;
        int  consecutive_silence_steps = 0;
        bool utterance_active          = false;
    };

    void finalize_utterance(StreamState& state, const std::string& tentative,
                            WhisperStreamCallback& callback);
    void reset_window();

    WhisperStreamConfig m_config;
    AudioSource&        m_audio;
    whisper_context*    m_ctx       = nullptr;
    whisper_token       m_eot_token = 0;
    std::atomic<bool>   m_running{false};

    // Audio buffers
    std::vector<float> m_pcmf32;
    std::vector<float> m_pcmf32_old;
    std::vector<float> m_pcmf32_new;

    // Sample counts (computed from config in init)
    int m_n_samples_step = 0;
    int m_n_samples_len  = 0;
    int m_n_samples_keep = 0;
};
