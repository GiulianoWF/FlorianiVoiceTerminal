#include "whisper_stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

static constexpr int   SAMPLE_RATE            = 16000;
static constexpr int   MAX_PROMPT_TOKENS      = 224;
static constexpr float VAD_RMS_THRESHOLD      = 0.005f;
static constexpr int   MAX_HALLUCINATION_REPS = 2;

// ─── AudioRingBuffer implementation ───

AudioRingBuffer::AudioRingBuffer(int max_samples)
    : m_buf(max_samples, 0.0f), m_capacity(max_samples) {}

void AudioRingBuffer::push(const float* data, int frames) {
    std::lock_guard<std::mutex> lock(m_mtx);
    for (int i = 0; i < frames; i++) {
        m_buf[m_head] = data[i];
        m_head = (m_head + 1) % m_capacity;
        if (m_size < m_capacity) {
            m_size++;
        } else {
            m_tail = (m_tail + 1) % m_capacity;
        }
    }
}

void AudioRingBuffer::get_last(int n, std::vector<float>& out) const {
    std::lock_guard<std::mutex> lock(m_mtx);
    n = std::min(n, m_size);
    out.resize(n);
    int start = (m_head - n + m_capacity) % m_capacity;
    for (int i = 0; i < n; i++) {
        out[i] = m_buf[(start + i) % m_capacity];
    }
}

int AudioRingBuffer::size() const {
    std::lock_guard<std::mutex> lock(m_mtx);
    return m_size;
}

void AudioRingBuffer::clear() {
    std::lock_guard<std::mutex> lock(m_mtx);
    m_head = 0;
    m_tail = 0;
    m_size = 0;
}

// ─── Static helpers ───

static std::string normalize_word(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c >= 0x80) {
            out += static_cast<char>(std::tolower(c));
        }
    }
    return out;
}

static std::vector<std::string> split_into_words(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            if (!current.empty()) {
                words.push_back(normalize_word(current));
                current.clear();
            }
        } else {
            current += text[i];
        }
    }
    if (!current.empty()) {
        words.push_back(normalize_word(current));
    }
    return words;
}

static std::vector<std::string> split_raw_words(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        } else {
            current += text[i];
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

static bool is_hallucination(const std::string& text) {
    auto words = split_into_words(text);
    if (words.size() < 6) return false;

    for (int plen = 2; plen <= std::min(8, (int)words.size() / 2); plen++) {
        int reps = 1;
        for (int i = plen; i + plen <= (int)words.size(); i += plen) {
            bool match = true;
            for (int j = 0; j < plen; j++) {
                if (words[i + j] != words[i - plen + j]) {
                    match = false;
                    break;
                }
            }
            if (match) reps++;
            else break;
        }
        if (reps > MAX_HALLUCINATION_REPS) return true;
    }
    return false;
}

static float rms_energy(const float* data, int frames) {
    float sum = 0.0f;
    for (int i = 0; i < frames; i++) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum / frames);
}

// ─── WhisperStream implementation ───

WhisperStream::WhisperStream(const WhisperStreamConfig& config, AudioSource& audio)
    : m_config(config), m_audio(audio) {}

WhisperStream::~WhisperStream() {
    if (m_ctx) {
        whisper_free(m_ctx);
        m_ctx = nullptr;
    }
}

bool WhisperStream::init() {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = m_config.use_gpu;
    cparams.flash_attn = m_config.flash_attn;

    m_ctx = whisper_init_from_file_with_params(m_config.model_path.c_str(), cparams);
    if (!m_ctx) return false;

    m_eot_token = whisper_token_eot(m_ctx);

    m_n_samples_step = (m_config.step_ms * SAMPLE_RATE) / 1000;
    m_n_samples_len  = (m_config.length_ms * SAMPLE_RATE) / 1000;
    m_n_samples_keep = (m_config.keep_ms * SAMPLE_RATE) / 1000;

    return true;
}

void WhisperStream::stop() {
    m_running = false;
}

void WhisperStream::reset_window() {
    if ((int)m_pcmf32.size() > m_n_samples_keep) {
        m_pcmf32_old = std::vector<float>(
            m_pcmf32.end() - m_n_samples_keep, m_pcmf32.end());
    } else {
        m_pcmf32_old.clear();
    }
}

void WhisperStream::finalize_utterance(StreamState& state, const std::string& tentative,
                                       WhisperStreamCallback& callback) {
    state.committed_text += tentative.empty() ? "" :
        (state.committed_text.empty() ? tentative : " " + tentative);

    WhisperStreamResult result;
    result.committed   = state.committed_text;
    result.finalized   = true;
    callback(result);

    if ((int)state.prompt_token_ids.size() > MAX_PROMPT_TOKENS) {
        state.prompt_token_ids.erase(
            state.prompt_token_ids.begin(),
            state.prompt_token_ids.end() - MAX_PROMPT_TOKENS);
    }

    state.committed_text.clear();
    state.last_tentative_text.clear();
    state.pass_history.clear();
    state.utterance_active = false;
    state.consecutive_silence_steps = 0;
}

void WhisperStream::run(WhisperStreamCallback callback) {
    m_running = true;

    StreamState state;
    const int n_new_line = std::max(1, m_config.length_ms / m_config.step_ms - 1);
    int n_iter = 0;

    while (m_running) {
        // Wait for enough new audio
        while (m_running) {
            if (m_audio.available() >= m_n_samples_step) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!m_running) break;

        // Grab new audio
        m_audio.get_audio(m_n_samples_step, m_pcmf32_new);
        m_audio.clear();

        if ((int)m_pcmf32_new.size() > 2 * m_n_samples_step) {
            continue; // dropping audio
        }

        // VAD check
        float energy = rms_energy(m_pcmf32_new.data(), m_pcmf32_new.size());
        if (energy < VAD_RMS_THRESHOLD) {
            state.consecutive_silence_steps++;
            if (state.utterance_active &&
                state.consecutive_silence_steps >= m_config.silence_steps) {
                finalize_utterance(state, state.last_tentative_text, callback);
                m_pcmf32_old.clear();
                n_iter = 0;
            }
            continue;
        }
        state.consecutive_silence_steps = 0;
        state.utterance_active = true;

        // Build sliding window
        const int n_samples_new = m_pcmf32_new.size();
        const int n_samples_take = std::min(
            (int)m_pcmf32_old.size(),
            std::max(0, m_n_samples_keep + m_n_samples_len - n_samples_new));

        m_pcmf32.resize(n_samples_new + n_samples_take);
        for (int i = 0; i < n_samples_take; i++) {
            m_pcmf32[i] = m_pcmf32_old[m_pcmf32_old.size() - n_samples_take + i];
        }
        std::memcpy(m_pcmf32.data() + n_samples_take,
                     m_pcmf32_new.data(), n_samples_new * sizeof(float));
        m_pcmf32_old = m_pcmf32;

        // Whisper inference
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_progress   = false;
        wparams.print_special    = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.translate        = false;
        wparams.single_segment   = true;
        wparams.max_tokens       = 0;
        wparams.language         = m_config.language.c_str();
        wparams.n_threads        = m_config.n_threads;
        wparams.no_context       = true;
        wparams.suppress_blank   = true;
        wparams.temperature_inc  = 0.0f;

        auto t_start = std::chrono::high_resolution_clock::now();

        if (whisper_full(m_ctx, wparams, m_pcmf32.data(), m_pcmf32.size()) != 0) {
            continue;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        int inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();

        // Extract text
        std::string pass_text;
        const int n_segments = whisper_full_n_segments(m_ctx);
        for (int seg = 0; seg < n_segments; ++seg) {
            pass_text += whisper_full_get_segment_text(m_ctx, seg);
        }

        if (is_hallucination(pass_text)) continue;

        // Stability algorithm
        std::vector<std::string> pass_words = split_into_words(pass_text);

        state.pass_history.push_back(pass_words);
        if ((int)state.pass_history.size() > m_config.stable_passes + 1) {
            state.pass_history.pop_front();
        }

        // Find stable prefix
        int stable_count = 0;
        if ((int)state.pass_history.size() >= m_config.stable_passes) {
            const auto& latest = state.pass_history.back();
            int min_len = (int)latest.size();
            for (const auto& prev : state.pass_history) {
                min_len = std::min(min_len, (int)prev.size());
            }
            for (int w = 0; w < min_len; w++) {
                bool all_match = true;
                for (const auto& prev : state.pass_history) {
                    if (prev[w] != latest[w]) {
                        all_match = false;
                        break;
                    }
                }
                if (all_match) {
                    stable_count = w + 1;
                } else {
                    break;
                }
            }
        }

        // Split raw words for display
        std::vector<std::string> raw_words = split_raw_words(pass_text);

        int already_committed_words = (int)split_into_words(state.committed_text).size();
        int new_stable = std::max(0, stable_count - already_committed_words);

        // Commit newly stable words
        if (new_stable > 0) {
            for (int w = already_committed_words;
                 w < already_committed_words + new_stable && w < (int)raw_words.size(); w++) {
                if (!state.committed_text.empty()) {
                    state.committed_text += " ";
                }
                state.committed_text += raw_words[w];
            }

            state.prompt_token_ids.clear();
            for (int seg = 0; seg < n_segments; ++seg) {
                const int n_tok = whisper_full_n_tokens(m_ctx, seg);
                for (int tok = 0; tok < n_tok; ++tok) {
                    whisper_token_data tdata = whisper_full_get_token_data(m_ctx, seg, tok);
                    if (tdata.id < m_eot_token) {
                        state.prompt_token_ids.push_back(tdata.id);
                    }
                }
            }
        }

        // Build tentative text
        std::string tentative_text;
        for (int w = stable_count; w < (int)raw_words.size(); w++) {
            if (!tentative_text.empty()) tentative_text += " ";
            tentative_text += raw_words[w];
        }
        state.last_tentative_text = tentative_text;

        // Deliver result via callback
        WhisperStreamResult result;
        result.committed    = state.committed_text;
        result.tentative    = tentative_text;
        result.inference_ms = inference_ms;
        result.finalized    = false;
        callback(result);

        ++n_iter;

        // Periodic window reset
        if ((n_iter % n_new_line) == 0) {
            if (!state.committed_text.empty()) {
                // Finalize current line
                state.committed_text += tentative_text.empty() ? "" : " " + tentative_text;

                WhisperStreamResult fin_result;
                fin_result.committed    = state.committed_text;
                fin_result.inference_ms = inference_ms;
                fin_result.finalized    = true;
                callback(fin_result);

                if ((int)state.prompt_token_ids.size() > MAX_PROMPT_TOKENS) {
                    state.prompt_token_ids.erase(
                        state.prompt_token_ids.begin(),
                        state.prompt_token_ids.end() - MAX_PROMPT_TOKENS);
                }

                state.committed_text.clear();
                state.last_tentative_text.clear();
                state.pass_history.clear();
            }

            reset_window();
        }

        // Length-based forced finalization
        if ((int)state.committed_text.size() > m_config.max_committed_len) {
            finalize_utterance(state, tentative_text, callback);
            reset_window();
            n_iter = 0;
        }
    }

    // Flush remaining text
    if (!state.committed_text.empty() || !state.last_tentative_text.empty()) {
        state.committed_text += state.last_tentative_text.empty() ? "" :
            (state.committed_text.empty() ? state.last_tentative_text
                                          : " " + state.last_tentative_text);
        WhisperStreamResult result;
        result.committed  = state.committed_text;
        result.finalized  = true;
        callback(result);
    }
}
