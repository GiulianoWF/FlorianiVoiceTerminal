#include "TtsEngine.h"
#include "KokoroInference.h"
#include "Tokenizer.h"
#include <iostream>
#include <chrono>

TtsEngine::TtsEngine(const std::string& model_path, const std::string& default_voice_path)
    : current_voice_path_(default_voice_path) {
    kokoro_ = new KokoroInference(model_path, default_voice_path);
    tokenizer_ = new Tokenizer();
}

TtsEngine::~TtsEngine() {
    delete kokoro_;
    delete tokenizer_;
}

std::vector<float> TtsEngine::synthesize(const std::string& text,
                                          const std::string& voice_path,
                                          const std::string& language,
                                          std::atomic<bool>* interrupt,
                                          float speed) {
    std::string cleaned = clean_for_tts(text);
    if (cleaned.empty()) return {};

    std::lock_guard<std::mutex> lock(mutex_);

    // Switch voice if needed (only reloads the .bin embedding, ONNX model stays loaded)
    if (voice_path != current_voice_path_) {
        try {
            kokoro_->set_voice(voice_path);
            current_voice_path_ = voice_path;
            std::cerr << "[TTS] Voice switched to: " << voice_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[TTS] Voice switch failed: " << e.what() << std::endl;
            return {};
        }
    }

    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer_->tokenize(cleaned, language);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (interrupt && interrupt->load()) return {};

        auto audio = kokoro_->synthesize(tokens, speed);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (interrupt && interrupt->load()) return {};

        float audio_duration = (float)audio.size() / 24000.0f;
        auto tok_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto syn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cerr << "[TTS synth] \"" << cleaned.substr(0, 60)
                  << (cleaned.size() > 60 ? "..." : "") << "\""
                  << " | tok:" << tok_ms << "ms"
                  << " | syn:" << syn_ms << "ms"
                  << " | audio:" << audio_duration << "s" << std::endl;

        return audio;
    } catch (const std::exception& e) {
        std::cerr << "[TTS] Error: " << e.what() << std::endl;
        return {};
    }
}

// Moved from main.cpp — strips markdown formatting and emojis for cleaner TTS output.
std::string TtsEngine::clean_for_tts(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        // Strip * (markdown bold/italic)
        if (c == '*') { i++; continue; }
        // Strip # (markdown headers)
        if (c == '#') { i++; continue; }
        // Determine UTF-8 character length
        int len = 1;
        if      ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (len > 1 && i + len <= text.size()) {
            // Decode codepoint
            uint32_t cp = 0;
            if (len == 2) cp = (c & 0x1F) << 6  | (text[i+1] & 0x3F);
            if (len == 3) cp = (c & 0x0F) << 12 | (text[i+1] & 0x3F) << 6  | (text[i+2] & 0x3F);
            if (len == 4) cp = (c & 0x07) << 18 | (text[i+1] & 0x3F) << 12 | (text[i+2] & 0x3F) << 6 | (text[i+3] & 0x3F);
            // Keep Latin/common punctuation (up to U+024F), skip emojis and symbols above
            if (cp <= 0x024F || (cp >= 0x2000 && cp <= 0x206F)) {
                out.append(text, i, len);
            }
            i += len;
        } else {
            if (c < 0x80) out += (char)c;
            i++;
        }
    }
    return out;
}
