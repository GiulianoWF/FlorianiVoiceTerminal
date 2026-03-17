#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>

class KokoroInference;
class Tokenizer;

// Text in, audio samples out.
// Wraps Kokoro TTS + espeak-ng tokenizer with runtime voice switching.
// Thread-safe: serializes synthesis via mutex (ONNX session is not thread-safe).
class TtsEngine {
public:
    TtsEngine(const std::string& model_path, const std::string& default_voice_path);
    ~TtsEngine();

    // Synthesize text into 24kHz mono float PCM.
    // Switches voice automatically if voice_path differs from the current one.
    // Returns empty vector if text is empty after cleaning, or on error.
    // Checks interrupt between tokenization and synthesis for early abort.
    std::vector<float> synthesize(const std::string& text,
                                   const std::string& voice_path,
                                   const std::string& language,
                                   std::atomic<bool>* interrupt = nullptr,
                                   float speed = 1.3f);

private:
    KokoroInference* kokoro_;
    Tokenizer* tokenizer_;
    std::string current_voice_path_;
    std::mutex mutex_;

    // Strip markdown bold, headers, emojis — keep only speakable text.
    static std::string clean_for_tts(const std::string& text);
};
