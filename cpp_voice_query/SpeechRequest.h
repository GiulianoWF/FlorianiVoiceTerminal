#pragma once

#include <string>
#include <cstdint>

// How urgently should this text be spoken?
// Higher priority interrupts lower priority mid-playback.
enum class Priority : int { LOW = 0, NORMAL = 1, HIGH = 2, CRITICAL = 3 };

// A request to speak some text.
// One concept: what to say, how urgently, and in which voice.
struct SpeechRequest {
    std::string text;
    Priority    priority    = Priority::NORMAL;
    std::string voice_path;   // path to Kokoro .bin voice file
    std::string language;     // espeak-ng language code, e.g. "pt-br", "en-us"
    std::string channel;      // source channel identifier (e.g. "mic", "alert", "monitor")
    uint64_t    sequence    = 0; // monotonic ordering within same priority (assigned by queue)

    // Per-request response capture: after playing, capture user speech and POST to callback_url
    bool        wait_response  = false;
    std::string callback_url;

    // For std::priority_queue: higher priority first, then earlier sequence first.
    bool operator<(const SpeechRequest& other) const {
        if (priority != other.priority)
            return static_cast<int>(priority) < static_cast<int>(other.priority);
        return sequence > other.sequence;  // lower sequence = older = should play first
    }
};
