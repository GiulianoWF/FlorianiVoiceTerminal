#pragma once

#include "SpeechRequest.h"
#include <string>

// A named channel that an agent can POST text to.
// One concept: a voice identity with a priority level.
//
// Each channel becomes an HTTP endpoint: POST /speak/{name}
// Example channels:
//   { "assistant", "voices/pm_santa.bin", Priority::NORMAL, "pt-br" }
//   { "alert",     "voices/af.bin",       Priority::HIGH,   "en-us" }
//   { "monitor",   "voices/am_adam.bin",   Priority::LOW,    "en-us" }
//
struct VoiceChannel {
    std::string name;         // URL segment, e.g. "assistant", "alert"
    std::string voice_path;   // Kokoro voice .bin file
    Priority    priority;
    std::string language;     // espeak-ng language code
};
