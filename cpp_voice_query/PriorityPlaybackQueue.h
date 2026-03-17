#pragma once

#include "SpeechRequest.h"
#include <functional>
#include <atomic>
#include <memory>

class TtsEngine;

// Accepts SpeechRequests, synthesizes them via TtsEngine, and plays audio
// in priority order. Higher-priority requests interrupt lower-priority playback.
//
// One concept: the ordered pipeline from text to speaker.
//
// Thread-safe: submit() can be called from any thread (HTTP handlers, mic loop, etc.).
// Internally runs a single worker thread that processes one request at a time.
//
// Priority interruption:
//   When submit() receives a request with priority > what is currently playing,
//   it sets an interrupt flag. AudioPlayer checks this every 256 samples (~10ms),
//   then the worker picks the highest-priority item next.
//
class PriorityPlaybackQueue {
public:
    // The play function receives (samples, count, sampleRate, interrupt_flag).
    // Injected to avoid a hard dependency on AudioPlayer — pass AudioPlayer::play here.
    using PlayFn = std::function<void(const float*, size_t, int, std::atomic<bool>*)>;

    PriorityPlaybackQueue(TtsEngine& engine, PlayFn play_fn);
    ~PriorityPlaybackQueue();

    // Submit text to be spoken. Thread-safe. May interrupt current playback if higher priority.
    void submit(SpeechRequest request);

    // Start the worker thread. Call once after construction.
    void start();

    // Stop processing. Interrupts current playback and drains the queue.
    void stop();

    // Interrupt whatever is currently playing (e.g., called by VAD when user speaks).
    void interrupt_current();

    // True if currently playing or has queued items.
    bool is_active() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
