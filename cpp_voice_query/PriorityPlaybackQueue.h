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

    // Interrupt whatever is currently playing (e.g., Ctrl+C signal handler — async-signal-safe).
    void interrupt_current();

    // Interrupt all items from a specific channel: stops current if it matches,
    // removes all queued items from that channel, cancels pending response if from that channel.
    void interrupt_channel(const std::string& channel);

    // Returns the channel name of the currently playing item (empty if idle).
    std::string get_current_channel() const;

    // True if currently playing, has queued items, or awaiting a response.
    bool is_active() const;

    // --- Response capture API ---

    // True if the last played item had wait_response and completed without interruption.
    // The worker is held until complete_response() is called.
    bool has_pending_response() const;

    // Returns the callback URL for the pending response.
    std::string get_pending_callback_url() const;

    // Release the worker hold after the response has been captured and sent.
    void complete_response();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
