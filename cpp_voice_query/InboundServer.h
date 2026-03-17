#pragma once

#include "VoiceChannel.h"
#include <vector>
#include <memory>

class PriorityPlaybackQueue;

// HTTP server that exposes /speak/{channel} endpoints for an agent to POST text.
// One concept: receive text from the network, route it to the right voice.
//
// Endpoints:
//   POST /speak/:channel         — speak complete text: {"text": "Hello"}
//   POST /speak/:channel/stream  — speak streamed SSE text (sentence-split on arrival)
//   GET  /channels               — list configured channels
//   GET  /health                 — returns {"status": "ok"}
//
class InboundServer {
public:
    InboundServer(PriorityPlaybackQueue& queue,
                  const std::vector<VoiceChannel>& channels,
                  int port = 8090);
    ~InboundServer();

    // Start listening (non-blocking, runs in its own thread).
    void start();

    // Stop the server and join the thread.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
