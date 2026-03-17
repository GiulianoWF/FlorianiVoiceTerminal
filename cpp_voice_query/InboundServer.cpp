#include "InboundServer.h"
#include "PriorityPlaybackQueue.h"
#include "SentenceSplitter.h"
#include <httplib/httplib.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <unordered_map>
#include <iostream>
#include <sstream>

using json = nlohmann::json;

struct InboundServer::Impl {
    httplib::Server server;
    PriorityPlaybackQueue& queue;
    std::unordered_map<std::string, VoiceChannel> channels;
    int port;
    std::thread thread;

    Impl(PriorityPlaybackQueue& q, const std::vector<VoiceChannel>& ch, int p)
        : queue(q), port(p) {
        for (const auto& c : ch) {
            channels[c.name] = c;
        }
        setup_routes();
    }

    void setup_routes() {
        // POST /speak/:channel — speak complete text
        server.Post("/speak/:channel", [this](const httplib::Request& req, httplib::Response& res) {
            auto channel_name = req.path_params.at("channel");
            auto it = channels.find(channel_name);
            if (it == channels.end()) {
                res.status = 404;
                res.set_content(R"({"error": "unknown channel"})", "application/json");
                return;
            }
            const auto& channel = it->second;

            json body;
            try {
                body = json::parse(req.body);
            } catch (const json::parse_error&) {
                res.status = 400;
                res.set_content(R"({"error": "invalid JSON"})", "application/json");
                return;
            }

            if (!body.contains("text") || !body["text"].is_string()) {
                res.status = 400;
                res.set_content(R"({"error": "missing 'text' field"})", "application/json");
                return;
            }

            std::string text = body["text"].get<std::string>();
            if (text.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "empty text"})", "application/json");
                return;
            }

            // Check if this is a streaming request
            bool stream = body.value("stream", false);

            if (stream) {
                handle_stream(channel, text, req, res);
            } else {
                SpeechRequest speech;
                speech.text = std::move(text);
                speech.priority = channel.priority;
                speech.voice_path = channel.voice_path;
                speech.language = channel.language;
                queue.submit(std::move(speech));

                res.status = 202;
                res.set_content(R"({"status": "queued"})", "application/json");
            }

            std::cerr << "[Server] /speak/" << channel_name
                      << " | priority=" << static_cast<int>(channel.priority)
                      << " | stream=" << (stream ? "true" : "false") << std::endl;
        });

        // POST /speak/:channel/stream — SSE streaming endpoint
        server.Post("/speak/:channel/stream", [this](const httplib::Request& req, httplib::Response& res) {
            auto channel_name = req.path_params.at("channel");
            auto it = channels.find(channel_name);
            if (it == channels.end()) {
                res.status = 404;
                res.set_content(R"({"error": "unknown channel"})", "application/json");
                return;
            }
            const auto& channel = it->second;
            handle_stream(channel, req.body, req, res);

            res.status = 202;
            res.set_content(R"({"status": "streamed"})", "application/json");

            std::cerr << "[Server] /speak/" << channel_name << "/stream" << std::endl;
        });

        // GET /channels — list configured channels
        server.Get("/channels", [this](const httplib::Request&, httplib::Response& res) {
            json arr = json::array();
            for (const auto& [name, ch] : channels) {
                arr.push_back({
                    {"name", ch.name},
                    {"voice", ch.voice_path},
                    {"priority", static_cast<int>(ch.priority)},
                    {"language", ch.language}
                });
            }
            res.set_content(arr.dump(), "application/json");
        });

        // GET /health
        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"status": "ok"})", "application/json");
        });
    }

    // Parse SSE body and submit one SpeechRequest per sentence.
    // Accepts the raw body which may contain OpenAI-format SSE lines.
    void handle_stream(const VoiceChannel& channel,
                       const std::string& body,
                       const httplib::Request&,
                       httplib::Response&) {
        SentenceSplitter splitter([this, &channel](const std::string& sentence) {
            SpeechRequest speech;
            speech.text = sentence;
            speech.priority = channel.priority;
            speech.voice_path = channel.voice_path;
            speech.language = channel.language;
            queue.submit(std::move(speech));
        });

        // Parse SSE lines from the body
        std::istringstream stream(body);
        std::string line;
        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.rfind("data: ", 0) != 0) continue;
            std::string data = line.substr(6);
            if (data == "[DONE]") break;

            try {
                auto j = json::parse(data);
                if (j.contains("choices") && !j["choices"].empty()) {
                    auto& delta = j["choices"][0]["delta"];
                    if (delta.contains("content") && delta["content"].is_string()) {
                        splitter.feed(delta["content"].get<std::string>());
                    }
                }
            } catch (const json::parse_error&) {
                // Skip malformed chunks
            }
        }

        // Flush remaining text as a final sentence
        std::string remaining = splitter.flush();
        if (!remaining.empty()) {
            SpeechRequest speech;
            speech.text = std::move(remaining);
            speech.priority = channel.priority;
            speech.voice_path = channel.voice_path;
            speech.language = channel.language;
            queue.submit(std::move(speech));
        }
    }
};

InboundServer::InboundServer(PriorityPlaybackQueue& queue,
                             const std::vector<VoiceChannel>& channels,
                             int port)
    : impl_(std::make_unique<Impl>(queue, channels, port)) {}

InboundServer::~InboundServer() {
    stop();
}

void InboundServer::start() {
    impl_->thread = std::thread([this] {
        std::cerr << "[Server] Listening on port " << impl_->port << std::endl;
        impl_->server.listen("0.0.0.0", impl_->port);
    });
}

void InboundServer::stop() {
    impl_->server.stop();
    if (impl_->thread.joinable()) {
        impl_->thread.join();
    }
}
