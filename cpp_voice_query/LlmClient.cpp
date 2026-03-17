#include "LlmClient.h"
#include "SentenceSplitter.h"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <iostream>
#include <sstream>

using json = nlohmann::json;

// State passed through libcurl's write callback
struct StreamState {
    std::string full_reply;    // complete reply for history
    std::string line_buffer;   // partial SSE line
    SentenceSplitter* splitter = nullptr;
    std::atomic<bool>* interrupt = nullptr;
};

LlmClient::LlmClient(const std::string& base_url) : base_url_(base_url) {}

bool LlmClient::health_check() {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string url = base_url_ + "/health";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return res == CURLE_OK;
}

size_t LlmClient::write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb;
    auto* state = static_cast<StreamState*>(userdata);

    // Abort the transfer if interrupted (returning 0 makes curl stop)
    if (state->interrupt && state->interrupt->load()) {
        return 0;
    }

    state->line_buffer.append(ptr, total);

    std::istringstream stream(state->line_buffer);
    std::string line;
    std::string remaining;
    bool has_remaining = false;

    while (std::getline(stream, line)) {
        if (stream.eof() && state->line_buffer.back() != '\n') {
            remaining = line;
            has_remaining = true;
            break;
        }

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.rfind("data: ", 0) != 0) continue;

        std::string data = line.substr(6);
        if (data == "[DONE]") continue;

        try {
            auto j = json::parse(data);
            if (j.contains("choices") && !j["choices"].empty()) {
                auto& delta = j["choices"][0]["delta"];
                if (delta.contains("content") && delta["content"].is_string()) {
                    std::string content = delta["content"].get<std::string>();
                    state->full_reply += content;
                    if (state->splitter) {
                        state->splitter->feed(content);
                    }
                }
            }
        } catch (const json::parse_error&) {
            // Skip malformed chunks
        }
    }

    state->line_buffer = has_remaining ? remaining : "";
    return total;
}

std::string LlmClient::chat(const std::string& user_msg,
                              std::vector<Message>& history,
                              std::function<void(const std::string&)> on_sentence,
                              std::atomic<bool>* interrupt) {
    history.push_back({"user", user_msg});

    // Build request payload (last 10 messages)
    json messages = json::array();
    size_t start = history.size() > 10 ? history.size() - 10 : 0;
    for (size_t i = start; i < history.size(); i++) {
        messages.push_back({{"role", history[i].role}, {"content", history[i].content}});
    }

    json payload = {
        {"messages", messages},
        {"stream", true}
    };
    std::string body = payload.dump();

    CURL* curl = curl_easy_init();
    if (!curl) {
        return "[curl init failed]";
    }

    std::string url = base_url_ + "/v1/chat/completions";
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    SentenceSplitter splitter(on_sentence);

    StreamState state;
    state.splitter = on_sentence ? &splitter : nullptr;
    state.interrupt = interrupt;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK && res != CURLE_WRITE_ERROR) {
        // CURLE_WRITE_ERROR is expected when we interrupt via write_callback returning 0
        return std::string("[LLM error: ") + curl_easy_strerror(res) + "]";
    }

    // Speak any remaining partial sentence (unless interrupted)
    bool interrupted = interrupt && interrupt->load();
    if (on_sentence && !interrupted) {
        std::string remaining = splitter.flush();
        if (!remaining.empty()) {
            on_sentence(remaining);
        }
    }

    history.push_back({"assistant", state.full_reply});
    return state.full_reply;
}
