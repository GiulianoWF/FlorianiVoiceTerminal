#pragma once

#include <string>
#include <vector>
#include <functional>
#include <atomic>

struct Message {
    std::string role;
    std::string content;
};

class LlmClient {
public:
    explicit LlmClient(const std::string& base_url);

    bool health_check();

    // Stream chat response. on_sentence called per sentence for real-time TTS.
    // If interrupt is non-null and becomes true, stops streaming and returns early.
    std::string chat(const std::string& user_msg,
                     std::vector<Message>& history,
                     std::function<void(const std::string&)> on_sentence = nullptr,
                     std::atomic<bool>* interrupt = nullptr);

private:
    std::string base_url_;

    static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata);
};
