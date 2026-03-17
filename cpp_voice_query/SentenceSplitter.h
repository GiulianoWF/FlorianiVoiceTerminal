#pragma once

#include <string>
#include <functional>

// Accumulates streaming text and emits complete sentences at .!? boundaries.
// One concept: turn a stream of text fragments into a stream of sentences.
//
// Usage:
//   SentenceSplitter splitter([](const std::string& s) { speak(s); });
//   splitter.feed("Hello world. ");   // calls callback with "Hello world."
//   splitter.feed("How are ");
//   splitter.feed("you? Fine. ");     // calls callback with "How are you?" then "Fine."
//   std::string leftover = splitter.flush();  // returns "" (nothing left)
//
class SentenceSplitter {
public:
    using OnSentence = std::function<void(const std::string&)>;

    explicit SentenceSplitter(OnSentence callback) : on_sentence_(std::move(callback)) {}

    // Feed text fragments as they arrive from a stream.
    // Calls the callback for each complete sentence found.
    void feed(const std::string& text) {
        buffer_ += text;

        size_t pos = 0;
        while (pos < buffer_.size()) {
            size_t end = buffer_.find_first_of(".!?", pos);
            if (end == std::string::npos) break;

            // A sentence ends at punctuation followed by a space
            if (end + 1 < buffer_.size() && buffer_[end + 1] == ' ') {
                std::string sentence = buffer_.substr(0, end + 1);
                if (!sentence.empty() && on_sentence_) {
                    on_sentence_(sentence);
                }
                buffer_ = buffer_.substr(end + 2);
                pos = 0;
            } else {
                pos = end + 1;
            }
        }
    }

    // Flush any remaining partial sentence (call at end of stream).
    // Returns the flushed text, or empty if nothing remained.
    std::string flush() {
        std::string remaining = std::move(buffer_);
        buffer_.clear();
        return remaining;
    }

private:
    std::string buffer_;
    OnSentence on_sentence_;
};
