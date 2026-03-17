#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Tokenizer {
public:
    Tokenizer();

    // Convert text to phoneme token IDs using espeak-ng
    std::vector<int64_t> tokenize(const std::string& text, const std::string& lang = "en-us");

private:
    std::unordered_map<std::string, int64_t> vocab;

    // Convert text to IPA phonemes via espeak-ng
    std::string text_to_phonemes(const std::string& text, const std::string& lang);

    // Convert IPA string to token IDs using the vocabulary
    std::vector<int64_t> phonemes_to_ids(const std::string& phonemes);

    // Decode one UTF-8 character from a string at position pos, advance pos
    static std::string next_utf8_char(const std::string& s, size_t& pos);
};
