#include "Tokenizer.h"
#include <stdexcept>
#include <cstdio>

Tokenizer::Tokenizer() {
    // Kokoro phoneme vocabulary from config.json
    vocab = {
        {";", 1}, {":", 2}, {",", 3}, {".", 4}, {"!", 5}, {"?", 6},
        {"\u2014", 9},   // —
        {"\u2026", 10},  // …
        {"\"", 11}, {"(", 12}, {")", 13},
        {"\u201C", 14},  // "
        {"\u201D", 15},  // "
        {" ", 16},
        {"\u0303", 17},  // combining tilde
        {"\u02A3", 18},  // ʣ
        {"\u02A5", 19},  // ʥ
        {"\u02A6", 20},  // ʦ
        {"\u02A8", 21},  // ʨ
        {"\u1D5D", 22},  // ᵝ
        {"\uAB67", 23},
        {"A", 24}, {"I", 25}, {"O", 31}, {"Q", 33}, {"S", 35}, {"T", 36},
        {"W", 39}, {"Y", 41},
        {"\u1D4A", 42},  // ᵊ
        {"a", 43}, {"b", 44}, {"c", 45}, {"d", 46}, {"e", 47}, {"f", 48},
        {"h", 50}, {"i", 51}, {"j", 52}, {"k", 53}, {"l", 54}, {"m", 55},
        {"n", 56}, {"o", 57}, {"p", 58}, {"q", 59}, {"r", 60}, {"s", 61},
        {"t", 62}, {"u", 63}, {"v", 64}, {"w", 65}, {"x", 66}, {"y", 67},
        {"z", 68},
        {"\u0251", 69},  // ɑ
        {"\u0250", 70},  // ɐ
        {"\u0252", 71},  // ɒ
        {"\u00E6", 72},  // æ
        {"\u03B2", 75},  // β
        {"\u0254", 76},  // ɔ
        {"\u0255", 77},  // ɕ
        {"\u00E7", 78},  // ç
        {"\u0256", 80},  // ɖ
        {"\u00F0", 81},  // ð
        {"\u02A4", 82},  // ʤ
        {"\u0259", 83},  // ə
        {"\u025A", 85},  // ɚ
        {"\u025B", 86},  // ɛ
        {"\u025C", 87},  // ɜ
        {"\u025F", 90},  // ɟ
        {"\u0261", 92},  // ɡ
        {"\u0265", 99},  // ɥ
        {"\u0268", 101}, // ɨ
        {"\u026A", 102}, // ɪ
        {"\u029D", 103}, // ʝ
        {"\u026F", 110}, // ɯ
        {"\u0270", 111}, // ɰ
        {"\u014B", 112}, // ŋ
        {"\u0273", 113}, // ɳ
        {"\u0272", 114}, // ɲ
        {"\u0274", 115}, // ɴ
        {"\u00F8", 116}, // ø
        {"\u0278", 118}, // ɸ
        {"\u03B8", 119}, // θ
        {"\u0153", 120}, // œ
        {"\u0279", 123}, // ɹ
        {"\u027E", 125}, // ɾ
        {"\u027B", 126}, // ɻ
        {"\u0281", 128}, // ʁ
        {"\u027D", 129}, // ɽ
        {"\u0282", 130}, // ʂ
        {"\u0283", 131}, // ʃ
        {"\u0288", 132}, // ʈ
        {"\u02A7", 133}, // ʧ
        {"\u028A", 135}, // ʊ
        {"\u028B", 136}, // ʋ
        {"\u028C", 138}, // ʌ
        {"\u0263", 139}, // ɣ
        {"\u0264", 140}, // ɤ
        {"\u03C7", 142}, // χ
        {"\u028E", 143}, // ʎ
        {"\u0292", 147}, // ʒ
        {"\u0294", 148}, // ʔ
        {"\u02C8", 156}, // ˈ (primary stress)
        {"\u02CC", 157}, // ˌ (secondary stress)
        {"\u02D0", 158}, // ː (length)
        {"\u02B0", 162}, // ʰ (aspiration)
        {"\u02B2", 164}, // ʲ (palatalization)
        {"\u2193", 169}, // ↓
        {"\u2192", 171}, // →
        {"\u2197", 172}, // ↗
        {"\u2198", 173}, // ↘
        {"\u1D7B", 177}, // ᵻ
    };
}

std::string Tokenizer::text_to_phonemes(const std::string& text, const std::string& lang) {
    // Shell out to espeak-ng for reliable IPA conversion
    // Escape single quotes in text for shell safety
    std::string escaped;
    for (char c : text) {
        if (c == '\'') escaped += "'\\''";
        else escaped += c;
    }
    std::string cmd = "espeak-ng -v " + lang + " -q --ipa '" + escaped + "' 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to run espeak-ng");
    }

    std::string result;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) {
        result += buf;
    }
    pclose(pipe);

    // Remove trailing newlines
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

std::string Tokenizer::next_utf8_char(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return "";
    unsigned char c = s[pos];
    size_t len = 1;
    if ((c & 0x80) == 0) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;

    std::string ch = s.substr(pos, len);
    pos += len;
    return ch;
}

std::vector<int64_t> Tokenizer::phonemes_to_ids(const std::string& phonemes) {
    std::vector<int64_t> ids;
    size_t pos = 0;
    while (pos < phonemes.size()) {
        std::string ch = next_utf8_char(phonemes, pos);
        auto it = vocab.find(ch);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        }
        // Skip unknown characters (newlines, etc.)
    }
    return ids;
}

std::vector<int64_t> Tokenizer::tokenize(const std::string& text, const std::string& lang) {
    std::string phonemes = text_to_phonemes(text, lang);
    auto ids = phonemes_to_ids(phonemes);
    if (ids.size() > 510) {
        ids.resize(510); // max 510 tokens (+ 2 padding = 512)
    }
    return ids;
}
