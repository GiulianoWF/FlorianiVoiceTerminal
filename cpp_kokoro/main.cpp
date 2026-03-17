#include "KokoroInference.h"
#include "Tokenizer.h"
#include "AudioPlayer.h"

#include <iostream>

int main() {
    try {
        // 1. Load model, voice, and tokenizer
        KokoroInference tts("kokoro-v1.0.onnx", "voices/af.bin");
        Tokenizer tokenizer;

        // 2. Convert text to phoneme token IDs
        // Use "en-us" for English, "pt-br" for Brazilian Portuguese, etc.
        std::string text = "Olá, isso é um teste de texto para fala.";
        auto tokens = tokenizer.tokenize(text, "pt-br");
        auto audio = tts.synthesize(tokens, 1.0f);
        AudioPlayer::play(audio.data(), audio.size(), 24000);

        text = "Hello, this is a test of Kokoro text to speech in C++.";
        tokens = tokenizer.tokenize(text, "en-us");

        // std::cout << "Text: " << text << std::endl;
        // std::cout << "Tokens: " << tokens.size() << std::endl;

        // 3. Synthesize audio
        auto audio2 = tts.synthesize(tokens, 1.0f);
        // std::cout << "Audio samples: " << audio.size() << std::endl;

        // 4. Play audio at 24kHz
        AudioPlayer::play(audio2.data(), audio2.size(), 24000);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
