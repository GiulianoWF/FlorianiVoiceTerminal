#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class KokoroInference {
public:
    KokoroInference(const std::string& model_path, const std::string& voice_path);

    // Switch to a different voice at runtime (reloads the .bin file, keeps the ONNX model).
    void set_voice(const std::string& voice_path);

    // Synthesize audio from pre-tokenized phoneme IDs
    // Token IDs come from misaki phoneme conversion + phoneme-to-ID mapping
    std::vector<float> synthesize(const std::vector<int64_t>& token_ids, float speed = 1.0f);

private:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "KokoroTTS"};
    Ort::Session session{nullptr};

    // Voice style data loaded from .bin file: shape (num_styles, 1, 256)
    std::vector<float> voice_data;
    size_t voice_num_styles = 0;

    std::vector<std::string> get_names(bool inputs);
};
