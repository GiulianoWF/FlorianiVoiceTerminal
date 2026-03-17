#include "KokoroInference.h"
#include <fstream>
#include <stdexcept>

void KokoroInference::set_voice(const std::string& voice_path) {
    std::ifstream voice_file(voice_path, std::ios::binary | std::ios::ate);
    if (!voice_file.is_open()) {
        throw std::runtime_error("Failed to open voice file: " + voice_path);
    }
    size_t file_size = voice_file.tellg();
    voice_file.seekg(0, std::ios::beg);
    voice_data.resize(file_size / sizeof(float));
    voice_file.read(reinterpret_cast<char*>(voice_data.data()), file_size);
    voice_num_styles = file_size / (256 * sizeof(float));
}

KokoroInference::KokoroInference(const std::string& model_path, const std::string& voice_path) {
    set_voice(voice_path);

    // Create session
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    try {
        session = Ort::Session(env, model_path.c_str(), session_options);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

std::vector<float> KokoroInference::synthesize(const std::vector<int64_t>& token_ids, float speed) {
    // Pad tokens: [0, ...token_ids..., 0]
    std::vector<int64_t> padded = {0};
    padded.insert(padded.end(), token_ids.begin(), token_ids.end());
    padded.push_back(0);

    // Select voice style based on token count (indexed by original token length)
    size_t style_idx = token_ids.size();
    if (style_idx >= voice_num_styles) {
        style_idx = voice_num_styles - 1;
    }
    // Style vector is at offset style_idx * 256, shape [1, 256]
    std::vector<float> style_vec(voice_data.begin() + style_idx * 256,
                                  voice_data.begin() + (style_idx + 1) * 256);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    // input_ids: [1, sequence_length]
    std::vector<int64_t> ids_shape = {1, static_cast<int64_t>(padded.size())};
    Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, padded.data(), padded.size(), ids_shape.data(), ids_shape.size());

    // style: [1, 256]
    std::vector<int64_t> style_shape = {1, 256};
    Ort::Value style_tensor = Ort::Value::CreateTensor<float>(
        mem_info, style_vec.data(), style_vec.size(), style_shape.data(), style_shape.size());

    // speed: [1]
    std::vector<float> speed_val = {speed};
    std::vector<int64_t> speed_shape = {1};
    Ort::Value speed_tensor = Ort::Value::CreateTensor<float>(
        mem_info, speed_val.data(), speed_val.size(), speed_shape.data(), speed_shape.size());

    // Run inference
    const char* input_names[] = {"input_ids", "style", "speed"};
    const char* output_names[] = {"waveform"};
    Ort::Value input_tensors[] = {std::move(ids_tensor), std::move(style_tensor), std::move(speed_tensor)};

    auto output_values = session.Run(Ort::RunOptions{nullptr},
        input_names, input_tensors, 3,
        output_names, 1);

    // Extract audio from waveform output
    float* audio_data = output_values[0].GetTensorMutableData<float>();
    size_t audio_size = output_values[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<float>(audio_data, audio_data + audio_size);
}

std::vector<std::string> KokoroInference::get_names(bool inputs) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    for (size_t i = 0; i < count; i++) {
        auto name = inputs ? session.GetInputNameAllocated(i, allocator)
                           : session.GetOutputNameAllocated(i, allocator);
        names.push_back(name.get());
    }
    return names;
}
