cmake_minimum_required(VERSION 3.14)
project(kokoro_tts)

# ONNX Runtime
set(ONNXRuntime_DIR "${CMAKE_SOURCE_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/lib/cmake/onnxruntime")
find_package(ONNXRuntime REQUIRED)

# Fix the include directory (ONNX Runtime config uses wrong path)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/include"
)

# PortAudio for audio playback
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
include_directories(${PORTAUDIO_INCLUDE_DIRS})

# espeak-ng is called as a subprocess for text-to-phoneme conversion
# Ensure it's installed: sudo apt install espeak-ng

# Add executable
add_executable(kokoro_tts main.cpp KokoroInference.cpp Tokenizer.cpp)

# Note: include directory is set on the onnxruntime::onnxruntime target above

target_link_libraries(kokoro_tts
    onnxruntime::onnxruntime
    ${PORTAUDIO_LIBRARIES}
)

# Embed rpath so the binary finds libonnxruntime at runtime
set_target_properties(kokoro_tts PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/lib"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/lib"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)