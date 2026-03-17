cmake_minimum_required(VERSION 3.14)
project(voice_query)

set(CMAKE_CXX_STANDARD 17)

# --- whisper.cpp (from sibling project) ---
add_subdirectory(${CMAKE_SOURCE_DIR}/../cpp_whisper/whisper.cpp ${CMAKE_BINARY_DIR}/whisper_build)

# --- ONNX Runtime (from kokoro's download) ---
set(KOKORO_DIR "${CMAKE_SOURCE_DIR}/../cpp_kokoro")
set(ONNXRuntime_DIR "${KOKORO_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/lib/cmake/onnxruntime")
find_package(ONNXRuntime REQUIRED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${KOKORO_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/include"
)

# --- PortAudio ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# --- libcurl ---
find_package(CURL REQUIRED)

# --- whisper_stream sources ---
set(WHISPER_STREAM_DIR "${CMAKE_SOURCE_DIR}/../cpp_whisper")

# --- Build executable ---
add_executable(voice_query
    main.cpp
    LlmClient.cpp
    ${WHISPER_STREAM_DIR}/whisper_stream.cpp
    # Kokoro sources from sibling project
    ${KOKORO_DIR}/KokoroInference.cpp
    ${KOKORO_DIR}/Tokenizer.cpp
)

target_include_directories(voice_query PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${WHISPER_STREAM_DIR}                                  # whisper_stream.h
    ${WHISPER_STREAM_DIR}/whisper.cpp/include               # whisper.h
    ${KOKORO_DIR}
    ${PORTAUDIO_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(voice_query
    whisper
    onnxruntime::onnxruntime
    ${PORTAUDIO_LIBRARIES}
    ${CURL_LIBRARIES}
)

# RPATH for ONNX Runtime
set_target_properties(voice_query PROPERTIES
    BUILD_RPATH "${KOKORO_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/lib"
    INSTALL_RPATH "${KOKORO_DIR}/onnx/onnxruntime-linux-x64-gpu-1.24.2/lib"
)
