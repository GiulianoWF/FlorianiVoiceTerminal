cmake_minimum_required(VERSION 3.14)
project(whisper_stt)

set(CMAKE_CXX_STANDARD 17)

# Build whisper.cpp as a subdirectory
add_subdirectory(whisper.cpp)

# PortAudio for microphone input
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Whisper streaming library (no PortAudio dependency)
add_library(whisper_stream STATIC whisper_stream.cpp)
target_include_directories(whisper_stream PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/whisper.cpp/include
)
target_link_libraries(whisper_stream PUBLIC whisper)

# CLI executable (PortAudio + whisper_stream)
add_executable(whisper_stt main.cpp)
target_include_directories(whisper_stt PRIVATE
    ${PORTAUDIO_INCLUDE_DIRS}
)
target_link_libraries(whisper_stt
    whisper_stream
    ${PORTAUDIO_LIBRARIES}
)
