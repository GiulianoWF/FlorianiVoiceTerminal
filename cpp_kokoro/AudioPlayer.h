#ifndef AUDIO_PLAYER_H
#define AUDIO_PLAYER_H

#include <vector>
#include <atomic>
#include <portaudio.h>
#include <stdexcept>
#include <cstdlib>

class AudioPlayer {
public:
    static void play(const float* audioData, size_t size, int sampleRate,
                     std::atomic<bool>* interrupt = nullptr) {
        PaError err;
        PaStream* stream;
        PaStreamParameters outputParams;

        outputParams.device = Pa_GetDefaultOutputDevice();
        outputParams.channelCount = 1;
        outputParams.sampleFormat = paFloat32;
        outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
        outputParams.hostApiSpecificStreamInfo = nullptr;

        err = Pa_OpenStream(&stream, nullptr, &outputParams, sampleRate, 256, paNoFlag, nullptr, nullptr);
        if (err != paNoError) {
            throw std::runtime_error("Failed to open audio stream: " + std::string(Pa_GetErrorText(err)));
        }

        err = Pa_StartStream(stream);
        if (err != paNoError) {
            Pa_CloseStream(stream);
            throw std::runtime_error("Failed to start audio stream: " + std::string(Pa_GetErrorText(err)));
        }

        // Write audio data in chunks
        size_t framesPerBuffer = 256;
        size_t offset = 0;
        while (offset < size) {
            if (interrupt && interrupt->load()) break;
            size_t framesToWrite = std::min(framesPerBuffer, size - offset);
            err = Pa_WriteStream(stream, audioData + offset, framesToWrite);
            if (err != paNoError && err != paOutputUnderflowed) {
                break;
            }
            offset += framesToWrite;
        }

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
    }
};

#endif // AUDIO_PLAYER_H
