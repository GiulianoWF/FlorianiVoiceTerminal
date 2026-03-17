#pragma once

#include "whisper_stream.h"
#include <portaudio.h>

// PortAudio-based AudioSource for WhisperStream.
// Assumes Pa_Initialize() has already been called externally.

class PortAudioSource : public AudioSource {
public:
    explicit PortAudioSource(int buffer_samples)
        : m_ring(buffer_samples) {}

    ~PortAudioSource() override { stop(); }

    bool start() override {
        PaError err = Pa_OpenDefaultStream(
            &m_stream, 1, 0, paFloat32,
            16000, 1024,
            pa_callback, this);

        if (err != paNoError) return false;

        err = Pa_StartStream(m_stream);
        return err == paNoError;
    }

    void stop() override {
        if (m_stream) {
            Pa_StopStream(m_stream);
            Pa_CloseStream(m_stream);
            m_stream = nullptr;
        }
    }

    int get_audio(int n, std::vector<float>& out) override {
        m_ring.get_last(n, out);
        return (int)out.size();
    }

    void clear() override { m_ring.clear(); }
    int  available() override { return m_ring.size(); }

private:
    static int pa_callback(const void* input, void* /*output*/,
                           unsigned long frameCount,
                           const PaStreamCallbackTimeInfo* /*timeInfo*/,
                           PaStreamCallbackFlags /*flags*/,
                           void* userData) {
        auto* self = static_cast<PortAudioSource*>(userData);
        self->m_ring.push(static_cast<const float*>(input),
                          static_cast<int>(frameCount));
        return paContinue;
    }

    AudioRingBuffer m_ring;
    PaStream*       m_stream = nullptr;
};
