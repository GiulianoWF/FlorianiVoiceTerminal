#include "PriorityPlaybackQueue.h"
#include "TtsEngine.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <iostream>

struct PriorityPlaybackQueue::Impl {
    TtsEngine& engine;
    PlayFn play_fn;

    std::priority_queue<SpeechRequest> queue;
    std::mutex mutex;
    std::condition_variable cv;

    std::atomic<bool> running{false};
    std::atomic<bool> playback_interrupt{false};
    std::atomic<Priority> current_priority{Priority::LOW};
    std::atomic<bool> playing{false};

    uint64_t next_sequence = 0;
    std::thread worker;

    Impl(TtsEngine& e, PlayFn fn) : engine(e), play_fn(std::move(fn)) {}

    void run() {
        while (running) {
            SpeechRequest request;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return !queue.empty() || !running; });
                if (!running) break;
                if (queue.empty()) continue;
                request = queue.top();
                queue.pop();
            }

            // Synthesize
            playing = true;
            current_priority = request.priority;
            playback_interrupt = false;

            std::cerr << "[Queue] Playing priority=" << static_cast<int>(request.priority)
                      << " text=\"" << request.text.substr(0, 50) << "\"" << std::endl;

            auto audio = engine.synthesize(request.text,
                                            request.voice_path,
                                            request.language,
                                            &playback_interrupt);

            // Play (unless interrupted during synthesis)
            if (!audio.empty() && !playback_interrupt) {
                play_fn(audio.data(), audio.size(), 24000, &playback_interrupt);
            }

            playing = false;
            current_priority = Priority::LOW;
        }
    }
};

PriorityPlaybackQueue::PriorityPlaybackQueue(TtsEngine& engine, PlayFn play_fn)
    : impl_(std::make_unique<Impl>(engine, std::move(play_fn))) {}

PriorityPlaybackQueue::~PriorityPlaybackQueue() {
    stop();
}

void PriorityPlaybackQueue::submit(SpeechRequest request) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    request.sequence = impl_->next_sequence++;
    Priority incoming = request.priority;
    impl_->queue.push(std::move(request));

    // Interrupt current playback if incoming priority is higher
    if (impl_->playing && incoming > impl_->current_priority.load()) {
        impl_->playback_interrupt = true;
        std::cerr << "[Queue] Interrupting: priority " << static_cast<int>(incoming)
                  << " > " << static_cast<int>(impl_->current_priority.load()) << std::endl;
    }

    impl_->cv.notify_one();
}

void PriorityPlaybackQueue::start() {
    impl_->running = true;
    impl_->worker = std::thread([this] { impl_->run(); });
}

void PriorityPlaybackQueue::stop() {
    impl_->running = false;
    impl_->playback_interrupt = true;
    impl_->cv.notify_one();
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
    // Drain remaining items
    std::lock_guard<std::mutex> lock(impl_->mutex);
    while (!impl_->queue.empty()) impl_->queue.pop();
}

void PriorityPlaybackQueue::interrupt_current() {
    impl_->playback_interrupt = true;
}

bool PriorityPlaybackQueue::is_active() const {
    return impl_->playing || [this] {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return !impl_->queue.empty();
    }();
}
