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
    bool playing = false;  // protected by mutex (no longer atomic — fixes TOCTOU in is_active)

    std::string current_channel_;       // channel of currently playing item (mutex-protected)

    // Pending response state: worker holds here after a wait_response item completes
    bool pending_response_ = false;
    std::string pending_callback_url_;
    std::string pending_channel_;

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

                // Set all state inside the lock to prevent race with interrupt_channel
                playing = true;
                current_priority = request.priority;
                current_channel_ = request.channel;
                playback_interrupt = false;
            }

            std::cerr << "[Queue] Playing channel=\"" << request.channel
                      << "\" priority=" << static_cast<int>(request.priority)
                      << " text=\"" << request.text.substr(0, 50) << "\"" << std::endl;

            auto audio = engine.synthesize(request.text,
                                            request.voice_path,
                                            request.language,
                                            &playback_interrupt);

            // Play (unless interrupted during synthesis)
            if (!audio.empty() && !playback_interrupt) {
                play_fn(audio.data(), audio.size(), 24000, &playback_interrupt);
            }

            bool completed = !playback_interrupt.load();

            {
                std::lock_guard<std::mutex> lock(mutex);
                playing = false;
                current_priority = Priority::LOW;
                current_channel_.clear();
            }

            // Hold the worker if this item needs a user response and wasn't interrupted
            if (completed && request.wait_response && !request.callback_url.empty()) {
                std::cerr << "[Queue] Waiting for user response (callback="
                          << request.callback_url << ")" << std::endl;
                std::unique_lock<std::mutex> lock(mutex);
                pending_response_ = true;
                pending_callback_url_ = request.callback_url;
                pending_channel_ = request.channel;
                cv.wait(lock, [this] { return !pending_response_ || !running; });
            }
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
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->running = false;
        impl_->pending_response_ = false;  // Release hold if waiting
    }
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

void PriorityPlaybackQueue::interrupt_channel(const std::string& channel) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Interrupt current playback if it belongs to this channel
    if (impl_->playing && impl_->current_channel_ == channel) {
        impl_->playback_interrupt = true;
    }

    // Cancel pending response if from this channel
    if (impl_->pending_response_ && impl_->pending_channel_ == channel) {
        impl_->pending_response_ = false;
        impl_->pending_callback_url_.clear();
        impl_->pending_channel_.clear();
        impl_->cv.notify_one();  // Release worker hold
    }

    // Rebuild queue without this channel's items
    std::priority_queue<SpeechRequest> filtered;
    while (!impl_->queue.empty()) {
        SpeechRequest item = impl_->queue.top();
        impl_->queue.pop();
        if (item.channel != channel) {
            filtered.push(std::move(item));
        }
    }
    std::swap(impl_->queue, filtered);
}

std::string PriorityPlaybackQueue::get_current_channel() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->current_channel_;
}

bool PriorityPlaybackQueue::is_active() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->playing || !impl_->queue.empty() || impl_->pending_response_;
}

bool PriorityPlaybackQueue::has_pending_response() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->pending_response_;
}

std::string PriorityPlaybackQueue::get_pending_callback_url() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->pending_callback_url_;
}

void PriorityPlaybackQueue::complete_response() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->pending_response_ = false;
    impl_->pending_callback_url_.clear();
    impl_->pending_channel_.clear();
    impl_->cv.notify_one();  // Wake the worker
}
