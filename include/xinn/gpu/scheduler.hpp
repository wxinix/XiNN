// SPDX-License-Identifier: BSD-3-Clause
// A std::execution scheduler for the GPU.
//
//   gpu::Queue q(device);                          // one thread; it submits the GPU work
//   auto work = schedule(q.scheduler())
//             | then([&] { return gpu::eval(device, sigmoid(x * w + b)); });
//   auto [y] = sync_wait(std::move(work)).value();
//
// A Queue is a worker thread with a list of tasks. schedule() gives a sender
// that, when started, puts its operation at the end of the list; the worker
// thread takes it off and completes it, so everything chained after it
// (then, let_value, ...) runs on that thread, in the order it was
// scheduled. The worker is the one thread that talks to the device's
// command queue: the program's other threads stay free, and GPU work is
// submitted in order.
//
// This is the smallest scheduler the concept allows: schedule() and
// equality. It does not customize bulk (bulk runs its iterations one after
// another on the worker thread, not as a kernel), and it does not support
// stop requests. The device is optional, so the scheduler works, and can be
// tested, on a machine without OpenCL.
#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "xinn/gpu/device.hpp"
#include "xinn/parallel.hpp"

#if XINN_PARALLEL

namespace xinn::gpu {

namespace ex = xinn::detail::ex;

class Queue {
    // Operations waiting to run: an intrusive list, so scheduling allocates nothing.
    struct Task {
        Task* next = nullptr;
        void (*run)(Task*) noexcept = nullptr;
    };

public:
    class Scheduler;

    explicit Queue(std::optional<Device> device = std::nullopt)
        : device_(std::move(device)), thread_([this] { loop(); }) {}

    // Runs what is still queued, then stops the thread.
    ~Queue() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        ready_.notify_one();
        thread_.join();
    }

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    Scheduler scheduler() noexcept;
    const Device* device() const noexcept { return device_ ? &*device_ : nullptr; }
    std::thread::id thread_id() const noexcept { return thread_.get_id(); }

private:
    void push(Task* t) {
        {
            std::lock_guard lock(mutex_);
            (tail_ ? tail_->next : head_) = t;
            tail_ = t;
        }
        ready_.notify_one();
    }

    void loop() {
        for (;;) {
            Task* t;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return head_ || stop_; });
                if (!head_) return;   // stopped, and nothing left
                t = head_;
                head_ = t->next;
                if (!head_) tail_ = nullptr;
            }
            t->run(t);   // completes the receiver: the continuation runs here
        }
    }

    std::optional<Device> device_;
    std::mutex mutex_;
    std::condition_variable ready_;
    Task* head_ = nullptr;
    Task* tail_ = nullptr;
    bool stop_ = false;
    std::thread thread_;   // last: starts after everything above is ready
};

class Queue::Scheduler {
public:
    using scheduler_concept = ex::scheduler_t;

private:
    // The operation state: connect(sender, receiver) makes one, start() queues it.
    template <class Receiver>
    struct Operation : Task {
        using operation_state_concept = ex::operation_state_t;

        Operation(Queue* q, Receiver r) : queue(q), receiver(std::move(r)) {
            run = [](Task* t) noexcept { ex::set_value(std::move(static_cast<Operation*>(t)->receiver)); };
        }
        Operation(Operation&&) = delete;   // the list points at it: it must not move

        void start() & noexcept { queue->push(this); }

        Queue* queue;
        Receiver receiver;
    };

    // Where the sender completes, for algorithms that ask (continues_on, ...).
    struct Env {
        Queue* queue;
        Scheduler query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept { return Scheduler(queue); }
    };

    struct Sender {
        using sender_concept = ex::sender_t;
        using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

        template <class Receiver>
        Operation<Receiver> connect(Receiver r) const {
            return Operation<Receiver>(queue, std::move(r));
        }
        Env get_env() const noexcept { return {queue}; }

        Queue* queue;
    };

public:
    Sender schedule() const noexcept { return {queue_}; }

    // The device the queue's thread drives, or null.
    const Device* device() const noexcept { return queue_->device(); }

    bool operator==(const Scheduler&) const noexcept = default;

private:
    friend class Queue;
    explicit Scheduler(Queue* q) noexcept : queue_(q) {}
    Queue* queue_;
};

inline Queue::Scheduler Queue::scheduler() noexcept { return Scheduler(this); }

}  // namespace xinn::gpu

#endif  // XINN_PARALLEL
