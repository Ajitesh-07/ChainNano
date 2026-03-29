#include "InferenceQueue.h"
#include <chrono>
#include <algorithm>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/stack.h>
#include <torch/utils.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/headeronly/core/DeviceType.h>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <ATen/ops/cat.h>
#include <cstdint>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <cstdlib>
#include <iostream>
#include <torch/types.h>

InferenceQueue::InferenceQueue(torch::jit::script::Module& neural_net, int batch_size, int timeout_microsecs)
    : model(neural_net), max_batch_size(batch_size), timeout_us(timeout_microsecs), running(true)
{
    // Boot up the background GPU thread immediately
    gpu_thread = std::thread(&InferenceQueue::gpu_loop, this);
}

InferenceQueue::~InferenceQueue() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        running = false;
    }
    cv.notify_all(); // Wake up the thread if it's sleeping so it can exit
    if (gpu_thread.joinable()) {
        gpu_thread.join();
    }
}

// =========================================================================
// THE AWAIT (Frontend)
// =========================================================================
InferenceResult InferenceQueue::infer(torch::Tensor board, int tId, int gId) {
    InferenceRequest req;
    req.boardTensor = board;
    req.threadId = tId;
    req.gameId = gId;

    auto future = req.promise.get_future();

    // Lock queue, insert pointer to our local request
    {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_back(&req);
    }

    cv.notify_one(); // "Hey GPU, I dropped a board off!"

    // FREEZE THIS CPU THREAD: Wait until the GPU fills the promise
    return future.get();
}

// Drop this at the bottom of InferenceQueue.cpp
void InferenceQueue::thread_finished() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (max_batch_size > 0) {
            max_batch_size--;
        }
    }
    // Wake up the GPU thread in case it was waiting for a ghost thread
    cv.notify_all();
}

// =========================================================================
// THE RESOLVER (Backend)
// =========================================================================
void InferenceQueue::gpu_loop() {
    while (true) {
        std::vector<InferenceRequest*> current_batch;

        {
            std::unique_lock<std::mutex> lock(mtx);

            // 1. Sleep until at least ONE board arrives or we shut down
            cv.wait(lock, [this]() { return !queue.empty() || !running; });
            if (!running && queue.empty()) break;

            // 2. THE TIMEOUT: Wait up to 150µs for more boards to arrive
            cv.wait_for(lock, std::chrono::microseconds(timeout_us), [this]() {
                return queue.size() >= max_batch_size || !running;
                });
            if (!running && queue.empty()) break;

            // 3. Scoop up the boards (up to max_batch_size)
            int take = std::min((int)queue.size(), max_batch_size);
            current_batch.insert(current_batch.end(), queue.begin(), queue.begin() + take);
            queue.erase(queue.begin(), queue.begin() + take);
        } // Mutex unlocks here!

        if (current_batch.empty()) continue;

        std::vector<torch::Tensor> board_tensors;
        std::vector<int64_t> split_sizes; // Remembers how many boards each thread sent

        for (auto* req : current_batch) {
            board_tensors.push_back(req->boardTensor);
            split_sizes.push_back(req->boardTensor.size(0)); // e.g., pushes 16
        }

        // Use cat() to flatten [16, 2, 6, 7] chunks into a single [256, 2, 6, 7] tensor
        torch::Tensor batch_input = torch::cat(board_tensors, 0).to(torch::kCUDA).to(torch::kBFloat16);

        // 5. Run the neural network ONCE for the mega-batch
        torch::Tensor policies, values;
        try {
            torch::NoGradGuard no_grad;
            auto output = model.forward({ batch_input }).toTuple();

            policies = output->elements()[0].toTensor().to(torch::kFloat32).cpu();
            values = output->elements()[1].toTensor().to(torch::kFloat32).cpu();
        }
        catch (const c10::Error& e) {
            std::cerr << "\n[PYTORCH FATAL ERROR] " << e.msg() << "\n";
            std::cerr << "Check your C++ tensor dimensions! The model expects [Batch, 119, 8, 8]\n";
            exit(1);
        }

        // 6. Split the 256 results back into the original chunks
        // This splits the [256, 7] tensor back into a vector of [16, 7] tensors
        auto policy_chunks = policies.split_with_sizes(split_sizes, 0);
        auto value_chunks = values.split_with_sizes(split_sizes, 0);

        // 7. Hand the correct chunk back to each frozen CPU thread
        for (size_t i = 0; i < current_batch.size(); ++i) {
            InferenceResult result;
            result.policies = policy_chunks[i];
            result.values = value_chunks[i];

            // This line instantly wakes up the specific CPU thread!
            current_batch[i]->promise.set_value(std::move(result));
        }
    }
}