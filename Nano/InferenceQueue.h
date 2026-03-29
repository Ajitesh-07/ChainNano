#pragma once

#include <torch/torch.h>
#include <future>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <ATen/core/TensorBody.h>
#include <torch/csrc/jit/api/module.h>

// The package handed back to the CPU thread
struct InferenceResult {
    torch::Tensor policies; // Shape: [Thread_Batch_Size, 7]
    torch::Tensor values;   // Shape: [Thread_Batch_Size, 1]
};

// The package the CPU thread drops into the queue
struct InferenceRequest {
    torch::Tensor boardTensor;
    std::promise<InferenceResult> promise; // The async contract
    int threadId;
    int gameId;
};

class InferenceQueue {
public:
    // Constructor and Destructor
    InferenceQueue(torch::jit::script::Module& neural_net, int batch_size = 16, int timeout_microsecs = 150);
    ~InferenceQueue();

    // The Await method called by your 16 independent Game/MCTS threads
    InferenceResult infer(torch::Tensor board, int tId = 0, int gId = 0);

    void thread_finished();
private:
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<InferenceRequest*> queue;

    torch::jit::script::Module& model;
    bool running;
    std::thread gpu_thread;

    int max_batch_size;
    int timeout_us;

    // The central background loop
    void gpu_loop();
};
