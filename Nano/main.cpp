#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <torch/torch.h>
#include <c10/core/Device.h>
#include <torch/cuda.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/stack.h>
#include <ATen/Parallel.h>
#include <torch/serialize.h>
#include <torch/types.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/headeronly/core/DeviceType.h>
#include <cstdint>
#include <functional>
#include <ios>
#include <filesystem>
#include <torch/csrc/jit/serialization/import.h>

// Chain Reaction Includes
#include "Gumbel_MCTS.hpp"
#include "cr_tensor.hpp"
#include "ChainReaction.h"
#include "InferenceQueue.h"

// --- Helper: Save Game Tensors to Disk ---
void save_game_data(
    const std::vector<torch::Tensor>& states,
    const std::vector<torch::Tensor>& policies,
    const std::vector<torch::Tensor>& rewards,
    int game_id
) {
    torch::Tensor stacked_states = torch::stack(states);
    torch::Tensor stacked_policies = torch::stack(policies);
    torch::Tensor stacked_rewards = torch::stack(rewards);

    std::vector<torch::Tensor> tensors_to_save = { stacked_states, stacked_policies, stacked_rewards };

    std::filesystem::create_directories("replay_buffer");

    std::string filename = "replay_buffer/game_" + std::to_string(game_id) + ".pt";
    torch::save(tensors_to_save, filename);
}

// --- Dashboard Tracker ---
struct ThreadProgress {
    std::atomic<int> games_completed{ 0 };
    int total_games_assigned{ 0 };
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

void worker_task(int thread_id, int start_idx, int games_to_play, InferenceQueue& queue, std::vector<ThreadProgress>& progress) {
    progress[thread_id].start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < games_to_play; i++) {
        int game_id = start_idx + i;

        ChainReaction state;
        Gumbel_MCTS::MCTS mcts(queue, thread_id, game_id);

        std::vector<torch::Tensor> history_states;
        std::vector<torch::Tensor> history_policies;
        std::vector<int> history_players;

        int move_limit = 200;

        while (state.check_winner() == 0 && state.moves_played < move_limit) {

            mcts.search_batched(state, 32, 8, false);

            history_states.push_back(encode_cr_board(state));
            history_players.push_back(state.moves_played % 2);

            torch::Tensor policy_target = torch::zeros({ 96 }, torch::kFloat32);
            for (size_t j = 0; j < mcts.root->final_policy_moves.size(); ++j) {
                int move_id = mcts.root->final_policy_moves[j];
                float prob = mcts.root->final_policy_probs[j];
                policy_target[move_id] = prob;
            }
            history_policies.push_back(policy_target);

            int best_move = -1;

            if (state.moves_played < 12) {
                std::vector<double> probs;
                std::vector<int> moves;
                for (size_t j = 0; j < mcts.root->final_policy_moves.size(); ++j) {
                    moves.push_back(mcts.root->final_policy_moves[j]);
                    probs.push_back(mcts.root->final_policy_probs[j]);
                }
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                std::mt19937 gen(std::random_device{}());
                best_move = moves[dist(gen)];
            }
            else {
                best_move = mcts.get_best_move();
            }

            state.apply_move(best_move);
            mcts.advanceRoot(best_move);
        }

        int winner_code = state.check_winner();
        std::vector<torch::Tensor> history_rewards;

        int current_parity = state.moves_played % 2;
        int winning_parity = 1 - current_parity;

        for (int p : history_players) {
            float r;
            if (winner_code == 0) r = 0.0f;
            else r = (p == winning_parity) ? 1.0f : -1.0f;

            history_rewards.push_back(torch::tensor({ r }, torch::kFloat32));
        }

        save_game_data(history_states, history_policies, history_rewards, game_id);

        progress[thread_id].games_completed++;
    }

    queue.thread_finished();
}

// --- Main Pipeline ---
void runMultiThreadSelfPlay(int total_games, int start_id, int num_threads, int timeout_us, const std::string& model_path) {
    std::cout << "\033[2J\033[H";
    std::cout << "=================================================\n";
    std::cout << "   GUMBEL MUZERO CHAIN REACTION GENERATOR\n";
    std::cout << "   Threads: " << num_threads << " | Total Games: " << total_games << "\n";
    std::cout << "   Model  : " << model_path << "\n";
    std::cout << "=================================================\n\n";

    ChainReaction::init_tables();

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    std::vector<ThreadProgress> progress(num_threads);
    std::vector<std::thread> workers;

    int current_start_idx = start_id;
    int base_games = total_games / num_threads;
    int leftover_games = total_games % num_threads;

    torch::jit::script::Module shared_model;
    try {
        shared_model = torch::jit::load(model_path);
        shared_model.to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        shared_model.eval();
        shared_model = torch::jit::optimize_for_inference(shared_model);

        std::cout << "[SYSTEM] Model loaded successfully: " << model_path << "\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "[ERROR] Could not load model from path: " << model_path << "\n";
        std::cerr << "[ERROR] " << e.msg() << "\n";
        return;
    }

    InferenceQueue queue(shared_model, num_threads, timeout_us);

    for (int i = 0; i < num_threads; i++) {
        int my_games = base_games + (i < leftover_games ? 1 : 0);
        progress[i].total_games_assigned = my_games;

        workers.emplace_back(worker_task, i, current_start_idx, my_games, std::ref(queue), std::ref(progress));
        current_start_idx += my_games;
    }

    // --- DASHBOARD UI LOOP ---
    for (int i = 0; i < num_threads + 2; i++) std::cout << "\n";

    bool all_done = false;
    while (!all_done) {
        std::cout << "\033[" << num_threads + 2 << "A";

        double total_gps = 0.0;
        int total_completed = 0;
        all_done = true;

        for (int i = 0; i < num_threads; i++) {
            int comp = progress[i].games_completed.load();
            int tot = progress[i].total_games_assigned;
            total_completed += comp;

            if (comp < tot) all_done = false;

            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - progress[i].start_time;

            double gps = (elapsed.count() > 0 && comp > 0) ? (comp / elapsed.count()) : 0.0;
            total_gps += gps;

            int bar_width = 20;
            int filled = (tot > 0) ? (comp * bar_width) / tot : 0;
            std::string bar = "[";
            for (int b = 0; b < bar_width; b++) {
                if (b < filled) bar += "=";
                else if (b == filled && comp < tot) bar += ">";
                else bar += " ";
            }
            bar += "]";

            std::cout << "\033[K"
                << "Thread " << std::setw(2) << i << " | "
                << bar << " " << std::setw(3) << comp << "/" << std::setw(3) << tot
                << " | Speed: " << std::fixed << std::setprecision(2) << gps << " games/sec\n";
        }

        std::cout << "\033[K-----------------------------------------------------------------\n";
        std::cout << "\033[K GLOBAL PIPELINE  | Games: " << std::setw(3) << total_completed
            << "/" << total_games
            << " | TOTAL SPEED: " << std::fixed << std::setprecision(2) << total_gps << " games/sec\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    for (auto& t : workers) {
        t.join();
    }

    std::cout << "\n[SYSTEM] ALL " << total_games << " GAMES COMPLETED AND SAVED.\n";
}

void print_usage(const char* program_name) {
    std::cout << "\nUsage:\n";
    std::cout << "  " << program_name << " [total_games] [start_id] [num_threads] [timeout_us] [model_path]\n\n";
    std::cout << "Arguments (all optional, use defaults if omitted):\n";
    std::cout << "  total_games   Number of games to generate     (default: 1000)\n";
    std::cout << "  start_id      Starting game ID for filenames   (default: 0)\n";
    std::cout << "  num_threads   Number of worker threads         (default: 24)\n";
    std::cout << "  timeout_us    Inference queue timeout (µs)     (default: 500)\n";
    std::cout << "  model_path    Path to TorchScript .pt model    (default: chain_reaction_bf16.pt)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "\n";
    std::cout << "  " << program_name << " 500\n";
    std::cout << "  " << program_name << " 500 0 8 500 models/v2.pt\n";
    std::cout << "  " << program_name << " 1000 5000 24 300 /checkpoints/iter_42.pt\n\n";
}

int main(int argc, char* argv[]) {
    // Default settings
    int         total_games = 1000;
    int         starting_id = 7000;
    int         num_threads = 1;
    int         timeout_us = 500;
    std::string model_path = "chain_reaction_bf16.pt";

    if (argc >= 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    if (argc >= 2) total_games = std::stoi(argv[1]);
    if (argc >= 3) starting_id = std::stoi(argv[2]);
    if (argc >= 4) num_threads = std::stoi(argv[3]);
    if (argc >= 5) timeout_us = std::stoi(argv[4]);
    if (argc >= 6) model_path = std::string(argv[5]);

    // Validate model path is not empty
    if (model_path.empty()) {
        std::cerr << "[ERROR] Model path cannot be empty.\n";
        print_usage(argv[0]);
        return 1;
    }

    at::set_num_threads(1);

    runMultiThreadSelfPlay(total_games, starting_id, num_threads, timeout_us, model_path);

    return 0;
}