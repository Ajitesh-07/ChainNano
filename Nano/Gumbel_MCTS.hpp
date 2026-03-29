#pragma once

#include <unordered_map>
#include <vector>
#include <deque>
#include <string>
#include <cstdint>
#include <torch/torch.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/api/module.h>
#include <ATen/core/TensorBody.h>

// Replaced chess.hpp with our Chain Reaction engine
#include "ChainReaction.h"
#include "InferenceQueue.h"
#include <iomanip>
#include <ios>
#include <iostream>

namespace Gumbel_MCTS {

    struct Node {
        float logit = 0.0f;
        float prior_prob = 0.0f;

        float gumbel_noise = 0.0f;
        float completed_q = 0.0f;

        int   num_visited = 0;
        float valueSum = 0.0f;
        bool  isExpanded = false;
        Node* parent = nullptr;

        // Changed from uint16_t to int for the 96 Chain Reaction cells
        std::unordered_map<int, Node*> children;

        std::vector<int> final_policy_moves;
        std::vector<float> final_policy_probs;

        Node(float prior = 0.0f, Node* p = nullptr);
        ~Node();

        float q_value() const;
        float improved_policy() const;
    };

    // Note: ChessState was completely removed. We pass ChainReaction directly.

    class MCTS {
    public:
        Node* root;
        InferenceQueue& queue; // <--- The central pipeline
        int thread_id;         // <--- Thread tracker
        int game_id;           // <--- Game tracker

        // Constructor remains aligned with the Inference Queue
        MCTS(InferenceQueue& q, int t_id, int g_id);
        ~MCTS();

        void advanceRoot(int move_id);

        // Swapped ChessState for ChainReaction
        void search_batched(const ChainReaction& initial_state, int total_searches, int K, bool is_eval);

        // Returns the cell index (0-95) instead of chess::Move
        int get_best_move();

        float q_min = 1e9f;   // global across entire search
        float q_max = -1e9f;   // global across entire search

        void update_q_bounds(float q) {
            // Symmetric Update: Track both the current player's score 
            // and the implied opponent's score to safely handle Negamax flips.
            if (q < q_min) q_min = q;
            if (q > q_max) q_max = q;

            if (-q < q_min) q_min = -q;
            if (-q > q_max) q_max = -q;
        }

        float normalize_q(float q) const {
            if (q_max - q_min < 1e-6f) return 0.5f;
            
            float norm = (q - q_min) / (q_max - q_min);
            return norm;
        }
    private:
        std::vector<Node*> traverse(ChainReaction& simState, Node* startNode);

        // Swapped chess::Movelist for std::vector<int> and dropped the leafBoard parameter
        void backpropagate(std::vector<Node*>& searchPath, float leafValue,
            const torch::Tensor& policyBatch, int nn_idx,
            const std::vector<int>& validActions);
    };
};