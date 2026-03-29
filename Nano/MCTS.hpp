#pragma once

#include <vector>
#include <unordered_map>
#include <torch/torch.h>
#include <ATen/core/TensorBody.h>
#include <string>
#include "ChainReaction.h"
#include "InferenceQueue.h"

struct Node {
    float prior_prob = 0.0f;
    int numVisited = 0;
    int virtualVisits = 0; // Essential for Batched Inference
    float valueSum = 0.0f;
    bool isExpanded = false;
    Node* parent = nullptr;

    // Maps the integer of the Chain Reaction move (0-95) to a Node
    std::unordered_map<int, Node*> children;

    Node(float prior = 0.0f, Node* p = nullptr);
    ~Node();
    float q_value() const;
};

// Note: ChessState was completely removed. We just pass ChainReaction directly!

class MCTS {
public:
    Node* root;

    // Swapped the internal PyTorch model/device for the Centralized Queue Pipeline
    InferenceQueue& queue;
    int thread_id;
    int game_id;

    float c_puct = 1.5f;

    // Updated Constructor for Queue ingestion
    MCTS(InferenceQueue& q, int t_id, int g_id);
    ~MCTS();

    // Replaced uint16_t with int for our 96 actions
    void advanceRoot(int move_id);

    // The core Batched Inference Search uses ChainReaction now
    void search_batched(const ChainReaction& initial_state, int total_searches, int batch_size);

    // Returns int (0-95) instead of chess::Move
    int get_best_move();

private:
    std::vector<Node*> traverse(ChainReaction& simState);

    // Removed leafBoard (no longer needed for get_policy_index) and replaced Movelist
    void backpropagate(std::vector<Node*>& searchPath, float leafValue,
        const torch::Tensor& policyBatch, int nn_idx,
        const std::vector<int>& validActions);
};