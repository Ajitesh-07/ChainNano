#include "MCTS.hpp"
#include "cr_tensor.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/stack.h>
#include <torch/utils.h>

// --- NODE IMPL ---
Node::Node(float prior, Node* p) : prior_prob(prior), parent(p) {}

Node::~Node() {
    for (auto& pair : children) {
        delete pair.second;
    }
}

float Node::q_value() const {
    return numVisited == 0 ? 0.0f : valueSum / numVisited;
}


// --- MCTS IMPL ---
// Model and device initialization completely replaced by the Inference Queue
MCTS::MCTS(InferenceQueue& q, int t_id, int g_id) : queue(q), thread_id(t_id), game_id(g_id) {
    root = new Node();
}

MCTS::~MCTS() {
    delete root;
}

void MCTS::advanceRoot(int move_id) {
    if (root->children.find(move_id) != root->children.end()) {
        Node* new_root = root->children[move_id];
        root->children.erase(move_id);
        new_root->parent = nullptr;
        delete root;
        root = new_root;
    }
    else {
        delete root;
        root = new Node();
    }
}

std::vector<Node*> MCTS::traverse(ChainReaction& simState) {
    std::vector<Node*> searchPath = { root };
    Node* crr = root;

    // Run until a terminal state is hit or we find an unexpanded node
    while (crr->isExpanded && simState.check_winner() == 0) {
        std::vector<int> moves;
        simState.get_valid_moves(moves);

        float best_puct = -1e9;
        Node* best_child = nullptr;
        int best_move = -1;

        for (int move : moves) {
            if (crr->children.find(move) == crr->children.end()) continue;

            Node* child = crr->children[move];
            float q = -child->q_value();
            float u = c_puct * child->prior_prob * std::sqrt((float)(crr->numVisited + crr->virtualVisits)) /
                (1.0f + child->numVisited + child->virtualVisits);

            if (q + u > best_puct) {
                best_puct = q + u;
                best_child = child;
                best_move = move;
            }
        }

        // Apply the move (ChainReaction natively handles the Negamax swap internally)
        simState.apply_move(best_move);
        crr = best_child;
        searchPath.push_back(crr);
    }

    // Apply Virtual Loss to prevent other threads/sims from exploring this exact same path
    for (Node* n : searchPath) {
        n->virtualVisits += 1;
    }
    return searchPath;
}

void MCTS::backpropagate(std::vector<Node*>& searchPath, float leafValue, const torch::Tensor& policyBatch, int nn_idx, const std::vector<int>& validActions) {
    Node* leaf = searchPath.back();

    if (!leaf->isExpanded && !validActions.empty() && nn_idx != -1) {
        auto prob_accessor = policyBatch.accessor<float, 2>();
        float sum_prob = 0.0f;

        for (int move : validActions) {
            // Chain Reaction moves map 1:1 with the policy index (0 to 95)
            float prob = (move >= 0 && move < 96) ? prob_accessor[nn_idx][move] : 0.0f;
            leaf->children[move] = new Node(prob, leaf);
            sum_prob += prob;
        }

        if (sum_prob > 0.0f) {
            for (auto& pair : leaf->children) {
                pair.second->prior_prob /= sum_prob;
            }
        }
        else {
            for (auto& pair : leaf->children) {
                pair.second->prior_prob = 1.0f / leaf->children.size();
            }
        }
        leaf->isExpanded = true;
    }

    // Standard Negamax Backprop & Virtual Loss Removal
    float current_val = leafValue;
    for (int i = searchPath.size() - 1; i >= 0; --i) {
        Node* n = searchPath[i];
        n->virtualVisits -= 1;
        n->numVisited += 1;
        n->valueSum += current_val;
        current_val = -current_val; // Flip the perspective as we climb back up the tree
    }
}

void MCTS::search_batched(const ChainReaction& initial_state, int total_searches, int batch_size) {
    int steps = total_searches / batch_size;

    for (int step = 0; step < steps; ++step) {
        std::vector<std::vector<Node*>> searchPaths;
        std::vector<ChainReaction> batchStates;
        std::vector<torch::Tensor> batchTensors;

        // 1. MCTS TRAVERSAL
        for (int b = 0; b < batch_size; ++b) {
            ChainReaction sim = initial_state;
            std::vector<Node*> path = traverse(sim);
            searchPaths.push_back(path);
            batchStates.push_back(sim);
        }

        // 2. STATE ENCODING
        for (int b = 0; b < batch_size; ++b) {
            if (batchStates[b].check_winner() == 0) {
                batchTensors.push_back(encode_cr_board(batchStates[b]));
            }
        }

        torch::Tensor policyBatch, valueBatch;
        if (!batchTensors.empty()) {
            // 3. TENSOR STACKING
            torch::Tensor stacked = torch::stack(batchTensors);

            // 4. CALL THE INFERENCE QUEUE (Replaces the local PyTorch model execution)
            InferenceResult res = queue.infer(stacked, thread_id, game_id);

            // 5. EXTRACT & APPLY SOFTMAX
            // The Inference Queue returns raw logits from our ResTNet. We must apply softmax here.
            policyBatch = torch::softmax(res.policies, 1);
            valueBatch = res.values;
        }

        // 6. BACKPROPAGATION & TREE UPDATE
        int nn_idx = 0;

        for (int b = 0; b < batch_size; ++b) {
            float leaf_val = 0.0f;
            std::vector<int> moves;
            batchStates[b].get_valid_moves(moves);

            if (batchStates[b].check_winner() != 0) {
                // If a state is terminal, the player whose turn it is has LOST. (-1.0f)
                // (Chain Reaction rarely draws, but an infinite loop cutoff yields 0.0f handled gracefully)
                leaf_val = -1.0f;
                backpropagate(searchPaths[b], leaf_val, policyBatch, -1, moves);
            }
            else {
                leaf_val = valueBatch[nn_idx][0].item<float>();
                backpropagate(searchPaths[b], leaf_val, policyBatch, nn_idx, moves);
                nn_idx++;
            }
        }
    }
}

int MCTS::get_best_move() {
    int best_visits = -1;
    int best_move = -1;
    for (const auto& pair : root->children) {
        if (pair.second->numVisited > best_visits) {
            best_visits = pair.second->numVisited;
            best_move = pair.first;
        }
    }
    return best_move;
}