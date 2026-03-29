#include "Gumbel_MCTS.hpp"
#include "cr_tensor.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/csrc/jit/serialization/import.h>
#include <climits>
#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/stack.h>
#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

using namespace Gumbel_MCTS;

float getGumbelNoise() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dis(1e-6f, 1.0f - 1e-6f);
    float u = dis(gen);
    return -std::log(-std::log(u));
}

float scale_q(float q_norm, int max_visits) {
    float c_visit = 50.0f;
    float c_scale = 1.0f;
    return (c_visit + max_visits) * c_scale * q_norm;
}

Node::Node(float prior, Node* p) : logit(prior), prior_prob(prior), parent(p) {}

Node::~Node() {
    for (auto& pair : children) {
        delete pair.second;
    }
}

float Node::q_value() const {
    if (num_visited == 0) return 0.0f;
    return valueSum / num_visited;
}

MCTS::MCTS(InferenceQueue& q, int t_id, int g_id) : queue(q), thread_id(t_id), game_id(g_id) {
    root = new Node();
}

MCTS::~MCTS() {
    delete root;
}

void MCTS::advanceRoot(int move_id) {
    if (root->children.find(move_id) == root->children.end()) {
        delete root;
        root = new Node();
        return;
    }
    Node* new_root = root->children[move_id];
    root->children.erase(move_id);
    new_root->parent = nullptr;
    delete root;
    root = new_root;
}

std::vector<Node*> MCTS::traverse(ChainReaction& simState, Node* startNode) {
    std::vector<Node*> searchPath;
    Node* crr = startNode;
    searchPath.push_back(crr);

    while (crr->isExpanded && simState.check_winner() == 0) {
        std::vector<int> moves;
        simState.get_valid_moves(moves);

        // 1. Calculate the Root Value (Baseline for unvisited nodes)
        float v_mix = crr->q_value();

        // 2. Calculate pi' (Improved Policy) for all children using Safe Softmax
        std::vector<float> raw_scores(moves.size(), 0.0f);
        std::vector<int> active_moves;
        float max_score = -1e9f;

        // Find max_visits among children for dynamic scaling
        int max_v = 0;
        for (int i = 0; i < moves.size(); i++) {
            Node* child = crr->children[moves[i]];
            if (child->num_visited > max_v) max_v = child->num_visited;
        }

        for (int i = 0; i < moves.size(); i++) {
            int move_id = moves[i];
            active_moves.push_back(move_id);
            Node* child = crr->children[move_id];

            // Negate to get parent's perspective
            float completed_q = (child->num_visited > 0) ? -child->q_value() : v_mix;
            child->completed_q = completed_q;

            float score = child->logit + scale_q(normalize_q(completed_q), max_v);
            raw_scores[i] = score;

            if (score > max_score) {
                max_score = score;
            }
        }

        // Safe Softmax implementation
        std::vector<float> pi_prime(moves.size(), 0.0f);
        float sum_exp = 0.0f;
        for (int i = 0; i < raw_scores.size(); i++) {
            float val = std::exp(raw_scores[i] - max_score);
            pi_prime[i] = val;
            sum_exp += val;
        }

        // Normalize pi'
        for (int i = 0; i < pi_prime.size(); i++) {
            pi_prime[i] /= sum_exp;
        }

        // 3. Deterministic Selection: argmax (pi' - N / (1 + N_total))
        float best_score = -1e9f;
        Node* best_child = nullptr;
        int best_move = -1;

        for (int i = 0; i < active_moves.size(); i++) {
            int move_id = active_moves[i];
            Node* child = crr->children[move_id];

            float visit_fraction = (float)child->num_visited / (1.0f + crr->num_visited);
            float score = pi_prime[i] - visit_fraction;

            if (score > best_score) {
                best_score = score;
                best_child = child;
                best_move = move_id;
            }
        }

        // Step down
        simState.apply_move(best_move);
        crr = best_child;
        searchPath.push_back(crr);
    }

    return searchPath;
}

void MCTS::backpropagate(std::vector<Node*>& searchPath, float leafValue,
    const torch::Tensor& policyBatch, int nn_idx,
    const std::vector<int>& validActions) {

    update_q_bounds(leafValue);

    Node* leaf = searchPath.back();


    if (!leaf->isExpanded && !validActions.empty() && nn_idx != -1) {
        auto logits_accessor = policyBatch.accessor<float, 2>();

        for (int i = 0; i < validActions.size(); i++) {
            int move_id = validActions[i];
            // Directly map move_id (0-95) to the policy head index
            float prob = (move_id >= 0 && move_id < 96) ? logits_accessor[nn_idx][move_id] : -10000.0f;
            leaf->children[move_id] = new Node(prob, leaf);
        }
        leaf->isExpanded = true;
    }

    float currentValue = leafValue;
    for (int i = searchPath.size() - 1; i >= 0; i--) {
        Node* n = searchPath[i];

        n->num_visited += 1;
        n->valueSum += currentValue;
        currentValue = -currentValue;
    }
}

int MCTS::get_best_move() {
    int best_visits = -1;
    int best_move = -1;
    for (const auto& pair : root->children) {
        if (pair.second->num_visited > best_visits) {
            best_visits = pair.second->num_visited;
            best_move = pair.first;
        }
    }
    return best_move;
}

void MCTS::search_batched(const ChainReaction& initial_state, int total_searches, int K, bool is_eval) {

    // 1. Expand Root
    if (!root->isExpanded) {
        std::vector<int> legal_actions;
        initial_state.get_valid_moves(legal_actions);

        if (initial_state.check_winner() != 0 || legal_actions.empty()) {
            return;
        }

        torch::Tensor state_tensor = encode_cr_board(initial_state).unsqueeze(0);
        torch::Tensor policy_logits, value_tensor;

        // --- CALL THE DYNAMIC INFERENCE QUEUE ---
        InferenceResult res = queue.infer(state_tensor, thread_id, game_id);
        policy_logits = res.policies;
        value_tensor = res.values;

        auto logits_accessor = policy_logits.accessor<float, 2>();

        for (int i = 0; i < legal_actions.size(); i++) {
            int move_id = legal_actions[i];
            float logit_val = (move_id >= 0 && move_id < 96) ? logits_accessor[0][move_id] : -10000.0f;
            root->children[move_id] = new Node(logit_val, root);
        }
        root->valueSum = value_tensor[0][0].item<float>();
        root->num_visited = 1;
        root->isExpanded = true;

        update_q_bounds(root->valueSum);

        //float min_logit = 1e9f, max_logit = -1e9f;
        //float min_sq = 1e9f, max_sq = -1e9f;
        //for (auto& p : root->children) {
        //    min_logit = std::min(min_logit, p.second->logit);
        //    max_logit = std::max(max_logit, p.second->logit);
        //    float sq = scale_q(0.0f, 8);  // approximate
        //    min_sq = std::min(min_sq, sq);
        //    max_sq = std::max(max_sq, sq);
        //}
        //std::cout << "Logit range: [" << min_logit << ", " << max_logit << "]\n";
        //std::cout << "scale_q(0, 8) = " << scale_q(0.0f, 8) << "\n";
    }

    // 2. Initialize Gumbel Candidates
    int num_legal_moves = root->children.size();
    K = std::min(K, num_legal_moves);

    std::vector<std::pair<float, int>> scored_moves;
    for (auto& pair : root->children) {
        int move_id = pair.first;
        Node* child = pair.second;
        child->gumbel_noise = is_eval ? 0.0f : getGumbelNoise();
        float score = child->logit + child->gumbel_noise;
        scored_moves.push_back({ score, move_id });
    }
    std::sort(scored_moves.rbegin(), scored_moves.rend());
    std::vector<int> survivors;
    for (int i = 0; i < K; i++) survivors.push_back(scored_moves[i].second);

    // 3. Sequential Halving Math
    int num_phases = std::ceil(std::log2(K));
    if (num_phases == 0) num_phases = 1;
    int phase_budget = total_searches / num_phases;
    int total_sims_spent = 0;

    // 4. The Synchronous Tournament Loop
    for (int phase = 0; phase < num_phases; phase++) {
        int num_survivors = survivors.size();
        int sims_per_move;

        if (phase == num_phases - 1) {
            int remaining_total_budget = total_searches - total_sims_spent;
            sims_per_move = remaining_total_budget / num_survivors;
        }
        else {
            sims_per_move = phase_budget / num_survivors;
        }
        total_sims_spent += (sims_per_move * num_survivors);

        for (int s = 0; s < sims_per_move; s++) {
            std::vector<ChainReaction> batch_states;
            std::vector<std::vector<Node*>> batch_paths;
            std::vector<std::vector<int>> batch_valid_actions;
            std::vector<torch::Tensor> batchTensors;

            for (int move_id : survivors) {
                ChainReaction simState = initial_state;
                simState.apply_move(move_id);
                Node* child_node = root->children[move_id];

                std::vector<Node*> searchPath;
                searchPath.push_back(root);

                std::vector<Node*> deepPath = traverse(simState, child_node);
                searchPath.insert(searchPath.end(), deepPath.begin(), deepPath.end());

                std::vector<int> legal_actions;
                simState.get_valid_moves(legal_actions);

                batch_states.push_back(simState);
                batch_paths.push_back(searchPath);
                batch_valid_actions.push_back(legal_actions);
            }

            for (int b = 0; b < batch_states.size(); ++b) {
                if (batch_states[b].check_winner() == 0) {
                    batchTensors.push_back(encode_cr_board(batch_states[b]));
                }
            }

            torch::Tensor policyBatch, valueBatch;
            if (!batchTensors.empty()) {
                torch::Tensor stacked = torch::stack(batchTensors);

                // --- CALL THE DYNAMIC INFERENCE QUEUE ---
                InferenceResult res = queue.infer(stacked, thread_id, game_id);
                policyBatch = res.policies;
                valueBatch = res.values;
            }

            int nn_idx = 0;
            for (int b = 0; b < batch_states.size(); ++b) {
                float leaf_val = 0.0f;
                std::vector<int> moves = batch_valid_actions[b];

                if (batch_states[b].check_winner() != 0) {
                    // Terminal state hit during traversal. Current player lost.
                    leaf_val = -1.0f;
                    backpropagate(batch_paths[b], leaf_val, policyBatch, -1, moves);
                }
                else {
                    leaf_val = valueBatch[nn_idx][0].item<float>();
                    backpropagate(batch_paths[b], leaf_val, policyBatch, nn_idx, moves);
                    nn_idx++;
                }
            }
        }

        if (phase < num_phases - 1) {
            std::vector<std::pair<float, int>> phase_scores;
            float v_mix = root->q_value();

            int max_root_visits = 0;
            for (int move_id : survivors) {
                if (root->children[move_id]->num_visited > max_root_visits) max_root_visits = root->children[move_id]->num_visited;
            }

            for (int move_id : survivors) {
                Node* child = root->children[move_id];
                float completed_q = (child->num_visited > 0) ? -child->q_value() : v_mix;
                float score = child->gumbel_noise + child->logit + scale_q(normalize_q(completed_q), max_root_visits);
                phase_scores.push_back({ score, move_id });
            }

            std::sort(phase_scores.rbegin(), phase_scores.rend());
            survivors.clear();
            int next_survivor_count = std::max(1, (int)phase_scores.size() / 2);
            for (int i = 0; i < next_survivor_count; i++) survivors.push_back(phase_scores[i].second);
        }
    }

    // --- TARGET MATH ---
    float max_score_target = -1e9f;
    std::vector<float> raw_target_scores(root->children.size(), 0.0f);

    root->final_policy_moves.clear();
    root->final_policy_probs.assign(root->children.size(), 0.0f);

    int max_root_visits = 0;
    for (auto& pair : root->children) {
        if (pair.second->num_visited > max_root_visits) max_root_visits = pair.second->num_visited;
    }

    int i = 0;
    for (auto& pair : root->children) {
        root->final_policy_moves.push_back(pair.first);
        Node* child = pair.second;
        float completed_q = (child->num_visited > 0) ? -child->q_value() : root->q_value();
        float score = child->logit + scale_q(normalize_q(completed_q), max_root_visits);
        raw_target_scores[i] = score;
        if (score > max_score_target) max_score_target = score;
        i++;
    }

    float sum_exp_target = 0.0f;
    for (size_t j = 0; j < raw_target_scores.size(); j++) {
        float val = std::exp(raw_target_scores[j] - max_score_target);
        root->final_policy_probs[j] = val;
        sum_exp_target += val;
    }

    for (size_t j = 0; j < root->final_policy_probs.size(); j++) {
        root->final_policy_probs[j] /= sum_exp_target;
    }
}