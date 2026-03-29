#pragma once
#include <torch/torch.h>
#include "chainreaction.h"
#include <ATen/core/TensorBody.h>
#include <torch/types.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

inline torch::Tensor encode_cr_board(const ChainReaction& env) {
    // Ch 0-2: my orbs with count 1,2,3
    // Ch 3-5: opp orbs with count 1,2,3
    // Ch 6:   cell capacity (static geometry)
    // Ch 7:   turn indicator
    auto tensor = torch::zeros({ 8, 12, 8 }, torch::kFloat32);
    auto t = tensor.accessor<float, 3>();

    for (int r = 0; r < 12; ++r) {
        for (int c = 0; c < 8; ++c) {
            int idx = r * 8 + c;
            int orbs = env.orbs[idx];
            int cap = ChainReaction::capacities[idx];

            // Ch 6: normalized capacity, present everywhere including empty cells
            t[6][r][c] = (float)cap / 4.0f;

            if (orbs == 0) continue;

            int orb_ch = orbs - 1; // maps count 1→0, 2→1, 3→2

            if (env.my_mask.test(idx))
                t[orb_ch][r][c] = 1.0f;
            else if (env.opp_mask.test(idx))
                t[3 + orb_ch][r][c] = 1.0f;
        }
    }

    // Ch 7: turn — all 1s if it's player 0's turn
    float turn = (env.moves_played % 2 == 0) ? 1.0f : 0.0f;
    for (int r = 0; r < 12; ++r)
        for (int c = 0; c < 8; ++c)
            t[7][r][c] = turn;

    return tensor;
}
