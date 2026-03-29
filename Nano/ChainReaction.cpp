#include "ChainReaction.h"
#include <algorithm>

// Define static members
uint8_t ChainReaction::capacities[N_SQUARES];
uint8_t ChainReaction::num_neighbors[N_SQUARES];
uint8_t ChainReaction::neighbors[N_SQUARES][4];

void ChainReaction::init_tables() {
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int idx = r * COLS + c;
            int n_idx = 0;

            if (r > 0) neighbors[idx][n_idx++] = (r - 1) * COLS + c;
            if (r < ROWS - 1) neighbors[idx][n_idx++] = (r + 1) * COLS + c;
            if (c > 0) neighbors[idx][n_idx++] = r * COLS + (c - 1);
            if (c < COLS - 1) neighbors[idx][n_idx++] = r * COLS + (c + 1);

            num_neighbors[idx] = n_idx;
            capacities[idx] = n_idx; // Capacity is exactly the number of neighbors
        }
    }
}

ChainReaction::ChainReaction() {
    my_mask = Bitboard96();
    opp_mask = Bitboard96();
    std::fill(orbs, orbs + N_SQUARES, 0);
    moves_played = 0;
}

void ChainReaction::get_valid_moves(std::vector<int>& moves) const {
    moves.clear();
    // Valid moves are anywhere the opponent DOES NOT have a piece
    Bitboard96 valid = ~opp_mask;

    int idx;
    Bitboard96 temp = valid;
    while ((idx = temp.pop_lsb()) != -1) {
        if (idx < N_SQUARES) { // Safety bound
            moves.push_back(idx);
        }
    }
}

void ChainReaction::apply_move(int move_idx) {
    // 1. Place the initial orb
    orbs[move_idx]++;
    my_mask.set(move_idx);

    // 2. High-speed fixed-size stack queue for cascades (avoids dynamic allocation)
    uint8_t q[2048];
    int head = 0, tail = 0;

    if (orbs[move_idx] >= capacities[move_idx]) {
        q[tail++] = move_idx;
    }

    // 3. Resolve Explosions
    while (head < tail) {
        int curr = q[head++];

        // It's possible a previous explosion in the queue reduced this cell
        if (orbs[curr] < capacities[curr]) continue;

        // Early exit: if opponent is wiped out, stop evaluating cascades
        if (moves_played >= 1 && opp_mask.is_empty()) break;

        uint8_t cap = capacities[curr];
        orbs[curr] -= cap;

        if (orbs[curr] == 0) {
            my_mask.clear(curr);
        }
        else if (orbs[curr] >= cap) {
            q[tail++] = curr; // Re-evaluate if it double-exploded
        }

        // Distribute to neighbors
        for (int i = 0; i < num_neighbors[curr]; ++i) {
            int n = neighbors[curr][i];
            orbs[n]++;

            // Ownership flip: Transfer from opponent to me
            if (opp_mask.test(n)) {
                opp_mask.clear(n);
                my_mask.set(n);
            }
            else if (!my_mask.test(n)) {
                my_mask.set(n);
            }

            // Push to queue only on exact hit to prevent duplicate enqueuing
            if (orbs[n] == capacities[n]) {
                q[tail++] = n;
            }
        }
    }

    moves_played++;

    // 4. NEGAMAX SWAP: The board perspective completely flips.
    // The previous opponent is now "my_mask", and I am "opp_mask".
    Bitboard96 temp_mask = my_mask;
    my_mask = opp_mask;
    opp_mask = temp_mask;
}

int ChainReaction::check_winner() const {
    if (moves_played < 2) return 0; // Nobody can win before both play once

    // Remember, perspective just swapped at the end of apply_move!
    // If 'opp_mask' (the player who just moved) wiped out 'my_mask' (the player about to move)
    if (my_mask.is_empty()) return -1; // Current player lost

    // If somehow the player who just moved wiped themselves out (technically impossible in normal rules, but safe)
    if (opp_mask.is_empty()) return 1;

    return 0; // Game ongoing
}

void ChainReaction::print_board() const {
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int idx = r * COLS + c;
            char owner = '.';
            if (my_mask.test(idx)) owner = 'X';      // Current player to move
            else if (opp_mask.test(idx)) owner = 'O';// Opponent

            std::cout << owner << (int)orbs[idx] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Moves: " << moves_played << "\n";
}