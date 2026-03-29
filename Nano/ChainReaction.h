#pragma once

#include <cstdint>
#include <vector>
#include <iostream>

// Cross-compiler wrapper for Trailing Zero Count
inline int count_trailing_zeros64(uint64_t val) {
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, val);
    return (int)index;
#else
    return __builtin_ctzll(val);
#endif
}

inline int count_trailing_zeros32(uint32_t val) {
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, val);
    return (int)index;
#else
    return __builtin_ctz(val);
#endif
}

// Custom 96-bit Bitboard to handle the 12x8 grid (0 to 95)
struct Bitboard96 {
    uint64_t lo; // Squares 0-63
    uint32_t hi; // Squares 64-95 (Changed to uint32_t)

    Bitboard96() : lo(0), hi(0) {}
    Bitboard96(uint64_t l, uint32_t h) : lo(l), hi(h) {}

    inline void set(int idx) {
        if (idx < 64) lo |= (1ULL << idx);
        else hi |= (1U << (idx - 64)); // Use 1U for 32-bit
    }

    inline void clear(int idx) {
        if (idx < 64) lo &= ~(1ULL << idx);
        else hi &= ~(1U << (idx - 64)); // Use 1U for 32-bit
    }

    inline bool test(int idx) const {
        if (idx < 64) return (lo & (1ULL << idx)) != 0;
        return (hi & (1U << (idx - 64))) != 0;
    }

    inline bool is_empty() const {
        return lo == 0 && hi == 0;
    }

    inline Bitboard96 operator~() const {
        // Since hi is uint32_t, ~hi is automatically masked to 32 bits natively
        return Bitboard96(~lo, ~hi);
    }

    inline int pop_lsb() {
        if (lo) {
            int idx = count_trailing_zeros64(lo);
            lo &= lo - 1;
            return idx;
        }
        else if (hi) {
            int idx = count_trailing_zeros32(hi); // Use standard 32-bit ctz
            hi &= hi - 1;
            return idx + 64;
        }
        return -1;
    }
};

class ChainReaction {
public:
    static constexpr int ROWS = 12;
    static constexpr int COLS = 8;
    static constexpr int N_SQUARES = ROWS * COLS;

    // Negamax Perspective: Always evaluated from the current player's point of view
    Bitboard96 my_mask;
    Bitboard96 opp_mask;
    uint8_t orbs[N_SQUARES];
    int moves_played;

    ChainReaction();

    // Setup the static lookup tables (Call this once at program start)
    static void init_tables();

    // MCTS Interface
    void get_valid_moves(std::vector<int>& moves) const;
    void apply_move(int move_idx);

    // Returns 1 if current player won, -1 if lost, 0 if game is ongoing
    int check_winner() const;

    // Debugging
    void print_board() const;
    // Static lookup tables for O(1) physics
    static uint8_t capacities[N_SQUARES];
    static uint8_t num_neighbors[N_SQUARES];
    static uint8_t neighbors[N_SQUARES][4];
};