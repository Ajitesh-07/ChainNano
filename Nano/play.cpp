#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <torch/script.h>
#include <torch/torch.h>

#include "ChainReaction.h" 
#include "MCTS.hpp"          
#include "Gumbel_MCTS.hpp"
#include "cr_tensor.hpp"     
#include "InferenceQueue.h" 
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/cuda.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/headeronly/core/DeviceType.h>

// ANSI Color Codes for the terminal
const std::string RESET = "\033[0m";
const std::string RED = "\033[1;31m";
const std::string BLUE = "\033[1;34m";
const std::string BOLD = "\033[1m";

void print_absolute_board(const ChainReaction& game, int current_player) {
    // Because the game uses Negamax, my_mask is always the player WHOSE TURN IT IS.
    // We map them back to absolute players (0 = Human/Red, 1 = AI/Blue) for rendering.
    Bitboard96 p0_mask = (current_player == 0) ? game.my_mask : game.opp_mask;
    Bitboard96 p1_mask = (current_player == 1) ? game.my_mask : game.opp_mask;

    std::cout << "\n    " << BOLD << "0 1 2 3 4 5 6 7" << RESET << "\n";
    std::cout << "  +-----------------+\n";

    for (int r = 0; r < ChainReaction::ROWS; ++r) {
        // Print row label
        if (r < 10) std::cout << BOLD << r << " | " << RESET;
        else std::cout << BOLD << r << "| " << RESET;

        for (int c = 0; c < ChainReaction::COLS; ++c) {
            int idx = r * ChainReaction::COLS + c;
            int orbs = game.orbs[idx];

            if (orbs == 0) {
                std::cout << ". ";
            }
            else if (p0_mask.test(idx)) {
                std::cout << RED << orbs << RESET << " "; // Human is Red
            }
            else if (p1_mask.test(idx)) {
                std::cout << BLUE << orbs << RESET << " "; // AI is Blue
            }
            else {
                std::cout << "? "; // Fallback, shouldn't happen
            }
        }
        std::cout << BOLD << "|" << RESET << "\n";
    }
    std::cout << "  +-----------------+\n";
    std::cout << "Total Moves Played: " << game.moves_played << "\n\n";
}

int mainPlay() {
    std::cout << BOLD << "=== LOADING NANO BRAIN ===" << RESET << "\n";
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    torch::jit::script::Module model;

    try {
        model = torch::jit::load("models/chainreaction_v22_bf16.pt", device);
        model.eval();
        std::cout << "Model loaded successfully on " << (device.is_cuda() ? "GPU" : "CPU") << "!\n";
    }
    catch (const c10::Error& e) {
        std::cerr << RED << "[ERROR] Could not load model. Ensure the path is correct and compiled via torch.jit.script()." << RESET << "\n";
        return -1;
    }

    // Initialize Native Queue and MCTS
    InferenceQueue queue(model, 1, 150);
    Gumbel_MCTS::MCTS mcts(queue, 0, 0);

    ChainReaction::init_tables();
    ChainReaction game;

    int human_player = 0;
    int current_player = 0; // 0 = Human, 1 = AI
    const int COLS = ChainReaction::COLS;

    std::cout << "\n" << BOLD << "=== CHAIN REACTION C++ ENGINE ===" << RESET << "\n";
    std::cout << "You are " << RED << "Player 0 (Red)" << RESET << ". AI is " << BLUE << "Player 1 (Blue)" << RESET << ".\n";

    // 0 = ongoing, 1 = current player won, -1 = current player lost
    while (game.check_winner() == 0) {
        print_absolute_board(game, current_player);

        if (current_player == human_player) {
            std::cout << RED << "Your Turn." << RESET << " Enter move (row col): ";

            int r, c;
            // Robust input loop to catch letters or out of bounds
            while (true) {
                if (std::cin >> r >> c) {
                    int move_idx = r * COLS + c;
                    std::vector<int> valid_moves;
                    game.get_valid_moves(valid_moves);

                    if (r >= 0 && r < ChainReaction::ROWS && c >= 0 && c < COLS &&
                        std::find(valid_moves.begin(), valid_moves.end(), move_idx) != valid_moves.end()) {

                        game.apply_move(move_idx);
                        mcts.advanceRoot(move_idx);
                        break; // Valid move, break the input loop
                    }
                    else {
                        std::cout << "Illegal move! Square is out of bounds or owned by AI. Try again: ";
                    }
                }
                else {
                    std::cout << "Invalid input. Please enter numbers (e.g., '5 4'): ";
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
            }
        }
        else {
            std::cout << BLUE << "AI is thinking (MCTS)..." << RESET << "\n";

            // 800 total searches, processed in batches of 16
            mcts.search_batched(game, 800, 16, true);

            int best_move = mcts.get_best_move();
            int r = best_move / COLS;
            int c = best_move % COLS;

            std::cout << BLUE << "AI Plays: " << r << " " << c << RESET << "\n";

            game.apply_move(best_move);
            mcts.advanceRoot(best_move);
        }

        current_player = 1 - current_player; // Swap turn
    }

    // Determine absolute winner
    // If we break the loop, check_winner is either 1 or -1 for the CURRENT player
    print_absolute_board(game, current_player);
    std::cout << BOLD << "\n=== GAME OVER ===" << RESET << "\n";

    int win_status = game.check_winner();
    int winner = -1;

    if (win_status == -1) winner = 1 - current_player; // Current player lost, other won
    else if (win_status == 1) winner = current_player; // Current player won

    if (winner == human_player) {
        std::cout << RED << "Congratulations! You defeated the Nano Brain!" << RESET << "\n";
    }
    else {
        std::cout << BLUE << "Nano Brain wins! Better luck next time." << RESET << "\n";
    }

    return 0;
}