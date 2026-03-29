import torch
import torch.nn.functional as F
import os
import glob
from collections import deque
from copy import deepcopy

# ── 1. Your Original Python Engine ─────────────────────────────────────────────
class ChainReactionGame:
    def __init__(self, rows=12, cols=8):
        self.rows = rows
        self.cols = cols
        self.board = [[(None, 0) for _ in range(cols)] for _ in range(rows)]
        self.moves_played = {0: 0, 1: 0}

    def get_valid_moves(self, player_id):
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                owner, _count = self.board[r][c]
                if owner is None or owner == player_id:
                    moves.append(r * self.cols + c) # Flattened to 0-95
        return moves

    def capacity(self, r, c):
        is_top = (r == 0)
        is_bottom = (r == self.rows - 1)
        is_left = (c == 0)
        is_right = (c == self.cols - 1)
        if (is_top and is_left) or (is_top and is_right) or (is_bottom and is_left) or (is_bottom and is_right):
            return 2
        if is_top or is_bottom or is_left or is_right:
            return 3
        return 4

    def neighbors(self, r, c):
        res = []
        if r - 1 >= 0: res.append((r - 1, c))
        if r + 1 < self.rows: res.append((r + 1, c))
        if c - 1 >= 0: res.append((r, c - 1))
        if c + 1 < self.cols: res.append((r, c + 1))
        return res

    def check_winner(self):
        if self.moves_played[0] == 0 or self.moves_played[1] == 0:
            return None
        counts = {0: 0, 1: 0}
        for r in range(self.rows):
            for c in range(self.cols):
                owner, orb_count = self.board[r][c]
                if owner in (0, 1) and orb_count > 0:
                    counts[owner] += 1
        if counts[0] == 0 and counts[1] > 0: return 1
        if counts[1] == 0 and counts[0] > 0: return 0
        return None

    def apply_move(self, player_id, move_idx):
        r, c = divmod(move_idx, self.cols)
        owner, count = self.board[r][c]
        
        self.board[r][c] = (player_id, count + 1)
        self.moves_played[player_id] += 1

        q = deque()
        if self.board[r][c][1] >= self.capacity(r, c):
            q.append((r, c))

        while q:
            cr, cc = q.popleft()
            cur_owner, cur_count = self.board[cr][cc]

            if cur_count < self.capacity(cr, cc):
                continue

            exploding_owner = cur_owner
            if self.check_winner() is not None:
                break # Early exit Parity

            cap = self.capacity(cr, cc)
            remaining = cur_count - cap
            if remaining > 0:
                self.board[cr][cc] = (exploding_owner, remaining)
                if remaining >= cap:
                    q.append((cr, cc))
            else:
                self.board[cr][cc] = (None, 0)

            for nr, nc in self.neighbors(cr, cc):
                n_owner, n_count = self.board[nr][nc]
                self.board[nr][nc] = (exploding_owner, n_count + 1)
                if self.board[nr][nc][1] == self.capacity(nr, nc):
                    q.append((nr, nc))


# ── 2. AlphaZero Tensor Encoder ───────────────────────────────────────────────
def encode_board(game: ChainReactionGame, current_player: int) -> torch.Tensor:
    """Matches the exact 8-channel logic of our highly optimized C++ encoder."""
    tensor = torch.zeros(8, 12, 8, dtype=torch.float32)
    
    # Calculate total plies to mimic the C++ env.moves_played logic
    total_moves = game.moves_played[0] + game.moves_played[1]
    turn_val = 1.0 if (total_moves % 2 == 0) else 0.0
    
    for r in range(12):
        for c in range(8):
            # Channel 6: Static Capacity Map
            cap = game.capacity(r, c)
            tensor[6, r, c] = cap / 4.0
            
            # Channel 7: Turn Parity
            tensor[7, r, c] = turn_val
            
            owner, orbs = game.board[r][c]
            if orbs == 0:
                continue
                
            # Map exact orb counts to channels based on ownership
            if owner == current_player:
                if orbs == 1: tensor[0, r, c] = 1.0
                elif orbs == 2: tensor[2, r, c] = 1.0
                elif orbs >= 3: tensor[4, r, c] = 1.0
            else:
                if orbs == 1: tensor[1, r, c] = 1.0
                elif orbs == 2: tensor[3, r, c] = 1.0
                elif orbs >= 3: tensor[5, r, c] = 1.0
                
    return tensor

def print_board(game: ChainReactionGame):
    print("\n   " + "  ".join([f"{c}" for c in range(8)]))
    print("  +" + "---"*8 + "+")
    for r in range(12):
        row_strs = []
        for c in range(8):
            owner, orbs = game.board[r][c]
            if owner is None:
                row_strs.append(" . ")
            elif owner == 0:
                row_strs.append(f"\033[92mO{orbs}\033[0m ") # Green for Player 0
            else:
                row_strs.append(f"\033[91mX{orbs}\033[0m ") # Red for Player 1
        print(f"{r:2d}| " + "".join(row_strs) + "|")
    print("  +" + "---"*8 + "+")


# ── 3. The Play Loop ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Find latest model
    models = glob.glob(os.path.join("models", "*.pt"))
    if not models:
        print("[ERROR] No models found in models/ directory.")
        exit(1)
    
    latest_model_path = max(models, key=os.path.getctime)
    print(f"[SYSTEM] Loading Brain: {latest_model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(latest_model_path, map_location=device)
    model.eval()

    game = ChainReactionGame()
    
    # You are Player 0 (Green), AI is Player 1 (Red)
    HUMAN = 1
    AI = 0
    current_turn = 0

    print("\n=== CHAIN REACTION VS ALPHAZERO ===")
    print("You are Green (O). AI is Red (X).")
    
    while game.check_winner() is None:
        print_board(game)
        
        legal_moves = game.get_valid_moves(current_turn)
        
        if current_turn == HUMAN:
            while True:
                try:
                    move_str = input("\nEnter your move (row col) e.g., '4 3': ")
                    r, c = map(int, move_str.strip().split())
                    move_idx = r * 8 + c
                    if move_idx in legal_moves:
                        game.apply_move(current_turn, move_idx)
                        break
                    else:
                        print("Illegal move! That cell is owned by the opponent.")
                except ValueError:
                    print("Invalid input. Please enter 'row col'.")
        else:
            print("\nAI is thinking...")
            # Encode board from AI's perspective
            tensor = encode_board(game, AI).unsqueeze(0).to(device).to(torch.bfloat16)
            
            with torch.no_grad():
                p_logits, v_pred = model(tensor)
                
            p_logits = p_logits.squeeze(0).cpu()
            
            # --- 1. Get RAW Network Probabilities ---
            # Convert raw output logits into a proper percentage distribution
            raw_probs = torch.softmax(p_logits, dim=0)
            
            # Find what the network thinks is the absolute best move
            raw_best_idx = torch.argmax(raw_probs).item()
            raw_best_prob = raw_probs[raw_best_idx].item()
            raw_r, raw_c = divmod(raw_best_idx, 8)
            
            # --- 2. Apply Legal Mask ---
            masked_logits = p_logits.clone()
            mask = torch.ones(96, dtype=torch.bool)
            mask[legal_moves] = False
            masked_logits[mask] = -1e9 # Destroy illegal logits
            
            # --- 3. Find Actual Move Chosen ---
            # Recompute softmax over the MASKED logits so percentages sum to 100% properly
            masked_probs = torch.softmax(masked_logits, dim=0)
            best_move = torch.argmax(masked_probs).item()
            best_move_prob = masked_probs[best_move].item()
            r, c = divmod(best_move, 8)
            
            # --- PRINT DIAGNOSTICS ---
            print(f"┌──────────────────────────────────────┐")
            print(f"│ AI Evaluation (Win Confidence: {((v_pred.item() + 1) / 2) * 100:.1f}%) │")
            print(f"├──────────────────────────────────────┤")

            # Did the raw network guess an illegal move?
            if raw_best_idx not in legal_moves:
                print(f"│ \033[93mRaw Top Move : {raw_r} {raw_c}  ({raw_best_prob*100:.1f}%) - ILLEGAL\033[0m │")
            else:
                print(f"│ Raw Top Move : {raw_r} {raw_c}  ({raw_best_prob*100:.1f}%) - LEGAL   │")
                
            print(f"│ AI Plays     : {r} {c}  ({best_move_prob*100:.1f}%)          │")
            print(f"└──────────────────────────────────────┘")
            
            game.apply_move(current_turn, best_move)

        current_turn = 1 - current_turn

    print_board(game)
    winner = "Human" if game.check_winner() == HUMAN else "AI"
    print(f"\n=== GAME OVER! {winner} Wins! ===")