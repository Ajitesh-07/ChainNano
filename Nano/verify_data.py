import torch
import numpy as np
import os

# ── Constants & Helpers ───────────────────────────────────────────────────────

ROWS = 12
COLS = 8
NUM_ACTIONS = ROWS * COLS

def get_capacity(r: int, c: int) -> int:
    """Returns the critical mass (capacity) of a cell."""
    is_top = (r == 0)
    is_bot = (r == ROWS - 1)
    is_left = (c == 0)
    is_right = (c == COLS - 1)
    
    if (is_top or is_bot) and (is_left or is_right):
        return 2 # Corners
    if is_top or is_bot or is_left or is_right:
        return 3 # Edges
    return 4 # Centers

def idx_to_move(idx: int) -> str:
    if not (0 <= idx < NUM_ACTIONS):
        return "FAIL"
    r, c = divmod(idx, COLS)
    return f"({r:2d},{c:2d})"

# ── State tensor → reconstructed board ───────────────────────────────────────
def tensor_to_board(state_tensor: torch.Tensor):
    """
    Reconstructs the Exact Chain Reaction board from the 8-channel tensor.
    Returns a 2D list of (Owner, OrbCount).
    Owner is 'Us' (Current Player), 'Them' (Opponent), or None.
    """
    planes = state_tensor.numpy() # Shape: (8, 12, 8)
    board = [[(None, 0) for _ in range(COLS)] for _ in range(ROWS)]
    
    for r in range(ROWS):
        for c in range(COLS):
            # Us: Channels 0 (1 orb), 2 (2 orbs), 4 (3 orbs)
            if planes[0, r, c] > 0.5:
                board[r][c] = ('Us', 1)
            elif planes[1, r, c] > 0.5:
                board[r][c] = ('Us', 2)
            elif planes[2, r, c] > 0.5:
                board[r][c] = ('Us', 3)
                
            # Them: Channels 1 (1 orb), 3 (2 orbs), 5 (3 orbs)
            elif planes[3, r, c] > 0.5:
                board[r][c] = ('Them', 1)
            elif planes[4, r, c] > 0.5:
                board[r][c] = ('Them', 2)
            elif planes[5, r, c] > 0.5:
                board[r][c] = ('Them', 3)
                
    return board

def get_legal_moves(board) -> list:
    """A move is legal if the cell is empty or owned by 'Us'."""
    legal_idxs = []
    for r in range(ROWS):
        for c in range(COLS):
            owner, _ = board[r][c]
            if owner in (None, 'Us'):
                legal_idxs.append(r * COLS + c)
    return legal_idxs

def board_to_ascii(board) -> str:
    lines = []
    lines.append("    " + "  ".join([f"{c}" for c in range(COLS)]))
    lines.append("  +" + "---" * COLS + "+")
    
    for r in range(ROWS):
        row_strs = []
        for c in range(COLS):
            owner, orbs = board[r][c]
            if owner is None:
                row_strs.append(" . ")
            elif owner == 'Us':
                # Green for Current Player
                row_strs.append(f"\033[92mO{orbs}\033[0m ")
            else:
                # Red for Opponent
                row_strs.append(f"\033[91mX{orbs}\033[0m ")
        lines.append(f"{r:2d}| " + "".join(row_strs) + "|")
        
    lines.append("  +" + "---" * COLS + "+")
    return "\n".join(lines)

# ── Aggregated checks ─────────────────────────────────────────────────────────

def check_legal_move_coverage(states: torch.Tensor,
                               policies: torch.Tensor,
                               n_moves: int = 20):
    print(f"\n{'─'*65}")
    print(f"  LEGAL-MOVE PROBABILITY COVERAGE  (first {n_moves} moves)")
    print(f"{'─'*65}")
    print(f"  {'Move':>4}  {'#Legal':>6}  {'Legal Mass':>12}  {'Top Predicted Move'}")
    print(f"  {'----':>4}  {'------':>6}  {'----------':>12}  {'------------------'}")

    bad_moves = []
    
    for t in range(min(n_moves, states.shape[0])):
        board = tensor_to_board(states[t])
        pi    = policies[t] # (96,)

        legal_idxs = get_legal_moves(board)
        legal_mass = pi[legal_idxs].sum().item() if legal_idxs else 0.0

        # Top predicted move
        top_idx  = pi.argmax().item()
        top_str  = idx_to_move(top_idx)
        is_legal = top_idx in legal_idxs
        
        top_display = f"{top_str} ✅" if is_legal else f"{top_str} ❌ (Illegal)"

        flag = "" if legal_mass > 0.99 else "  ⚠️  LOW"
        print(f"  {t:>4}  {len(legal_idxs):>6}  {legal_mass:>12.4f}{flag}  {top_display}")

        if legal_mass < 0.99:
            bad_moves.append(t)

    if bad_moves:
        print(f"\n  ⚠️  Moves with legal-mass < 0.99: {bad_moves}")
    else:
        print(f"\n  ✅ All sampled positions have ≥99% probability on legal moves.")


def print_move_spotlight(states: torch.Tensor,
                          policies: torch.Tensor,
                          rewards: torch.Tensor,
                          move_indices: list,
                          top_k: int = 5):
    """Deep-dive: board diagram + top-k policy moves for chosen plies."""
    for t in move_indices:
        if t >= states.shape[0]:
            continue
            
        board = tensor_to_board(states[t])
        pi    = policies[t]
        z     = rewards[t].item()

        print(f"\n{'═'*52}")
        print(f"  PLY {t}  |  Negamax Perspective  |  Reward z = {z:+.1f}")
        print(f"{'═'*52}")
        print(board_to_ascii(board))
        print(f"\n  Top-{top_k} policy moves:")
        print(f"  {'Rank':<5} {'Index':>6} {'Prob':>8}  {'Move':<10} {'Legal?'}")
        print(f"  {'----':<5} {'-----':>6} {'----':>8}  {'----':<10} {'------'}")

        # Decode top-K
        probs, idxs = torch.topk(pi, top_k)
        legal_idxs = get_legal_moves(board)
        
        for rank, (prob, idx) in enumerate(zip(probs.tolist(), idxs.tolist()), 1):
            move_str = idx_to_move(idx)
            legal = idx in legal_idxs
            tick = "✅" if legal else "❌"
            print(f"  {rank:<5} {idx:>6} {prob:>8.4f}  {move_str:<10} {tick}")

        legal_mass = pi[legal_idxs].sum().item() if legal_idxs else 0.0
        print(f"\n  Total probability on legal moves : {legal_mass:.4f}")
        print(f"  Total probability on all moves   : {pi.sum().item():.4f}")


def check_reward_alternation(rewards: torch.Tensor):
    """Verify that rewards flip sign every ply (Negamax winner's perspective)."""
    print(f"\n{'─'*60}")
    print(f"  REWARD ALTERNATION CHECK (NEGAMAX)")
    print(f"{'─'*60}")
    vals = rewards.squeeze().tolist()
    violations = []
    for i in range(len(vals) - 1):
        if vals[i] != 0.0 and vals[i+1] != 0.0:
            if vals[i] * vals[i+1] > 0:           # same sign → bad
                violations.append(i)
                
    print(f"  Last move reward       : {vals[-1]:+.1f}")
    print(f"  First 10 rewards       : {[f'{v:+.0f}' for v in vals[:10]]}")
    if violations:
        print(f"  ⚠️  Sign violations at plies: {violations[:10]}")
    else:
        print(f"  ✅ Rewards alternate correctly throughout the game.")


# ── Main ──────────────────────────────────────────────────────────────────────

def verify_game_data(filepath: str,
                     spotlight_moves: list = None,
                     coverage_moves: int   = 20,
                     top_k: int            = 5):

    print(f"\n{'═'*60}")
    print(f"  CHAIN REACTION DATA SANITY CHECK")
    print(f"  {filepath}")
    print(f"{'═'*60}")

    try:
        states, policies, rewards = list(torch.jit.load(filepath).parameters())
    except Exception as e:
        print(f"[FATAL] Could not load: {e}")
        return

    # Convert to float32 for policy arithmetic
    states   = states.float()
    policies = policies.float()
    rewards  = rewards.float()

    L = states.shape[0]
    print(f"\n  Total plies in game : {L}")

    # ── 1. Shape / dtype / NaN ──────────────────────────────────────────────
    print(f"\n[1] STATES")
    print(f"    Shape        : {list(states.shape)}  expected [{L}, 6, 12, 8]")
    print(f"    dtype        : {states.dtype}")
    print(f"    NaN present  : {torch.isnan(states).any().item()}")
    print(f"    Value range  : [{states.min().item():.3f}, {states.max().item():.3f}]")

    print(f"\n[2] POLICIES")
    print(f"    Shape        : {list(policies.shape)}  expected [{L}, 96]")
    sums = policies.sum(dim=1)
    print(f"    Sum per row  : min={sums.min().item():.5f}  "
          f"max={sums.max().item():.5f}  "
          f"mean={sums.mean().item():.5f}")
    bad_rows = (torch.abs(sums - 1.0) > 1e-3).sum().item()
    print(f"    Rows ≠ 1.0   : {bad_rows}  {'✅' if bad_rows == 0 else '⚠️'}")
    
    sparsity = (policies == 0).float().mean().item()
    print(f"    Sparsity     : {sparsity:.2%} zeros")

    print(f"\n[3] REWARDS")
    print(f"    Shape        : {list(rewards.shape)}  expected [{L}, 1]")
    unique_vals = torch.unique(rewards).tolist()
    print(f"    Unique values: {[f'{v:+.1f}' for v in unique_vals]}")

    # ── 2. Reward alternation ───────────────────────────────────────────────
    check_reward_alternation(rewards)

    # ── 3. Legal-move coverage sweep ───────────────────────────────────────
    check_legal_move_coverage(states, policies, n_moves=coverage_moves)

    # ── 4. Per-move spotlight ───────────────────────────────────────────────
    if spotlight_moves is None:
        # Auto-pick: move 0, middle, last
        spotlight_moves = sorted(list(set([0, 1, 2, 3, 4, 5, L // 3,  L // 2, max(0, L - 1)])))

    print_move_spotlight(states, policies, rewards,
                         move_indices=spotlight_moves, top_k=top_k)

    print(f"\n{'═'*60}")
    print(f"  CHECK COMPLETE")
    print(f"{'═'*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_FILE = "replay_buffer/game_0.pt" # Update as needed

    # if not os.path.exists(TEST_FILE):
    #     print(f"File not found: {TEST_FILE}. Make sure the C++ pipeline ran successfully!")
    verify_game_data(
        filepath        = "replay_buffer/game_40287.pt",
        spotlight_moves = None,
        coverage_moves  = 30,  
        top_k           = 5,    
    )