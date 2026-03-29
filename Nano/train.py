import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import glob
import re
import argparse
from tqdm import tqdm
from torch.utils.data import IterableDataset, get_worker_info
import math

# TURN ON CUDNN AUTO-TUNER
torch.backends.cudnn.benchmark = True

# ── Resolve directory structure ───────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# As requested, looking directly in these folders relative to the script
MODELS_DIR  = os.path.join(SCRIPT_DIR, "models")
REPLAY_DIR  = os.path.join(SCRIPT_DIR, "replay_buffer")
LOG_PATH    = os.path.join(SCRIPT_DIR, "rl_training_log.txt")

# ── Helpers ───────────────────────────────────────────────────────────────────

def vram_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A (CPU)"
    alloc   = torch.cuda.memory_allocated(device)  / 1024**3
    reserve = torch.cuda.memory_reserved(device)   / 1024**3
    total   = torch.cuda.get_device_properties(device).total_memory / 1024**3
    peak    = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"alloc {alloc:.2f} GB  |  reserved {reserve:.2f} GB  |  peak {peak:.2f} GB  |  total {total:.2f} GB"


def get_latest_model(directory: str):
    """Finds the TorchScript model with the highest version number in directory."""
    # Matches chainreaction_v1_bf16.pt, chainreaction_v2_bf16.pt, etc.
    pattern = re.compile(r"chainreaction_(?:v)?(\d+)?.*\.pt$")
    models  = glob.glob(os.path.join(directory, "*.pt"))

    latest_model = None
    max_version  = -1

    for model_path in models:
        if "game_" in model_path or "checkpoint" in model_path:
            continue
        match = pattern.search(os.path.basename(model_path))
        if match:
            version = int(match.group(1)) if match.group(1) else 0
            if version > max_version:
                max_version  = version
                latest_model = model_path

    # Fallback to the initial traced model if no versioned models exist
    if latest_model is None:
        latest_model = os.path.join(directory, "chainreaction_bf16.pt")
        max_version  = 0

    return latest_model, max_version


# ── 1. Streaming Dataset ──────────────────────────────────────────────────────

class StreamingCRDataset(IterableDataset):
    def __init__(self, data_dir: str, max_games: int = None, buffer_size_games: int = 500):
        super().__init__()
        self.data_dir = data_dir
        self.buffer_size_games = buffer_size_games

        self.all_files = glob.glob(os.path.join(data_dir, "*.pt"))
        np.random.shuffle(self.all_files)

        if max_games and max_games < len(self.all_files):
            self.all_files = self.all_files[:max_games]

        tqdm.write(f" [DATA] Streaming dataset: {len(self.all_files):,} games  |  buffer: {buffer_size_games} games/worker")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            files_for_worker = self.all_files
        else:
            per_worker = int(math.ceil(len(self.all_files) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end   = min(start + per_worker, len(self.all_files))
            files_for_worker = self.all_files[start:end]

        file_idx = 0
        while file_idx < len(files_for_worker):
            buffer_states, buffer_policies, buffer_rewards = [], [], []
            games_loaded = 0

            while games_loaded < self.buffer_size_games and file_idx < len(files_for_worker):
                f = files_for_worker[file_idx]
                file_idx += 1
                try:
                    # weights_only=True is safer and slightly faster for pure tensor files
                    tensors = list(torch.jit.load(f, map_location='cpu').parameters())
                    buffer_states.append(tensors[0])
                    buffer_policies.append(tensors[1])
                    buffer_rewards.append(tensors[2])
                    games_loaded += 1
                except Exception:
                    pass

            if not buffer_states:
                break

            b_states   = torch.cat(buffer_states,   dim=0)
            b_policies = torch.cat(buffer_policies, dim=0)
            b_rewards  = torch.cat(buffer_rewards,  dim=0).squeeze(-1)

            indices = torch.randperm(b_states.size(0))
            for i in indices:
                yield {
                    'state':         b_states[i],
                    'policy_target': b_policies[i],
                    'value_target':  b_rewards[i],
                }


# ── 2. AlphaZero Dual Loss ────────────────────────────────────────────────────

class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # CrossEntropyLoss expects target probabilities (which we have) when passed as floats
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn  = nn.MSELoss()

    def forward(self, pred_policy, pred_value, target_policy, target_value):
        p_loss     = self.policy_loss_fn(pred_policy, target_policy)
        v_loss     = self.value_loss_fn(pred_value.view(-1), target_value.view(-1))
        return p_loss + v_loss, p_loss, v_loss


# ── 3. Training Loop ──────────────────────────────────────────────────────────

def train(
    data_dir:           str   = REPLAY_DIR,
    epochs:             int   = 1,
    batch_size:         int   = 1024,
    lr:                 float = 1e-3,
    accumulation_steps: int   = 4,
    num_workers:        int   = 4,
    max_games:          int   = 25000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"\n [SYSTEM] Training on : {device}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        tqdm.write(f" [SYSTEM] GPU         : {props.name}  |  VRAM: {props.total_memory / 1024**3:.2f} GB\n")
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Find and load latest model
    latest_model_path, current_version = get_latest_model(MODELS_DIR)
    tqdm.write(f" [SYSTEM] Model       : {latest_model_path}  (generation {current_version})")

    if not os.path.exists(latest_model_path):
        tqdm.write(f" [FATAL] Model not found: {latest_model_path}")
        tqdm.write(f"         Place your seed model inside: {MODELS_DIR}")
        return

    try:
        model = torch.jit.load(latest_model_path, map_location=device)
        model.train()
    except Exception as e:
        tqdm.write(f" [FATAL] Could not load model: {e}")
        return

    # 2. Build dataset and dataloader
    dataset    = StreamingCRDataset(data_dir, max_games=max_games)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
    )

    # Count exact positions for accurate progress bar
    tqdm.write(f" [DATA] Counting positions across {len(dataset.all_files):,} files...")
    exact_total_positions = 0
    count_bar = tqdm(dataset.all_files, desc=" Counting", leave=False, dynamic_ncols=True)
    for f in count_bar:
        try:
            tensors = list(torch.jit.load(f, map_location='cpu').parameters())
            exact_total_positions += tensors[0].size(0)
        except Exception:
            pass
            
    total_steps = max(1, exact_total_positions // batch_size)
    tqdm.write(f" [DATA] {exact_total_positions:,} positions  →  {total_steps:,} steps/epoch")

    criterion = AlphaZeroLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4,
        fused=(device.type == "cuda"),
    )

    global_step = 0
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n========================================\n")
        f.write(f"Starting CR Generation {current_version + 1}: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"max_games={max_games}  epochs={epochs}  batch_size={batch_size}\n")
        f.write("========================================\n")

    optimizer.zero_grad(set_to_none=True)

    epoch_bar = tqdm(
        range(epochs), desc="  Epochs", unit="epoch", position=0, leave=True,
        bar_format="  {desc}  [{bar:30}]  {n}/{total}  {elapsed}<{remaining}",
    )

    for epoch in epoch_bar:
        running_loss   = 0.0
        running_p_loss = 0.0
        running_v_loss = 0.0
        smooth_loss    = None
        alpha          = 0.02
        total_samples  = 0
        epoch_start    = time.monotonic()
        timer_5k_start = time.monotonic()

        step_bar = tqdm(
            dataloader,
            desc=f"  Epoch {epoch+1:>2}/{epochs}",
            unit="batch",
            position=1,
            leave=False,
            dynamic_ncols=True,
            total=total_steps,
            bar_format="  {desc}  [{bar:40}]  {n_fmt}/{total_fmt}  {rate_fmt}  ETA {remaining}  {postfix}",
        )

        for step, batch in enumerate(step_bar):
            global_step += 1

            states    = batch['state'].to(device,          non_blocking=True)
            p_targets = batch['policy_target'].to(device,  non_blocking=True)
            v_targets = batch['value_target'].to(device,   non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                p_logits, v_pred     = model(states)
                loss, p_loss, v_loss = criterion(p_logits, v_pred, p_targets, v_targets)
                scaled_loss          = loss / accumulation_steps

                # --- ADD THIS SANITY CHECK BLOCK ---
                if global_step == 10:
                    tqdm.write(f"\n{'='*50}")
                    tqdm.write(f" [SHAPE SANITY CHECK] - Step 1")
                    tqdm.write(f"   v_pred (Model)        : {list(v_pred.shape)}")
                    tqdm.write(f"   v_targets (Batch)     : {list(v_targets.shape)}")
                    tqdm.write(f"   v_pred.view(-1)       : {list(v_pred.view(-1).shape)}")
                    tqdm.write(f"   v_targets.view(-1)    : {list(v_targets.view(-1).shape)}")
                    tqdm.write(f"{'='*50}\n")
                # -----------------------------------

            scaled_loss.backward()

            if global_step % accumulation_steps == 0 or (step + 1) == total_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss   += loss.item()
            running_p_loss += p_loss.item()
            running_v_loss += v_loss.item()
            total_samples  += states.size(0)

            smooth_loss = loss.item() if smooth_loss is None else (1 - alpha) * smooth_loss + alpha * loss.item()
            elapsed     = time.monotonic() - epoch_start
            pos_per_sec = total_samples / elapsed if elapsed > 0 else 0.0

            postfix = {
                "loss":  f"{smooth_loss:.4f}",
                "pol":   f"{p_loss.item():.4f}",
                "val":   f"{v_loss.item():.4f}",
                "pos/s": f"{pos_per_sec:,.0f}",
            }
            if device.type == "cuda":
                postfix["vram"] = f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
            step_bar.set_postfix(ordered_dict=postfix, refresh=False)

            if global_step % 5000 == 0:
                time_for_5k = time.monotonic() - timer_5k_start
                with open(LOG_PATH, "a") as f:
                    f.write(
                        f"Epoch: {epoch+1} | Step: {global_step} | "
                        f"Loss: {smooth_loss:.4f} | "
                        f"5k-time: {time_for_5k:.2f}s | "
                        f"Pos/s: {pos_per_sec:,.0f}\n"
                    )
                timer_5k_start = time.monotonic()

        step_bar.close()

        avg_loss   = running_loss   / total_steps
        avg_p_loss = running_p_loss / total_steps
        avg_v_loss = running_v_loss / total_steps
        elapsed    = time.monotonic() - epoch_start
        vram_line  = f"\n              VRAM  →  {vram_str(device)}" if device.type == "cuda" else ""

        tqdm.write(
            f"  ✓ Generation {current_version + 1} (Epoch {epoch+1}) complete!\n"
            f"      Avg Loss : {avg_loss:.4f}  (Pol: {avg_p_loss:.4f} | Val: {avg_v_loss:.4f})\n"
            f"      Speed    : {total_samples / elapsed:,.0f} pos/s  [{elapsed:.1f}s]{vram_line}"
        )

    epoch_bar.close()

    # Save new model
    os.makedirs(MODELS_DIR, exist_ok=True)
    new_version    = current_version + 1
    new_model_path = os.path.join(MODELS_DIR, f"chainreaction_v{new_version}_bf16.pt")
    model.eval()
    model.save(new_model_path)

    tqdm.write(f"\n ── Final VRAM ──────────────────────────────────────")
    tqdm.write(f" {vram_str(device)}")
    tqdm.write(f"\n [SYSTEM] Saved Generation {new_version} → {new_model_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chain Reaction AlphaZero training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      "-e", type=int,   default=1,      help="Epochs per generation")
    parser.add_argument("--batch-size",  "-b", type=int,   default=1024,   help="Batch size")
    parser.add_argument("--lr",          "-l", type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--accum-steps", "-a", type=int,   default=4,      help="Gradient accumulation steps")
    parser.add_argument("--workers",     "-w", type=int,   default=4,      help="DataLoader worker processes")
    parser.add_argument("--max-games",   "-m", type=int,   default=25000,  help="Max games to sample from replay buffer")
    return parser.parse_args()


if __name__ == "__main__":
    if not os.path.exists(REPLAY_DIR):
        print(f"[ERROR] Replay buffer not found: {REPLAY_DIR}")
        print(f"        Run the C++ data generator first.")
    elif not os.path.exists(MODELS_DIR):
        print(f"[ERROR] Models directory not found: {MODELS_DIR}")
        print(f"        Create a 'models/' folder next to this script and place your seed model there.")
    else:
        args = parse_args()
        train(
            data_dir=REPLAY_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            accumulation_steps=args.accum_steps,
            num_workers=args.workers,
            max_games=args.max_games,
        )