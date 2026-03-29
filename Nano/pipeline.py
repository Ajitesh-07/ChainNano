import os
import subprocess
import glob
import time
import argparse

from train import train, get_latest_model, MODELS_DIR, REPLAY_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- AUTO-DETECT C++ EXECUTABLE ---
if os.name == 'nt':
    # Assuming standard Visual Studio/CMake output paths
    EXE_PATH = os.path.join(os.path.join(os.path.join("../", "x64"), "Release"), "Nano.exe")
else:
    EXE_PATH = os.path.join(SCRIPT_DIR, "bin", "ChainReactionZero")
    if not os.path.exists(EXE_PATH):
        EXE_PATH = os.path.join(SCRIPT_DIR, "out", "build", "linux-release", "ChainReactionZero")


def get_next_game_id():
    """Finds the highest game ID in the replay buffer to avoid overwriting."""
    files = glob.glob(os.path.join(REPLAY_DIR, "game_*.pt"))
    if not files:
        return 0
    max_id = -1
    for f in files:
        try:
            num_str = os.path.basename(f).replace("game_", "").replace(".pt", "")
            max_id = max(max_id, int(num_str))
        except ValueError:
            pass
    return max_id + 1


def enforce_sliding_window(max_buffer_size):
    """Deletes the oldest games if the buffer exceeds max_buffer_size."""
    files = glob.glob(os.path.join(REPLAY_DIR, "game_*.pt"))
    if len(files) > max_buffer_size:
        files.sort(key=lambda x: int(os.path.basename(x).replace("game_", "").replace(".pt", "")))
        to_delete = files[:-max_buffer_size]
        print(f"[PIPELINE] Sliding window: deleting {len(to_delete)} oldest games...")
        for f in to_delete:
            try:
                os.remove(f)
            except OSError:
                pass


def run_pipeline(
    iterations, games_per_iteration, num_threads, timeout_us,
    epochs, batch_size, lr, accum_steps, workers, max_games, max_buffer
):
    print(f"\n{'='*60}")
    print(f"  INITIALIZING CHAIN REACTION ALPHAZERO PIPELINE")
    print(f"  Executable        : {EXE_PATH}")
    print(f"  Iterations        : {iterations}")
    print(f"  Games / iteration : {games_per_iteration}")
    print(f"  Max Buffer Size   : {max_buffer} games")
    print(f"  Threads           : {num_threads}")
    print(f"  Training LR       : {lr}")
    print(f"  Batch Size        : {batch_size}")
    print(f"{'='*60}\n")

    if not os.path.exists(EXE_PATH):
        print(f"[FATAL] C++ executable not found at: {EXE_PATH}")
        print("Please update the EXE_PATH variable in this script to point to your compiled binary.")
        return

    os.makedirs(REPLAY_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    for iteration in range(iterations):
        print(f"\n\n{'='*50}")
        print(f"  PIPELINE ITERATION {iteration + 1}/{iterations}")
        print(f"{'='*50}")

        # 1. Pick latest model
        latest_model_path, version = get_latest_model(MODELS_DIR)
        print(f"[PIPELINE] Active model : {latest_model_path}  (generation {version})")

        # 2. Self-play
        start_id = get_next_game_id()
        
        # We now pass the absolute path of the model as the 5th argument
        cmd = [
            EXE_PATH,
            str(games_per_iteration),
            str(start_id),
            str(num_threads),
            str(timeout_us),
            latest_model_path
        ]

        print(f"[PIPELINE] Launching C++ self-play  (start_id={start_id}) ...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[FATAL] C++ generator crashed (exit code {e.returncode}). Stopping.")
            break
        except KeyboardInterrupt:
            print(f"\n[PIPELINE] Stopped by user during self-play.")
            break

        # 3. Sliding window cleanup
        enforce_sliding_window(max_buffer)

        # 4. Train
        print(f"\n[PIPELINE] Self-play done. Starting neural net update...")
        try:
            train(
                data_dir=REPLAY_DIR,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                accumulation_steps=accum_steps,
                num_workers=workers,
                max_games=max_games,
            )

        except KeyboardInterrupt:
            print(f"\n[PIPELINE] Stopped by user during training.")
            break

        print(f"[PIPELINE] Iteration {iteration + 1} complete.")
        time.sleep(2)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AlphaZero self-play + training pipeline for Chain Reaction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # --- C++ Generation Arguments ---
    parser.add_argument("--iterations", "-i", type=int, default=10000, help="Total number of self-play → train iterations to run")
    parser.add_argument("--games", "-g", type=int, default=1000, help="Number of self-play games to generate per iteration")
    parser.add_argument("--threads", "-t", type=int, default=24, help="Number of C++ worker threads for self-play")
    parser.add_argument("--timeout", "-u", type=int, default=1000, help="Inference queue timeout in microseconds")
    
    # --- Python Training Arguments ---
    parser.add_argument("--epochs", "-e", type=int, default=1, help="Epochs per generation")
    parser.add_argument("--batch-size", "-b", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--accum-steps", "-a", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--workers", "-w", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--max-games", "-m", type=int, default=25000, help="Max games to sample from replay buffer during training")
    parser.add_argument("--max-buffer", type=int, default=25000, help="Max games to keep on disk (sliding window limit)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        iterations=args.iterations,
        games_per_iteration=args.games,
        num_threads=args.threads,
        timeout_us=args.timeout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        accum_steps=args.accum_steps,
        workers=args.workers,
        max_games=args.max_games,
        max_buffer=args.max_buffer,
    )