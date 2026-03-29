#!/bin/bash
# =============================================================================
#  ChainNano - vast.ai Setup & Build Script
#  Usage:  bash setup_and_build.sh
#  Run this once on a freshly rented vast.ai instance.
# =============================================================================
set -e  # Exit immediately on any error

# ── Colours for output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# STEP 1 — Detect CUDA
# =============================================================================
log "Detecting CUDA installation..."

if ! command -v nvcc &>/dev/null; then
    # nvcc not on PATH, try common locations
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [ -x "$candidate" ]; then
            export PATH="$(dirname $candidate):$PATH"
            break
        fi
    done
fi

command -v nvcc &>/dev/null || fail "nvcc not found. Make sure you rented a CUDA-enabled instance on vast.ai."

CUDA_PATH=$(dirname $(dirname $(which nvcc)))
CUDA_VERSION_FULL=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
CUDA_MAJOR=$(echo $CUDA_VERSION_FULL | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION_FULL | cut -d. -f2)
CUDA_TAG="${CUDA_MAJOR}${CUDA_MINOR}"  # e.g. "130" for 13.0, "121" for 12.1

log "CUDA found at : $CUDA_PATH"
log "CUDA version  : $CUDA_VERSION_FULL (tag: cu${CUDA_TAG})"

# =============================================================================
# STEP 2 — Detect GPU architecture
# =============================================================================
log "Detecting GPU architecture..."

# Use nvidia-smi to get the GPU name and map to compute capability
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log "GPU: $GPU_NAME"

# Map common GPUs to their CMake CUDA arch number
case "$GPU_NAME" in
    *"RTX 50"* | *"GeForce RTX 50"*)
        CUDA_ARCH=120 ;;  # Blackwell (Consumer)
    *"B100"* | *"B200"* | *"GB200"*)
        CUDA_ARCH=100 ;;  # Blackwell (Data Center)
    *"RTX 40"* | *"RTX 4060"* | *"RTX 4070"* | *"RTX 4080"* | *"RTX 4090"*)
        CUDA_ARCH=89 ;;   # Ada Lovelace
    *"RTX 30"* | *"RTX 3060"* | *"RTX 3070"* | *"RTX 3080"* | *"RTX 3090"* | *"A100"*)
        CUDA_ARCH=80 ;;   # Ampere
    *"RTX 20"* | *"RTX 2060"* | *"RTX 2070"* | *"RTX 2080"* | *"T4"*)
        CUDA_ARCH=75 ;;   # Turing
    *"V100"*)
        CUDA_ARCH=70 ;;   # Volta
    *"H100"* | *"H200"*)
        CUDA_ARCH=90 ;;   # Hopper
    *"L40"* | *"L4"*)
        CUDA_ARCH=89 ;;   # Ada (datacenter)
    *)
        warn "Unknown GPU '$GPU_NAME', defaulting to arch 80 (Ampere). Build may not be optimal."
        CUDA_ARCH=80 ;;
esac

log "CUDA architecture: sm_${CUDA_ARCH}"

# =============================================================================
# STEP 3 — Install build tools
# =============================================================================
log "Installing build tools..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    cmake           \
    ninja-build     \
    build-essential \
    wget            \
    unzip           \
    git             \
    2>/dev/null

# Verify cmake version is >= 3.15
CMAKE_VERSION=$(cmake --version | head -1 | sed 's/cmake version //')
CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1)
CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d. -f2)
if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 15 ]); then
    warn "CMake $CMAKE_VERSION is too old (need >= 3.15), installing newer version..."
    wget -q https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh
    bash cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local
    rm cmake-3.28.0-linux-x86_64.sh
fi

log "CMake version : $(cmake --version | head -1)"

# =============================================================================
# STEP 4 — Download matching LibTorch
# =============================================================================
LIBTORCH_DIR="$HOME/libtorch"
PYTORCH_VERSION="2.11.0"

if [ -d "$LIBTORCH_DIR" ] && [ -f "$LIBTORCH_DIR/build-version" ]; then
    EXISTING=$(cat "$LIBTORCH_DIR/build-version")
    log "LibTorch already installed: $EXISTING"

    # Check if it matches our CUDA version
    if echo "$EXISTING" | grep -q "cu${CUDA_TAG}"; then
        log "LibTorch version matches CUDA. Skipping download."
    else
        warn "LibTorch version ($EXISTING) doesn't match CUDA cu${CUDA_TAG}. Re-downloading..."
        rm -rf "$LIBTORCH_DIR"
    fi
fi

if [ ! -d "$LIBTORCH_DIR" ]; then
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.11.0%2Bcu130.zip"

    log "Downloading LibTorch ${PYTORCH_VERSION}+cu${CUDA_TAG}..."
    log "URL: $LIBTORCH_URL"

    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip || \
        fail "Failed to download LibTorch. Check that PyTorch ${PYTORCH_VERSION}+cu${CUDA_TAG} exists at download.pytorch.org"

    log "Extracting LibTorch..."
    unzip -q libtorch.zip -d "$HOME/"
    rm libtorch.zip
    log "LibTorch installed at: $LIBTORCH_DIR"
fi

# =============================================================================
# STEP 5 — Build
# =============================================================================
log "Starting CMake build..."

# Find cudart — handle the targets/ subdirectory layout
CUDART_LIB=$(find "$CUDA_PATH" -name "libcudart.so" 2>/dev/null | head -1)
[ -z "$CUDART_LIB" ] && fail "Could not find libcudart.so under $CUDA_PATH"
CUDA_TARGETS_DIR=$(dirname "$CUDART_LIB")

log "libcudart found at: $CUDART_LIB"

BUILD_DIR="$(pwd)/build"
mkdir -p "$BUILD_DIR"

cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR;$CUDA_PATH;$CUDA_TARGETS_DIR/.." \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH" \
    -DCUDA_NVCC_EXECUTABLE="$CUDA_PATH/bin/nvcc" \
    -DCUDA_CUDART_LIBRARY="$CUDART_LIB" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -B "$BUILD_DIR" \
    . \
    || fail "CMake configuration failed."

cmake --build "$BUILD_DIR" --parallel $(nproc) \
    || fail "Build failed."

# =============================================================================
# STEP 6 — Set up directory structure
# =============================================================================
log "Setting up directory structure..."

mkdir -p bin models replay_buffers scripts

cp "$BUILD_DIR/ChainNano" bin/

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  BUILD SUCCESSFUL${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Executable : $(pwd)/bin/ChainNano"
echo "  Models dir : $(pwd)/models/"
echo "  Replay dir : $(pwd)/replay_buffers/"
echo ""
echo "  Run with:"
echo "    ./bin/ChainNano 1000 0 24 1000 chess_50k_bf16.pt"
echo ""