#!/bin/bash
# Scheme gamma: World Model Transfer (5 models: MLP, J-MLP, C-MLP, DeLaN, CRBA)
# Usage: cd E:/BinaryStars_sim && bash scripts/run_gamma.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Scheme gamma: World Model Transfer ==="
echo "Models: MLP, J-MLP (classical), C-MLP (quantum), DeLaN, CRBA"

# Step 1: Train all models on source domain
echo "[1/3] Training world models on 0 kg..."
python -m world_model.train --config configs/gamma.yaml

# Step 2: Evaluate transfer (includes indirect_coupling_rmse)
echo "[2/3] Evaluating transfer across payloads..."
python -m world_model.evaluate_transfer --config configs/gamma.yaml

# Step 3: Plot results (transfer curves + indirect coupling comparison)
echo "[3/3] Generating figures..."
python -m world_model.plot_results --results results/gamma/transfer_results.json

echo "=== Scheme gamma complete ==="
