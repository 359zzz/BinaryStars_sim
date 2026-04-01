#!/bin/bash
# Scheme beta: Coupling-Aware RL (5 variants)
# Usage: cd E:/BinaryStars_sim && bash scripts/run_beta.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_PARALLEL=${MAX_PARALLEL:-16}

echo "=== Scheme beta: Coupling-Aware RL ==="
echo "Running 5 variants x 5 seeds = 25 runs (max $MAX_PARALLEL parallel)"

run_experiment() {
    variant=$1
    seed=$2
    echo "  Starting $variant seed=$seed"
    python -u -m coupling_rl.train_ppo \
        --config configs/beta.yaml \
        --variant "$variant" \
        --seed "$seed" \
        > "results/beta/${variant}_seed${seed}.log" 2>&1
    echo "  Finished $variant seed=$seed"
}
export -f run_experiment

mkdir -p results/beta

# Launch all jobs
for variant in vanilla geometric coupling quantum_c quantum_decomp; do
    for seed in $(seq 0 4); do
        echo "$variant $seed"
    done
done | xargs -P "$MAX_PARALLEL" -n 2 bash -c 'run_experiment "$@"' _

# Plot results
echo "Generating figures..."
python -m coupling_rl.plot_results --save_dir results/beta

echo "=== Scheme beta complete ==="
