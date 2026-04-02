#!/bin/bash
# Pack all Scheme beta + delta results for download.
#
# Usage:
#   bash scripts/pack_beta_results.sh
#
# Output:
#   beta_export/
#   ├── raw/          ← 训练原始数据 (history, policy, value, cache)
#   ├── delta/        ← quantum_analysis 结果
#   ├── figures/      ← 已生成的图
#   ├── configs/      ← 训练配置
#   └── scripts/      ← 分析脚本 (可复现)
#
#   beta_export.tar.gz  ← 一个文件下载

set -euo pipefail
cd "$(dirname "$0")/.."

EXPORT=beta_export
rm -rf "$EXPORT"

echo "=== Packing beta results ==="

# 1. Raw training data (history + policy weights, skip large logs)
echo "[1/5] Raw training data..."
mkdir -p "$EXPORT/raw"
for d in results/beta/*/; do
    name=$(basename "$d")
    mkdir -p "$EXPORT/raw/$name"
    # Copy essentials only (skip .log which can be huge)
    for f in history.json policy.pt value.pt cache_info.json; do
        [ -f "$d/$f" ] && cp "$d/$f" "$EXPORT/raw/$name/"
    done
done

# 2. Delta analysis results
echo "[2/5] Delta analysis results..."
mkdir -p "$EXPORT/delta"
cp results/delta_*.json "$EXPORT/delta/" 2>/dev/null || echo "  (no delta files)"

# 3. Figures
echo "[3/5] Figures..."
mkdir -p "$EXPORT/figures"
for f in figures/beta_*.pdf figures/beta_*.png; do
    [ -f "$f" ] && cp "$f" "$EXPORT/figures/"
done

# 4. Configs
echo "[4/5] Configs..."
mkdir -p "$EXPORT/configs"
cp configs/beta.yaml "$EXPORT/configs/" 2>/dev/null || true

# 5. Analysis scripts (for reproducibility)
echo "[5/5] Scripts..."
mkdir -p "$EXPORT/scripts"
cp scripts/run_beta.py "$EXPORT/scripts/"
cp coupling_rl/quantum_analysis.py "$EXPORT/scripts/"
cp coupling_rl/plot_results.py "$EXPORT/scripts/"

# Summary
echo ""
echo "--- Contents ---"
find "$EXPORT" -type f | sort | head -60
N_FILES=$(find "$EXPORT" -type f | wc -l)
echo "... $N_FILES files total"

# Tar
echo ""
echo "Compressing..."
tar czf beta_export.tar.gz "$EXPORT"
SIZE=$(du -sh beta_export.tar.gz | cut -f1)
echo ""
echo "=== Done: beta_export.tar.gz ($SIZE) ==="
echo "Download with:  scp $(hostname):$(pwd)/beta_export.tar.gz ."
