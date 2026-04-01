"""Analyze Scheme gamma results for paper tables and figures.

Reads results/gamma/transfer_results.json and outputs:
1. Table: 1-step RMSE (5 models × 3 payloads × fine-tune sweep)
2. Table: Indirect coupling RMSE (smoking gun comparison)
3. Key statistics for paper text
4. LaTeX-ready tables

Usage:
    python scripts/analyze_gamma.py
    python scripts/analyze_gamma.py --results results/gamma/transfer_results.json
"""

from __future__ import annotations

import argparse
import json
import sys


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_zero_shot(results: dict) -> None:
    """Table 1: Zero-shot transfer (N=0 fine-tune) across payloads."""
    print_header("Table 1: Zero-shot Transfer (no fine-tuning)")

    models = ["mlp", "j_mlp", "c_mlp", "delan", "crba"]
    model_labels = {"mlp": "MLP", "j_mlp": "J-MLP", "c_mlp": "C-MLP",
                    "delan": "DeLaN", "crba": "CRBA"}

    payloads = sorted(results.keys())

    # Header
    print(f"\n{'Model':<10}", end="")
    for p in payloads:
        payload_kg = results[p]["payload_kg"]
        print(f"  {payload_kg:.1f}kg (1-step)", end="")
        print(f"  {payload_kg:.1f}kg (multi)", end="")
    print()
    print("-" * (10 + len(payloads) * 30))

    for model in models:
        label = model_labels.get(model, model)
        print(f"{label:<10}", end="")
        for p in payloads:
            payload_data = results[p]
            if model == "crba":
                key = "crba"
            else:
                key = f"{model}_ft_0"

            if key in payload_data:
                m = payload_data[key]
                print(f"  {m['total_rmse']:>13.6f}", end="")
                ms = m.get("multi_step_rmse", float("nan"))
                print(f"  {ms:>11.6f}", end="")
            else:
                print(f"  {'N/A':>13}", end="")
                print(f"  {'N/A':>11}", end="")
        print()


def analyze_finetune_sweep(results: dict) -> None:
    """Table 2: Fine-tuning sweep at target payload."""
    print_header("Table 2: Fine-tune Sweep (1-step RMSE)")

    models = ["mlp", "j_mlp", "c_mlp", "delan"]
    model_labels = {"mlp": "MLP", "j_mlp": "J-MLP", "c_mlp": "C-MLP", "delan": "DeLaN"}

    # Use the heaviest payload for fine-tune analysis
    payloads = sorted(results.keys())
    target_payload = payloads[-1]  # 1.0 kg
    payload_data = results[target_payload]
    payload_kg = payload_data["payload_kg"]

    print(f"\n  Target payload: {payload_kg:.1f} kg (trained on 0.0 kg)")

    # Find available fine-tune sizes
    ft_sizes = set()
    for key in payload_data:
        if "_ft_" in key:
            n = int(key.split("_ft_")[1])
            ft_sizes.add(n)
    ft_sizes = sorted(ft_sizes)

    # Header
    print(f"\n{'Model':<10}", end="")
    for n in ft_sizes:
        print(f"  N={n:>5}", end="")
    print()
    print("-" * (10 + len(ft_sizes) * 10))

    for model in models:
        label = model_labels.get(model, model)
        print(f"{label:<10}", end="")
        for n in ft_sizes:
            key = f"{model}_ft_{n}"
            if key in payload_data:
                rmse = payload_data[key]["total_rmse"]
                print(f"  {rmse:>7.4f}", end="")
            else:
                print(f"  {'N/A':>7}", end="")
        print()

    # CRBA baseline
    if "crba" in payload_data:
        crba_rmse = payload_data["crba"]["total_rmse"]
        print(f"{'CRBA':<10}", end="")
        for _ in ft_sizes:
            print(f"  {crba_rmse:>7.4f}", end="")
        print("  (zero-shot, no training)")


def analyze_indirect_coupling(results: dict) -> None:
    """Table 3: Indirect coupling RMSE — the smoking gun."""
    print_header("Table 3: Indirect Coupling RMSE (SMOKING GUN)")
    print("  Pairs with |J_ij| < 0.01 but C_ij > 0.05")
    print("  Classical models CANNOT see these; quantum models CAN.")

    models = ["mlp", "j_mlp", "c_mlp", "delan", "crba"]
    model_labels = {"mlp": "MLP", "j_mlp": "J-MLP", "c_mlp": "C-MLP",
                    "delan": "DeLaN", "crba": "CRBA"}

    payloads = sorted(results.keys())

    # Zero-shot comparison
    print(f"\n  --- Zero-shot (N=0) ---")
    print(f"{'Model':<10}", end="")
    for p in payloads:
        print(f"  {results[p]['payload_kg']:.1f}kg", end="")
    print()
    print("-" * (10 + len(payloads) * 10))

    for model in models:
        label = model_labels.get(model, model)
        print(f"{label:<10}", end="")
        for p in payloads:
            payload_data = results[p]
            key = "crba" if model == "crba" else f"{model}_ft_0"
            if key in payload_data:
                m = payload_data[key]
                indirect = m.get("mean_indirect_rmse", float("nan"))
                print(f"  {indirect:>7.4f}", end="")
            else:
                print(f"  {'N/A':>7}", end="")
        print()

    # Detailed per-pair analysis for target payload
    target_payload = payloads[-1]
    payload_data = results[target_payload]
    payload_kg = payload_data["payload_kg"]
    print(f"\n  --- Per-pair breakdown at {payload_kg:.1f} kg (N=0) ---")

    pairs = [(0,1),(0,2),(1,2),(1,5),(2,5),(2,6),(5,6)]
    print(f"{'Pair':<10}", end="")
    for model in models:
        print(f"  {model_labels[model]:>7}", end="")
    print()
    print("-" * (10 + len(models) * 10))

    for i, j in pairs:
        print(f"({i},{j}){'':<5}", end="")
        for model in models:
            key = "crba" if model == "crba" else f"{model}_ft_0"
            if key in payload_data:
                pair_key = f"pair_{i}_{j}_rmse"
                val = payload_data[key].get(pair_key, float("nan"))
                print(f"  {val:>7.4f}", end="")
            else:
                print(f"  {'N/A':>7}", end="")
        print()


def analyze_key_findings(results: dict) -> None:
    """Extract key numbers for paper text."""
    print_header("Key Findings for Paper")

    payloads = sorted(results.keys())

    for p in payloads:
        pd = results[p]
        payload_kg = pd["payload_kg"]

        mlp_0 = pd.get("mlp_ft_0", {})
        jmlp_0 = pd.get("j_mlp_ft_0", {})
        cmlp_0 = pd.get("c_mlp_ft_0", {})
        crba = pd.get("crba", {})

        print(f"\n  Payload = {payload_kg:.1f} kg:")

        # 1-step comparison
        if all(k in pd for k in ["mlp_ft_0", "j_mlp_ft_0", "c_mlp_ft_0"]):
            mlp_r = mlp_0["total_rmse"]
            jmlp_r = jmlp_0["total_rmse"]
            cmlp_r = cmlp_0["total_rmse"]
            print(f"    1-step RMSE:  MLP={mlp_r:.4f}  J-MLP={jmlp_r:.4f}  C-MLP={cmlp_r:.4f}")
            if cmlp_r < jmlp_r:
                improvement = (jmlp_r - cmlp_r) / jmlp_r * 100
                print(f"    C-MLP vs J-MLP: {improvement:.1f}% improvement (quantum > classical)")
            else:
                degradation = (cmlp_r - jmlp_r) / jmlp_r * 100
                print(f"    C-MLP vs J-MLP: {degradation:.1f}% WORSE (unexpected)")

        # Indirect coupling (smoking gun)
        if all(k in pd for k in ["mlp_ft_0", "j_mlp_ft_0", "c_mlp_ft_0"]):
            mlp_i = mlp_0.get("mean_indirect_rmse", float("nan"))
            jmlp_i = jmlp_0.get("mean_indirect_rmse", float("nan"))
            cmlp_i = cmlp_0.get("mean_indirect_rmse", float("nan"))
            print(f"    Indirect RMSE: MLP={mlp_i:.4f}  J-MLP={jmlp_i:.4f}  C-MLP={cmlp_i:.4f}")
            if cmlp_i < jmlp_i:
                ratio = jmlp_i / cmlp_i
                print(f"    SMOKING GUN: C-MLP {ratio:.1f}x better on indirect pairs")

        # Small-data advantage (N=50 vs N=500)
        for n_small in [50, 100]:
            cmlp_s = pd.get(f"c_mlp_ft_{n_small}", {})
            jmlp_s = pd.get(f"j_mlp_ft_{n_small}", {})
            mlp_s = pd.get(f"mlp_ft_{n_small}", {})
            if cmlp_s and jmlp_s:
                print(f"    N={n_small}: C-MLP={cmlp_s.get('total_rmse',0):.4f}  "
                      f"J-MLP={jmlp_s.get('total_rmse',0):.4f}  "
                      f"MLP={mlp_s.get('total_rmse',0):.4f}")


def generate_latex_table(results: dict) -> None:
    """Generate LaTeX table for paper."""
    print_header("LaTeX Table (copy-paste ready)")

    models = ["mlp", "j_mlp", "c_mlp", "delan", "crba"]
    model_labels = {"mlp": "MLP", "j_mlp": "$J$-MLP", "c_mlp": "$C$-MLP",
                    "delan": "DeLaN", "crba": "CRBA"}

    payloads = sorted(results.keys())

    print(r"""
\begin{table}[t]
\caption{World model transfer: 1-step RMSE (trained on 0\,kg, zero-shot transfer).
$C$-MLP uses quantum entanglement features; $J$-MLP uses classical coupling.}
\label{tab:gamma-transfer}
\centering
\begin{tabular}{l""", end="")
    for _ in payloads:
        print("cc", end="")
    print(r"""}
\toprule""")

    # Header
    print(r"Model", end="")
    for p in payloads:
        pkg = results[p]["payload_kg"]
        print(rf" & \multicolumn{{2}}{{c}}{{{pkg:.1f}\,kg}}", end="")
    print(r" \\")

    print(r"", end="")
    for _ in payloads:
        print(r" & 1-step & Indirect", end="")
    print(r" \\")
    print(r"\midrule")

    for model in models:
        label = model_labels[model]
        print(f"{label}", end="")
        for p in payloads:
            pd = results[p]
            key = "crba" if model == "crba" else f"{model}_ft_0"
            if key in pd:
                total = pd[key].get("total_rmse", float("nan"))
                indirect = pd[key].get("mean_indirect_rmse", float("nan"))
                # Bold the best
                print(f" & {total:.4f} & {indirect:.4f}", end="")
            else:
                print(r" & --- & ---", end="")
        print(r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/gamma/transfer_results.json")
    args = parser.parse_args()

    results = load_results(args.results)

    analyze_zero_shot(results)
    analyze_finetune_sweep(results)
    analyze_indirect_coupling(results)
    analyze_key_findings(results)
    generate_latex_table(results)


if __name__ == "__main__":
    main()
