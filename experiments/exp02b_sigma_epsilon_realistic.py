"""Experiment 02b: Sigma-to-epsilon table with realistic parameters.

Addresses editorial concern: original table used unrealistic sample_rate=2e-5.
This version adds realistic FL/DP-SGD parameters.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import json
import numpy as np
from pathlib import Path
from common import sigma_to_epsilon, epsilon_to_sigma

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    sigmas = [0.1, 0.25, 0.5, 0.63, 1.0, 2.0, 5.0, 10.0]
    epsilons = [0.1, 1.0, 8.0, 100.0]
    deltas = [1e-5]

    # Realistic sample rates for different deployment scenarios
    scenarios = {
        "single_step": {"sample_rate": 1.0, "steps": [1], "desc": "Single gradient observation"},
        "small_dataset": {"sample_rate": 0.01, "steps": [100, 1000], "desc": "B=500, |D|=50K"},
        "large_dataset": {"sample_rate": 0.001, "steps": [1000, 10000], "desc": "B=1000, |D|=1M"},
        "fl_round": {"sample_rate": 0.1, "steps": [10, 100], "desc": "FL: 10% client sampling"},
    }

    results = {"scenarios": {}}

    print("=" * 70)
    print("Sigma-to-Epsilon Conversion (Realistic Parameters)")
    print("=" * 70)

    for scenario_name, params in scenarios.items():
        q = params["sample_rate"]
        steps_list = params["steps"]
        print(f"\n--- {params['desc']} (q={q}) ---")

        s2e = []
        for sigma in sigmas:
            entry = {"sigma": sigma}
            for T in steps_list:
                try:
                    eps = sigma_to_epsilon(sigma, q, T, delta=1e-5)
                    entry[f"eps_T{T}"] = round(float(eps), 4)
                except Exception as e:
                    entry[f"eps_T{T}"] = f"error: {e}"
            s2e.append(entry)
            vals = " | ".join(f"T={T}: ε={entry.get(f'eps_T{T}', 'N/A')}" for T in steps_list)
            print(f"  σ={sigma:5.2f} → {vals}")

        e2s = []
        for eps in epsilons:
            entry = {"epsilon": eps}
            for T in steps_list:
                try:
                    sig = epsilon_to_sigma(eps, q, T, delta=1e-5)
                    entry[f"sigma_T{T}"] = round(float(sig), 4)
                except Exception as e:
                    entry[f"sigma_T{T}"] = f"error: {e}"
            e2s.append(entry)
            vals = " | ".join(f"T={T}: σ={entry.get(f'sigma_T{T}', 'N/A')}" for T in steps_list)
            print(f"  ε={eps:5.1f} → {vals}")

        results["scenarios"][scenario_name] = {
            "description": params["desc"],
            "sample_rate": q,
            "steps": steps_list,
            "delta": 1e-5,
            "sigma_to_epsilon": s2e,
            "epsilon_to_sigma": e2s,
        }

    # Practitioner summary table: for common epsilon targets, what sigma and MSE?
    norm_sq = 1.01**2
    print("\n" + "=" * 70)
    print("PRACTITIONER SUMMARY (single step, q=1.0)")
    print("=" * 70)
    print(f"{'ε':>6} | {'σ':>8} | {'MSE (N=3072)':>12} | {'PSNR (dB)':>10} | {'Quality':>20}")
    print("-" * 70)
    for eps in [0.1, 0.5, 1.0, 4.0, 8.0, 50.0, 100.0]:
        try:
            sig = epsilon_to_sigma(eps, 1.0, 1, delta=1e-5)
            mse = sig**2 * norm_sq
            psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
            if mse > 10:
                quality = "Pure noise"
            elif mse > 1:
                quality = "Unrecognizable"
            elif mse > 0.1:
                quality = "Heavily degraded"
            elif mse > 0.01:
                quality = "Some structure"
            elif mse > 0.001:
                quality = "Good quality"
            else:
                quality = "Near-perfect"
            print(f"{eps:6.1f} | {sig:8.4f} | {mse:12.4f} | {psnr:10.1f} | {quality:>20}")
        except Exception:
            print(f"{eps:6.1f} | {'error':>8}")

    with open(OUTPUT_DIR / "sigma_epsilon_realistic.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/sigma_epsilon_realistic.json")


if __name__ == "__main__":
    main()
