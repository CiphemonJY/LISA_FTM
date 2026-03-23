#!/usr/bin/env python3
"""
Suite 4: Differential Privacy Sweep
Sweeps NOISE_MULTIPLIER across [0.0, 0.1, 0.5, 1.0, 2.0]
Runs eval/autora.py for each config, records perplexity, plots tradeoff.
"""
import subprocess
import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"
CONFIG_FILE = EVAL_DIR / "fed_config.py"

NOISE_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0]
ROUNDS = 5  # matches fed_config.py default


def set_noise(multiplier: float) -> None:
    """Edit fed_config.py to set NOISE_MULTIPLIER."""
    content = CONFIG_FILE.read_text()
    # Replace the NOISE_MULTIPLIER line
    new_lines = []
    for line in content.splitlines():
        if line.strip().startswith("NOISE_MULTIPLIER"):
            new_lines.append(f"NOISE_MULTIPLIER = {multiplier}")
        else:
            new_lines.append(line)
    CONFIG_FILE.write_text("\n".join(new_lines) + "\n")
    print(f"[Suite4] Set NOISE_MULTIPLIER = {multiplier}")


def run_experiment() -> dict:
    """Run autora.py run and parse final perplexity from output."""
    log_file = RESULTS_DIR / "suite4_run.log"
    err_file = RESULTS_DIR / "suite4_run.err"

    with open(log_file, "w") as lf, open(err_file, "w") as ef:
        proc = subprocess.Popen(
            [sys.executable, "eval/autora.py", "run"],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            lf.write(line)
            print(line, end="")  # real-time output
        proc.wait()
        ef.write("")

    # Parse perplexity from log
    perplexity = None
    with open(log_file) as f:
        for line in f:
            m = re.search(r"FINAL.*?\|\s*([\d.]+)", line)
            if m:
                perplexity = float(m.group(1))
            # Also look for final_perplexity in JSON output
            exp_match = re.search(r'"exp_id":\s*"exp_(\d+)"', line)
            final_ppl_match = re.search(r'"final_perplexity":\s*([\d.]+)', line)
            if final_ppl_match:
                perplexity = float(final_ppl_match.group(1))

    # Find the result JSON
    result_files = sorted(RESULTS_DIR.glob("exp_*.json"), key=lambda p: p.stat().st_mtime)
    if result_files:
        latest = result_files[-1]
        with open(latest) as f:
            data = json.load(f)
            return data

    return {"final_perplexity": perplexity, "status": "unknown"}


def main():
    print("=" * 70)
    print("SUITE 4: Differential Privacy Sweep")
    print("=" * 70)

    results = []
    for noise in NOISE_VALUES:
        print(f"\n{'='*70}")
        print(f"Running NOISE_MULTIPLIER = {noise}")
        print("=" * 70)

        set_noise(noise)
        result = run_experiment()

        ppl = result.get("final_perplexity", None)
        print(f"  Final perplexity: {ppl}")
        results.append({
            "noise_multiplier": noise,
            "final_perplexity": ppl,
            "exp_id": result.get("exp_id"),
            "timestamp": datetime.now().isoformat(),
        })

        # Reset NOISE_MULTIPLIER to 0.0 for safety
        set_noise(0.0)

    print("\n" + "=" * 70)
    print("SUITE 4 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'NOISE_MULTIPLIER':>20} | {'FINAL PERPLEXITY':>18}")
    print("-" * 45)
    for r in results:
        print(f"{r['noise_multiplier']:>20} | {r['final_perplexity']:>18.2f}")

    # Save summary
    summary = {
        "suite": "Suite 4 - Differential Privacy Sweep",
        "timestamp": datetime.now().isoformat(),
        "noise_sweep": results,
    }
    summary_path = RESULTS_DIR / "suite4_dp_sweep.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Plot using matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        noises = [r["noise_multiplier"] for r in results]
        ppls = [r["final_perplexity"] for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(noises, ppls, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel("NOISE_MULTIPLIER (ε proxy)")
        ax.set_ylabel("Final Perplexity")
        ax.set_title("Privacy-Utility Tradeoff: Differential Privacy Sweep\n(Lower perplexity = better utility)")
        ax.grid(True, alpha=0.3)
        for i, (n, p) in enumerate(zip(noises, ppls)):
            ax.annotate(f"ε≈{n}\nppl={p:.1f}", (n, p), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
        plt.tight_layout()
        plot_path = RESULTS_DIR / "suite4_dp_tradeoff.png"
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
