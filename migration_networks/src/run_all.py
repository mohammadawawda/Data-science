"""Run the entire pipeline in order. Total runtime ~25-35 minutes,
dominated by the null models (700 Infomap runs) and the PPML fits.

Stages:
  0 prep_data         derive processed data files from raw downloads
  1 run_communities   Infomap + Louvain per period, validation vs blocs
  2 run_null          null models A (degree-preserving rewiring) and B (weight-shuffle)
  3 run_hubs          PageRank / strengths / betweenness + robustness
  4 run_gravity       PPML gravity, holdout, residuals, clustered SEs
  5 run_conflict      UCDP war flags x gravity residuals
  6 run_sensitivity   alternative-estimator check (sd_drop_neg)
  7 run_figures       all seven report figures
"""
from pathlib import Path
import subprocess, sys, time

STAGES = ["prep_data", "run_communities", "run_null", "run_hubs",
          "run_gravity", "run_conflict", "run_sensitivity", "run_figures"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"


def main():
    for s in STAGES:
        t = time.time()
        print(f"\n===== {s} =====")
        r = subprocess.run([sys.executable, str(SRC / f"{s}.py")],
                           cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            sys.exit(f"{s} failed")
        print(f"[{s} done in {time.time()-t:.0f}s]")
    print("\nAll stages complete. Results in out/results, figures in out/figures.")


if __name__ == "__main__":
    main()
