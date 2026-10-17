"""Compare a refit CSV (lsq or GA, run on the Monte Carlo synthetic data)
against the KNOWN TRUE generating parameters (test/mc_ground_truth_params.csv),
merged on (cycle, kind). Correct truth/refit labeling (compare_param_values.py's
own labels are generic _lsq/_ga based on ARGUMENT ORDER, not semantics --
easy to mix up, this script names them explicitly instead)."""
import sys

import numpy as np
import pandas as pd

SET_SIDE = ["vs", "etas", "ron", "aon", "ion"]
RESET_SIDE = ["vr", "etar", "gam", "roff", "aoff", "ioff"]


def compare(truth_path, refit_path, label):
    truth = pd.read_csv(truth_path)
    refit = pd.read_csv(refit_path)
    m = truth.merge(refit, on=["cycle", "kind"], suffixes=("_truth", "_refit"))
    print(f"\n=== {label}: truth vs refit, n={len(m)} ===")
    for kind, side in [("max", SET_SIDE), ("min", RESET_SIDE)]:
        sub = m[m["kind"] == kind]
        print(f"--- {kind} half-cycles, n={len(sub)} (free side: {side}) ---")
        for p in side:
            if p == "gam":
                continue
            tv, rv = sub[f"{p}_truth"].to_numpy(), sub[f"{p}_refit"].to_numpy()
            reldiff = np.abs(tv - rv) / np.maximum(np.maximum(np.abs(tv), np.abs(rv)), 1e-12)
            corr = np.corrcoef(tv, rv)[0, 1]
            print(f"  {p:6s} truth_med={np.median(tv):10.4g} refit_med={np.median(rv):10.4g}  "
                  f"rel_diff_med={np.median(reldiff)*100:6.1f}%  rel_diff_p90={np.quantile(reldiff,0.9)*100:6.1f}%  "
                  f"corr={corr:.3f}")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
