"""Monte Carlo ground-truth parameter-recovery test: takes an already-
fitted chain CSV (as produced by fit_chain_cli.py) as a set of KNOWN TRUE
parameters, forward-simulates noise-free synthetic current from them over
each half-cycle's own real (t, V) excitation, re-fits the synthetic
current with chain_half_cycles(), and writes out both the truth and the
refit so they can be compared (see compare_vs_truth.py). This is exactly
the procedure behind the paper's Section 5 (Monte Carlo ground-truth
comparison) and the pooled-fitting test of Section 6 -- the standard way
this project answers "is a method's own disagreement with another method,
or with the truth, actual mis-estimation, or just that the two land in
different but equally-valid points along a flat direction?".

Usage:
    python3 mc_param_recovery.py --source-chain demo_data/demo_chain_lsq.csv \\
        --data demo_data/dades_demo.mat --n-rows 20 --method lsq \\
        --out-dir mc_demo

Writes <out-dir>/mc_ground_truth_params.csv (the source rows, taken as
truth) and <out-dir>/mc_refit_<method>.csv/.log (the blind refit)."""
import argparse
import os
import time

import numpy as np
import pandas as pd

import dmm_model as dmm
import extract_fit as ef
from data_utils import DATA_PATH, load_data, cycle_bounds


def build_synthetic(curr_real, time_, volt, truth_df):
    curr_synth = curr_real.copy()
    for _, row in truth_df.iterrows():
        i_lo, i_hi = int(row["i_lo"]), int(row["i_hi"])
        t_w, V_w = time_[i_lo:i_hi + 1], volt[i_lo:i_hi + 1]
        params = {name: float(row[name]) for name in dmm.PARAM_NAMES}
        with np.errstate(over="ignore"):
            I_synth, _, _ = dmm.simulate(t_w, V_w, dmm.pack(params), g0_override=float(row["g0"]))
        curr_synth[i_lo:i_hi + 1] = I_synth
    return curr_synth


def run_chain(method, curr_synth, time_, volt, bounds, cycles, start_cycle, n_cycles,
              log_path, out_csv, fix_gam, etas_etar_bound, extra_kwargs=None):
    """Flushes out_csv/log_path incrementally as the chain progresses, so
    a slow run's progress can be checked mid-flight (same convention as
    fit_chain_cli.py's own header_written/last_flushed pattern)."""
    t0 = time.time()
    header_written = [False]
    last_flushed = [0]

    def _progress(ci, n, cyc, results):
        if ci % 20 != 0:
            return
        elapsed = time.time() - t0
        rate = len(results) / elapsed if elapsed > 0 else 0.0
        line = (f"chain_half_cycles: {ci}/{n} logical cycles (cycle {cyc}), "
                f"{len(results)} half-cycles so far ({elapsed:.0f}s, {rate:.2f} half-cycles/s)")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        new_rows = results[last_flushed[0]:]
        if new_rows:
            ef.chain_results_to_dataframe(new_rows).to_csv(
                out_csv, mode="a" if header_written[0] else "w", header=not header_written[0], index=False)
            header_written[0] = True
            last_flushed[0] = len(results)

    header_line = (f"=== method={method} start_cycle={start_cycle} n_cycles={n_cycles} "
                   f"fix_gam={fix_gam} etas_etar_bound={etas_etar_bound} ===")
    print(header_line, flush=True)
    with open(log_path, "w") as f:
        f.write(header_line + "\n")

    results = ef.chain_half_cycles(
        time_, volt, curr_synth, bounds, cycles, start_cycle, n_cycles,
        bounds_dict={"ron": (1.0, 500.0), "roff": (1.0, 500.0),
                     "etas": (0.5, etas_etar_bound), "etar": (0.5, etas_etar_bound)},
        fix_gam=fix_gam, method=method,
        ga_kwargs=extra_kwargs if method == "ga" else None,
        dga_kwargs=extra_kwargs if method == "directed_ga" else None,
        ga_lsq_kwargs=extra_kwargs if method == "ga_lsq" else None,
        progress_cb=_progress)
    elapsed = time.time() - t0
    new_rows = results[last_flushed[0]:]
    if new_rows:
        ef.chain_results_to_dataframe(new_rows).to_csv(
            out_csv, mode="a" if header_written[0] else "w", header=not header_written[0], index=False)
    df = ef.chain_results_to_dataframe(results)
    tail_line = (f"DONE: {len(df)} half-cycles in {elapsed:.1f}s ({len(df)/elapsed:.2f} half-cycles/s) -> {out_csv}\n"
                 f"rmse: median={df['rmse'].median():.4g} min={df['rmse'].min():.4g} max={df['rmse'].max():.4g}")
    print(tail_line, flush=True)
    with open(log_path, "a") as f:
        f.write(tail_line + "\n")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--source-chain", required=True,
                    help="a chain CSV (fit_chain_cli.py output) whose rows are taken as KNOWN TRUE params")
    ap.add_argument("--n-rows", type=int, default=500, help="how many of --source-chain's own rows to use")
    ap.add_argument("--method", default="lsq", choices=["lsq", "ga", "directed_ga", "ga_lsq"])
    ap.add_argument("--fix-gam", type=float, default=0.22)
    ap.add_argument("--etas-etar-upper-bound", type=float, default=300.0)
    ap.add_argument("--out-dir", default="test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    curr, cyc, time_, volt = load_data(args.data)
    bounds = cycle_bounds(cyc)
    cycles = sorted(bounds)

    source = pd.read_csv(args.source_chain).sort_values("i_lo").reset_index(drop=True)
    truth_df = source.iloc[:args.n_rows].copy()
    start_cycle = int(truth_df["cycle"].min())
    n_cycles = truth_df["cycle"].nunique()
    print(f"ground truth: {len(truth_df)} rows, cycles {truth_df['cycle'].min()}-{truth_df['cycle'].max()} "
          f"({n_cycles} logical cycles)")

    truth_path = os.path.join(args.out_dir, "mc_ground_truth_params.csv")
    truth_df.to_csv(truth_path, index=False)
    print(f"wrote {truth_path}")

    curr_synth = build_synthetic(curr, time_, volt, truth_df)

    run_chain(args.method, curr_synth, time_, volt, bounds, cycles, start_cycle, n_cycles,
              os.path.join(args.out_dir, f"mc_refit_{args.method}.log"),
              os.path.join(args.out_dir, f"mc_refit_{args.method}.csv"),
              args.fix_gam, args.etas_etar_upper_bound)


if __name__ == "__main__":
    main()
