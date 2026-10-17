"""CLI driver for extract_fit.chain_half_cycles() (by request, 2026-10-14,
"can't you share the results and the functions between the cli and the
gui? It seems inefficient to have them duplicated"): runs the real,
genuine-least_squares 0->Vmax->0 / 0->Vmin->0 half-cycle chain over N
consecutive logical cycles and writes the result via
extract_fit.chain_results_to_dataframe() -- the SAME row-building
function analyze_rolling_fit.py's own GUI "Save chain to CSV" button
uses, so a result saved from the CLI and a result saved from the GUI
are byte-identical in shape. Replaces the throwaway scratchpad scripts
this session kept reaching for to answer one-off "check cycle N"
questions.

Example:
    python3 fit_chain_cli.py --start-cycle 2000 --n-cycles 1000 \
        --out dades/chain_2000_2999.csv
"""
import argparse
import time

from data_utils import DATA_PATH, load_data, cycle_bounds
import extract_fit as ef


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--start-cycle", type=int, required=True)
    ap.add_argument("--n-cycles", type=int, required=True)
    ap.add_argument("--ron-roff-bound-pct", type=float, default=25.0,
                     help="Dynamic +-bound on ron/roff per half-cycle step, as a percent of the "
                          "previous half-cycle's own value (RULES_DOC.md section 19/20). Default 25.")
    ap.add_argument("--margin-frac", type=float, default=0.05,
                     help="Pin-escape margin (RULES_DOC.md): if a param is within this fraction of "
                          "its GLOBAL bound's span from either edge (linear space for ron/roff/vs/vr, "
                          "log space for the other 8), it's reset before continuing. Default 0.05 (5%%).")
    ap.add_argument("--ron-roff-reset-value", type=float, default=22.0,
                     help="Fixed pin-escape reset value for ron/roff specifically (not bounds-derived "
                          "-- validated against real data). Default 22.")
    ap.add_argument("--ron-roff-upper-bound", type=float, default=500.0,
                     help="Global upper bound for ron/roff (RULES_DOC.md: 500 was the original, "
                          "possibly too permissive -- being re-checked against real data). Default 500.")
    ap.add_argument("--etas-etar-upper-bound", type=float, default=3000.0,
                     help="Global upper bound for etas/etar. DEFAULT_BOUNDS's own 3000 is wide enough "
                          "that the margin_frac pin-escape check (which only guards the literal global "
                          "edges) never fires while etas/etar wander into the hundreds -- confirmed "
                          "directly 2026-10-16 on cycles 1500-1999: etas/etar drifted up to ~1700/1480 "
                          "with low rmse throughout but catastrophic peak-current error (>80%% of "
                          "cycles off by >10%%). fit_rolling_gui15oct.py's own CHAIN_BOUNDS already "
                          "tightened this to (10,150) for the same reason -- pass e.g. 150 here for "
                          "the equivalent guard on this chain. Default 3000 (unchanged/off).")
    ap.add_argument("--max-nfev", type=int, default=2000)
    ap.add_argument("--fix-gam", type=float, default=None,
                     help="Hold gam fixed at this value for the WHOLE chain instead of letting "
                          "RESET_SIDE fit it freely alongside etar. Added 2026-10-17 after etas/etar "
                          "pinning was found to be severe (48.8%%/10.8%% of half-cycles at bound=150) "
                          "-- root-caused to memory memristor_model_dmm_relations_restrictions's "
                          "already-documented gam/etar RIDGE degeneracy (freeing them jointly, as this "
                          "chain's RESET_SIDE does, is exactly what reproduces it) -- that memory's own "
                          "recommendation is gam=0.22 (not fit freely). Confirmed: fixing it cuts etar "
                          "pinning from 39.2%% to 10.8%% with no peak-current accuracy cost. Pass 0.22 "
                          "for the project's own established default. Default None (fit gam freely, "
                          "unchanged/old behavior).")
    ap.add_argument("--rmse-spike-factor", type=float, default=3.0,
                     help="--method lsq only: if a half-cycle's own fit rmse exceeds this many "
                          "times the median of the last --rmse-baseline-window half-cycles' rmse, "
                          "retry it once from its own free-side params reset to _geo_reset_value() "
                          "instead of the carried-forward x0, keeping whichever attempt fits better. "
                          "Added 2026-10-16 after a full-16602-cycle run found a single half-cycle's "
                          "fit can land vs/vr just past the window's own applied voltage range -- a "
                          "zero-gradient dead zone the optimizer can't locally escape -- which then "
                          "froze vs/vr bit-for-bit for the rest of the chain (69%% of that run). Pass "
                          "a very large value (or see code) to effectively disable. Default 3.0.")
    ap.add_argument("--rmse-spike-min-history", type=int, default=20,
                     help="--method lsq only: half-cycles of rmse history required before the "
                          "spike check activates. Default 20.")
    ap.add_argument("--rmse-baseline-window", type=int, default=50,
                     help="--method lsq only: how many of the most recent half-cycles' rmse to take "
                          "the median of as the local baseline for the spike check. Default 50.")
    ap.add_argument("--method", choices=["lsq", "ga", "directed_ga", "ga_lsq"], default="lsq",
                     help="Per-half-cycle fit method: 'lsq' (default) is the gradient-based joint "
                          "fit_region_pair() least_squares; 'ga' is fit_region_pair_ga() "
                          "(differential_evolution), with vs/vr locked near their own direct-"
                          "extraction value (see --ga-lock-margin-v). 'directed_ga' is "
                          "fit_region_pair_directed_ga() (added 2026-10-17 after a Monte Carlo "
                          "ground-truth recovery test found neither lsq nor plain ga reliably "
                          "recovers the true parameters even on noise-free synthetic data, see "
                          "memory memristor_model_chain_damping.md): a GA whose mutations are drawn "
                          "from a PCA of the local Jacobian, weighted toward the stiffest "
                          "directions, instead of isotropically. 'ga_lsq' is "
                          "fit_region_pair_ga_seeded_lsq() (added 2026-10-17, a \"desperate test\": "
                          "GA generates a population, --ga-lsq-top-frac of it each gets its own "
                          "local least_squares polish, best polished result wins) -- the most "
                          "expensive of the three, strictly more work than 'ga' alone. See each "
                          "function's own docstring for the full design rationale.")
    ap.add_argument("--ga-popsize", type=int, default=12)
    ap.add_argument("--ga-maxiter", type=int, default=100)
    ap.add_argument("--ga-lock-margin-v", type=float, default=0.05,
                     help="--method ga/directed_ga only: +-volts around the direct vs/vr extraction "
                          "that the search is allowed to range over, intersected with the ordinary "
                          "bounds. Default 0.05.")
    ap.add_argument("--ga-seed", type=int, default=None)
    ap.add_argument("--dga-sigma0", type=float, default=0.25,
                     help="--method directed_ga only: base mutation step size, as a fraction of a "
                          "typical free parameter's own bound span, along the TOP (largest-"
                          "eigenvalue) Jacobian-PCA direction -- lesser directions get "
                          "proportionally smaller steps (see fit_region_pair_directed_ga()'s own "
                          "docstring). Default 0.25.")
    ap.add_argument("--dga-random-frac", type=float, default=0.3,
                     help="--method directed_ga only: fraction of each new generation drawn as a "
                          "plain uniform-random individual instead of a Jacobian-directed one -- "
                          "keeps some isotropic global-exploration budget. Default 0.3.")
    ap.add_argument("--dga-jac-refresh-every", type=int, default=1,
                     help="--method directed_ga only: recompute the Jacobian/PCA every this many "
                          "generations (cost: 2*n_free extra simulate() calls per refresh). "
                          "Default 1 (every generation).")
    ap.add_argument("--ga-lsq-top-frac", type=float, default=1.0,
                     help="--method ga_lsq only: fraction of the GA's final population (lowest "
                          "DE energy first) to lsq-polish. Default 1.0 (the whole population, "
                          "matching the request literally) -- COST WARNING: this runs "
                          "popsize*n_free separate least_squares calls per half-cycle on top of "
                          "the GA phase itself; lower this for a cheaper (but less thorough) run.")
    ap.add_argument("--ga-lsq-polish-max-nfev", type=int, default=200,
                     help="--method ga_lsq only: max_nfev for each individual population member's "
                          "own least_squares polish. Default 200.")
    ap.add_argument("--ga-workers", type=int, default=1,
                     help="--method ga only: differential_evolution's own workers= (multiprocessing). "
                          "Default 1 (serial) -- NOT recommended to change: benchmarked 2026-10-16 on "
                          "real data (popsize=15/maxiter=150), workers=-1 (all cores) was only ~20%% "
                          "faster than workers=1 (3.98s vs 4.85s per half-cycle), and workers=4 was "
                          "dramatically WORSE (38s) -- each GA call's own population/runtime is too "
                          "small for multiprocessing's per-call pool spawn/teardown overhead to pay "
                          "for itself, since differential_evolution creates a fresh pool every call "
                          "(and the chain itself can't be parallelized across half-cycles either, "
                          "since each one's fit depends on the previous one's fitted params). Exposed "
                          "for future experimentation (e.g. a much larger popsize/maxiter) only.")
    ap.add_argument("--ga-lock-names", nargs="*", default=["vs"], choices=["vs", "vr"],
                     help="--method ga only: which of vs/vr get the direct-extraction lock. "
                          "Default: vs only -- NOT vr, whose own apparent-threshold estimate was "
                          "found (2026-10-16) to be off by ~0.9V from the genuinely-fitted value on "
                          "real data, consistent with memory memristor_model_dmm_relations_"
                          "restrictions's own earlier synthetic-ground-truth finding that apparent "
                          "vr correlates only weakly with true vr. Pass --ga-lock-names vs vr to "
                          "restore locking both (not recommended).")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--out", default=None, help="CSV output path. Default: "
                     "dades/chain_<start>_<start+n-1>.csv")
    args = ap.parse_args()

    curr, cyc, time_, volt = load_data(args.data)
    bounds = cycle_bounds(cyc)
    cycles = sorted(bounds)
    if args.start_cycle not in bounds:
        raise SystemExit(f"cycle {args.start_cycle} not present in {args.data}")

    out_path = args.out or f"dades/chain_{args.start_cycle}_{args.start_cycle + args.n_cycles - 1}.csv"
    print(f"chaining {args.n_cycles} logical cycles from {args.start_cycle}, method={args.method}, "
          f"ron/roff bound=+-{args.ron_roff_bound_pct}%, margin_frac={args.margin_frac}, "
          f"ron/roff upper bound={args.ron_roff_upper_bound}, ron/roff reset={args.ron_roff_reset_value}, "
          f"etas/etar upper bound={args.etas_etar_upper_bound} -> {out_path}")
    ga_kwargs = dict(popsize=args.ga_popsize, maxiter=args.ga_maxiter,
                      lock_margin_v=args.ga_lock_margin_v, seed=args.ga_seed,
                      lock_names=tuple(args.ga_lock_names), workers=args.ga_workers)
    dga_kwargs = dict(popsize=args.ga_popsize, maxiter=args.ga_maxiter,
                       lock_margin_v=args.ga_lock_margin_v, seed=args.ga_seed,
                       lock_names=tuple(args.ga_lock_names),
                       sigma0=args.dga_sigma0, random_frac=args.dga_random_frac,
                       jac_refresh_every=args.dga_jac_refresh_every)
    ga_lsq_kwargs = dict(popsize=args.ga_popsize, maxiter=args.ga_maxiter,
                          lock_margin_v=args.ga_lock_margin_v, seed=args.ga_seed,
                          lock_names=tuple(args.ga_lock_names),
                          top_frac=args.ga_lsq_top_frac, polish_max_nfev=args.ga_lsq_polish_max_nfev)

    t0 = time.time()
    # Incremental flush (by request, 2026-10-14, "remember to flush after
    # writing so we can monitor in real time" -- the first version of this
    # CLI only wrote out_path once, at the very end, so a long run gave a
    # live GUI "Refresh data" nothing to see for its whole duration; this
    # mirrors fit_all_cycles.py's own buffer-then-append-with-header-once
    # pattern): every progress_cb call (gated to args.progress_every, same
    # as the old print-only cadence), the half-cycles accumulated since the
    # last flush are appended to out_path.
    header_written = [False]
    last_flushed = [0]

    def _progress(ci, n, cyc, results):
        if ci % args.progress_every != 0:
            return
        elapsed = time.time() - t0
        rate = len(results) / elapsed if elapsed > 0 else 0.0
        print(f"chain_half_cycles: {ci}/{n} logical cycles (cycle {cyc}), "
              f"{len(results)} half-cycles so far ({elapsed:.0f}s, {rate:.1f} half-cycles/s)", flush=True)
        new_rows = results[last_flushed[0]:]
        if new_rows:
            ef.chain_results_to_dataframe(new_rows).to_csv(
                out_path, mode="a" if header_written[0] else "w", header=not header_written[0], index=False)
            header_written[0] = True
            last_flushed[0] = len(results)

    results = ef.chain_half_cycles(time_, volt, curr, bounds, cycles, args.start_cycle, args.n_cycles,
                                    max_nfev=args.max_nfev,
                                    bounds_dict={"ron": (1.0, args.ron_roff_upper_bound),
                                                 "roff": (1.0, args.ron_roff_upper_bound),
                                                 "etas": (0.5, args.etas_etar_upper_bound),
                                                 "etar": (0.5, args.etas_etar_upper_bound)},
                                    ron_roff_max_change_frac=args.ron_roff_bound_pct / 100.0,
                                    ron_roff_reset_value=args.ron_roff_reset_value,
                                    margin_frac=args.margin_frac,
                                    rmse_spike_factor=args.rmse_spike_factor,
                                    rmse_spike_min_history=args.rmse_spike_min_history,
                                    rmse_baseline_window=args.rmse_baseline_window,
                                    fix_gam=args.fix_gam,
                                    method=args.method, ga_kwargs=ga_kwargs, dga_kwargs=dga_kwargs,
                                    ga_lsq_kwargs=ga_lsq_kwargs,
                                    progress_cb=_progress)
    elapsed = time.time() - t0
    new_rows = results[last_flushed[0]:]
    if new_rows:
        ef.chain_results_to_dataframe(new_rows).to_csv(
            out_path, mode="a" if header_written[0] else "w", header=not header_written[0], index=False)
    df = ef.chain_results_to_dataframe(results)
    print(f"DONE: {len(df)} half-cycles in {elapsed:.1f}s ({len(df) / elapsed:.1f} half-cycles/s) "
          f"-> {out_path}")
    print(f"rmse: median={df['rmse'].median():.4g}  min={df['rmse'].min():.4g}  max={df['rmse'].max():.4g}")


if __name__ == "__main__":
    main()
