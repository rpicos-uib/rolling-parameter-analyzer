# rolling-parameter-analyzer

Code for extracting, validating, and inspecting the twelve parameters of
the Dynamic Memdiode Model (DMM, Miranda et al., *IEEE Electron Device
Letters* 47(6), 2026) from chained ReRAM switching measurements. It is
the companion repository for

> R. Picos, C. de Benito, O. Camps, "On the Inaccuracy of Global
> Optimization Methods to Extract ReRAM Parameters" (in preparation).
> Source and a compiled PDF are in [`paper/`](paper/main.pdf).

Three things live here: the **fitting CLI** (`fit_chain_cli.py`, four of
the paper's five extraction methods), the **Monte Carlo ground-truth
generator** (`mc_param_recovery.py` + `compare_vs_truth.py`, the test
behind the paper's Section 5), and the **viewer**
(`analyze_rolling_fit.py`, described below) used to inspect the result of
either one. A small synthetic demo dataset (`demo_data/`) is included so
every command below runs as-is, with no dependency on the paper's own
real, unpublished measurement data — see
[Demo dataset vs. the paper's real data](#demo-dataset-vs-the-papers-real-data).

## Installing

```bash
pip install -r requirements.txt
```

## Fitting methods

```bash
python3 fit_chain_cli.py --help
```

runs `extract_fit.chain_half_cycles()` over N consecutive logical cycles
of a dataset: each half-cycle (`0→Vmax→0` or `0→Vmin→0`) gets a joint
least-squares (or population-search) fit of its own free side, the other
side held fixed at its last fitted value, with the internal state `g`
carried forward exactly — see the paper's Section 3 for the full
rationale and the three safeguards (`--fix-gam`, `--etas-etar-upper-bound`,
and the built-in rmse-spike retry) that make it safe to run unattended
over a whole dataset. `--method` selects which of four strategies solves
each half-cycle's own local optimization problem:

```
--method {lsq,ga,directed_ga,ga_lsq}
```

(`lsq`: gradient-based least squares, the default and the fastest;
`ga`: differential evolution with a direct-extraction lock on the
free-side threshold; `directed_ga`: the same population search steered by
the local Jacobian's principal directions, after
García-Moreno/Íñiguez/Picos, ICSICT 2008; `ga_lsq`: every member of a GA's
final population individually polished by a local least-squares step,
keeping the best). The paper's fifth method, a CNN-based surrogate, was a
single exploratory pass (no architecture search, no cross-validation —
see the paper's own caveats in Sections 3 and 5) and its research-stage
code is not included in this release.

Example, run against the bundled demo dataset:

```bash
python3 fit_chain_cli.py --data demo_data/dades_demo.mat \
    --start-cycle 1 --n-cycles 20 --method lsq \
    --fix-gam 0.22 --etas-etar-upper-bound 300 \
    --out demo_chain.csv
```

```
chaining 20 logical cycles from 1, method=lsq, ron/roff bound=+-25.0%, margin_frac=0.05, ron/roff upper bound=500.0, ron/roff reset=22.0, etas/etar upper bound=300.0 -> demo_chain.csv
chain_half_cycles: 0/20 logical cycles (cycle 1), 0 half-cycles so far (0s, 0.0 half-cycles/s)
DONE: 40 half-cycles in 4.9s (8.2 half-cycles/s) -> demo_chain.csv
rmse: median=4.761e-05  min=3.449e-13  max=0.00012
```

`demo_chain.csv` is in the same format `analyze_rolling_fit.py` reads
(below) and the same format `mc_param_recovery.py` consumes as a source
of "known true" parameters.

## Monte Carlo ground-truth validation

This is the test behind the paper's central finding: fitting a model
well (low current-reconstruction error) and recovering its true
parameters are not the same claim, and the only way to tell them apart
is to fit something whose generating parameters are actually known.

`mc_param_recovery.py` takes an already-fitted chain CSV (such as the one
above), treats its rows as *known true* parameters, forward-simulates
noise-free synthetic current from them over each half-cycle's own real
`(t, V)` excitation, and re-fits that synthetic current blind to the
parameters that generated it:

```bash
python3 mc_param_recovery.py --help
python3 mc_param_recovery.py --source-chain demo_chain.csv \
    --data demo_data/dades_demo.mat --n-rows 20 --method lsq \
    --out-dir mc_demo
```

```
ground truth: 20 rows, cycles 1-10 (10 logical cycles)
wrote mc_demo/mc_ground_truth_params.csv
=== method=lsq start_cycle=1 n_cycles=10 fix_gam=0.22 etas_etar_bound=300.0 ===
chain_half_cycles: 0/10 logical cycles (cycle 1), 0 half-cycles so far (0s, 0.00 half-cycles/s)
DONE: 20 half-cycles in 9.9s (2.01 half-cycles/s) -> mc_demo/mc_refit_lsq.csv
rmse: median=1.092e-08 min=3.679e-15 max=1.324e-06
```

`compare_vs_truth.py` then merges the truth and the refit on
`(cycle, kind)` and reports, per free parameter, the median/p90 relative
error and the correlation with the known truth — the numbers behind
Table 1 and Figure 5 of the paper:

```bash
python3 compare_vs_truth.py mc_demo/mc_ground_truth_params.csv mc_demo/mc_refit_lsq.csv lsq
```

```
=== lsq: truth vs refit, n=20 ===
--- max half-cycles, n=10 (free side: ['vs', 'etas', 'ron', 'aon', 'ion']) ---
  vs     truth_med=      1.36 refit_med=      1.24  rel_diff_med=   0.5%  rel_diff_p90=   8.8%  corr=0.985
  etas   truth_med=     52.97 refit_med=     42.04  rel_diff_med=   1.5%  rel_diff_p90=  61.4%  corr=0.723
  ron    truth_med=     20.83 refit_med=     21.05  rel_diff_med=   0.7%  rel_diff_p90=   9.0%  corr=0.983
  aon    truth_med=        10 refit_med=        10  rel_diff_med=   0.0%  rel_diff_p90=  12.6%  corr=0.996
  ion    truth_med=     15.84 refit_med=     15.83  rel_diff_med=   0.1%  rel_diff_p90=   0.1%  corr=1.000
--- min half-cycles, n=10 (free side: ['vr', 'etar', 'gam', 'roff', 'aoff', 'ioff']) ---
  vr     truth_med=    -1.064 refit_med=   -0.4107  rel_diff_med=   0.3%  rel_diff_p90=  81.1%  corr=0.578
  etar   truth_med=     19.52 refit_med=     31.19  rel_diff_med=   0.4%  rel_diff_p90=  56.2%  corr=0.586
  roff   truth_med=     34.18 refit_med=     34.17  rel_diff_med=   0.0%  rel_diff_p90=   0.0%  corr=1.000
  aoff   truth_med=     1.995 refit_med=     1.993  rel_diff_med=   0.1%  rel_diff_p90=   0.1%  corr=1.000
  ioff   truth_med=     1e-07 refit_med=     1e-07  rel_diff_med=   0.0%  rel_diff_p90=   9.3%  corr=1.000
```

Note the `aon` row: the *true* value here is whatever `fit_chain_cli.py`
itself landed on when it produced `demo_chain.csv` — and it landed
exactly on `aon`'s own upper bound (10), on a demo dataset whose actual
generating `aon` never moves from the DMM's NOMINAL card (4.1). Even at
this tiny, 10-cycle, mostly noise-free scale, that is a live instance of
the paper's own central finding (Sections 5-6): `aon` is one of the
parameters a single fixed excitation waveform does not pin down, and an
optimizer pinned against its own box bound is the textbook signature of
why.

Re-running with `--method ga` (same source, same ground truth) and
comparing both against truth side by side is how Table 1 and Figure 5
were built, just at full scale (500 half-cycles, five methods) instead of
this 20-row demo.

## The viewer

`analyze_rolling_fit.py` reads a chained-fit CSV (one row per half-cycle,
with the twelve DMM parameters, an `rmse` column, and `cycle`/`kind`
identifiers) and opens a six-tab viewer:

1. **Evolution vs cycle** — any subset of the twelve parameters, plus the
   fit `rmse`, each independently rescaled to `[-1, 1]` so parameters that
   live on very different scales (e.g. `ioff ~ 1e-4` against `ron ~ 50`)
   can be compared on one shared axis against cycle number. A cycle
   slider opens a popup overlaying the real measured curve against the
   model's own forward simulation for that cycle and its neighbours.
2. **Parameter vs parameter** — scatter of any two parameters, same cycle
   or lag-1 (previous cycle), with a regression line and its R².
3. **Distribution** — a histogram of one parameter (or `rmse`) with any
   combination of Gaussian / Lognormal / Gamma / Exponential / Logistic /
   Skew-normal / Weibull fits (plain or as a k-component mixture)
   overlaid.
4. **Parameter table** — a live, scrollable table of all twelve
   parameters (rows) against cycles (columns), centred on a chosen cycle.
5. **Curve inspector** — actual-vs-model I(t) and I-V for a chosen cycle,
   correctly reconstructed for chain-shaped CSVs (two half-cycle rows per
   cycle, with the internal state `g` carried continuously across them,
   not re-simulated from a single merged parameter set).
6. **Fuzzy extractor** — runs the project's own interpretable,
   per-parameter iterative extraction procedure live on one chosen
   cycle, as an alternative, step-by-step view alongside the joint
   least-squares fit.

```bash
python3 analyze_rolling_fit.py --csv demo_chain.csv
```

Data is read from disk on demand ("Refresh data"), so the tool doubles as
a live monitor for a still-running background extraction job.

## Full example, start to finish

Everything above, chained together on a fresh clone:

```bash
git clone https://github.com/rpicos-uib/rolling-parameter-analyzer.git
cd rolling-parameter-analyzer
pip install -r requirements.txt

# 1. generate the small synthetic demo dataset (skip this -- it's
#    already committed at demo_data/dades_demo.mat -- unless you want
#    to regenerate it, e.g. with a different seed or cycle count)
python3 generate_demo_dataset.py   # writes demo_data/dades_demo.mat

# 2. fit it: four methods are available, lsq is the fast default
python3 fit_chain_cli.py --data demo_data/dades_demo.mat \
    --start-cycle 1 --n-cycles 20 --method lsq \
    --fix-gam 0.22 --etas-etar-upper-bound 300 --out demo_chain.csv

# 3. validate: is the fit actually recovering the right parameters, or
#    just reproducing the current? Build a noise-free synthetic ground
#    truth from step 2's own output, re-fit it blind, and compare.
python3 mc_param_recovery.py --source-chain demo_chain.csv \
    --data demo_data/dades_demo.mat --n-rows 20 --method lsq \
    --out-dir mc_demo
python3 compare_vs_truth.py mc_demo/mc_ground_truth_params.csv \
    mc_demo/mc_refit_lsq.csv lsq

# 4. inspect step 2's own result cycle by cycle
python3 analyze_rolling_fit.py --csv demo_chain.csv
```

Steps 2-4 above reproduce exactly the commands and real console output
shown in the sections above (step 1 is already committed so the other
steps are reproducible without a random regeneration step in between).
On the paper's own real, full-scale dataset (14604 cycles, 500-half-cycle
Monte Carlo test), the same four commands are what produced Sections 3
and 5 of the paper.

## Demo dataset vs. the paper's real data

`demo_data/dades_demo.mat` (80 cycles, generated by
`generate_demo_dataset.py`) is a **synthetic** dataset: a DMM forward
simulation from the model's own NOMINAL parameter card, with the SET/RESET
free-side parameters (Table 1 of the paper) random-walked a little cycle
to cycle so the chain fit has something non-trivial to recover. It exists
purely so every command on this page can be run by anyone who clones the
repository, with no dependency on real measurement hardware or data.

**It is not the dataset the paper's own quantitative results are based
on.** Those come from a real TiN/Ti/HfO$_2$/W device, from the same
measurement line described in García et al., "Effect of Temperature on
the Multilevel Properties and Set and Reset Transitions in HfO2-Based
Resistive Switching Devices," 14th Spanish Conference on Electron Devices
(CDE), Valencia, 2023 — that real dataset is not included in this
repository.

## Input format (chain CSV)

A CSV with one row per half-cycle switching event, columns:

```
cycle, kind, i_lo, i_peak, i_hi, rmse, retried,
ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam, g0
```

`kind` is `max` for a 0→V_max→0 (SET) half-cycle or `min` for a
0→V_min→0 (RESET) half-cycle. `i_lo`/`i_peak`/`i_hi` are sample indices
into the raw (t, V, I) trace this row's window spans. This is exactly
the format `fit_chain_cli.py` writes and `analyze_rolling_fit.py`/
`mc_param_recovery.py` read.

## Files

- `fit_chain_cli.py` — the fitting CLI (lsq / ga / directed_ga / ga_lsq).
- `mc_param_recovery.py` — the Monte Carlo ground-truth generator.
- `compare_vs_truth.py` — truth-vs-refit comparison (correlation, relative error).
- `analyze_rolling_fit.py` — the viewer.
- `generate_demo_dataset.py` — builds `demo_data/dades_demo.mat`.
- `demo_data/` — the synthetic demo dataset and an example chain CSV.
- `dmm_model.py` — the DMM forward model (`simulate()`), Numba-jitted.
- `fit_region_pair.py` — the joint least-squares / GA / directed-GA /
  GA-seeded-lsq fitting routines, and the near-V=0/curvature-based direct
  extraction relations described in the paper's Section 4.
- `extract_fit.py` — half-cycle window detection and
  `chain_half_cycles()`, the engine behind `fit_chain_cli.py` and
  `mc_param_recovery.py` alike.
- `fuzzy_fit.py` — the interpretable, per-parameter iterative extraction
  procedure used by the viewer's Fuzzy Extractor tab.
- `data_utils.py` — raw-data loading and the default parameter-bounds
  table shared by every fitting routine.
- `fit_all_cycles.py` — `SET_SIDE`/`RESET_SIDE` parameter groupings and
  a per-cycle batch-fitting routine the viewer's data model mirrors.

## License

MIT (see `LICENSE`).
