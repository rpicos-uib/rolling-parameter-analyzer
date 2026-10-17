# rolling-parameter-analyzer

A desktop (Tkinter) tool for inspecting how the twelve parameters of the
Dynamic Memdiode Model (DMM, Miranda et al., *IEEE Electron Device
Letters* 47(6), 2026) evolve when they are extracted cycle by cycle from
a chain of ReRAM switching measurements. It is the companion tool for

> R. Picos, C. de Benito, O. Camps, "On the Inaccuracy of Global
> Optimization Methods to Extract ReRAM Parameters" (in preparation).
> Source and a compiled PDF are in [`paper/`](paper/main.pdf).

The extraction pipeline that produces its input CSVs is not part of this
repository; this tool is the *viewer*, built to make the per-cycle
trajectories, distributions and fit quality of an already-run extraction
chain inspectable without re-plotting everything by hand every time a
new run finishes.

## What it does

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

Data is read from disk on demand ("Refresh data"), so the tool doubles as
a live monitor for a still-running background extraction job.

## Input format

A CSV with one row per half-cycle switching event, columns:

```
cycle, kind, i_lo, i_peak, i_hi, rmse, retried,
ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam, g0
```

`kind` is `max` for a 0→V_max→0 (SET) half-cycle or `min` for a
0→V_min→0 (RESET) half-cycle. `i_lo`/`i_peak`/`i_hi` are sample indices
into the raw (t, V, I) trace this row's window spans. This is exactly
the format written by a chained `fit_region_pair()`-based extraction run
(see the paper above for the full fitting procedure).

## Running it

```bash
pip install -r requirements.txt
python3 analyze_rolling_fit.py --csv path/to/rolling_fit.csv
```

The raw measurement trace (`t`, `V`, `I`) is expected at the path named
by `data_utils.DATA_PATH` (`dades/dades.mat` by default, a MATLAB v7.3
file with `curr`/`cycle`/`time`/`volt` datasets) — override by editing
that constant, or point your own loader at it.

## Files

- `analyze_rolling_fit.py` — the viewer itself.
- `dmm_model.py` — the DMM forward model (`simulate()`), Numba-jitted.
- `fit_region_pair.py` — the joint least-squares / GA / directed-GA /
  GA-seeded-lsq fitting routines the viewer's Curve Inspector and Fuzzy
  Extractor tabs call for live resimulation and reference fits.
- `extract_fit.py` — half-cycle window detection and the interpretable,
  per-parameter iterative extraction procedure used by the Fuzzy
  Extractor tab.
- `data_utils.py` — raw-data loading and the default parameter-bounds
  table shared by every fitting routine.
- `fit_all_cycles.py` — `SET_SIDE`/`RESET_SIDE` parameter groupings and
  the per-cycle batch-fitting routine the viewer's data model mirrors.

## License

MIT (see `LICENSE`).
