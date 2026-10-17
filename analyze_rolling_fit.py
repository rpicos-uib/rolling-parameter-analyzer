"""Analyzer for fit_all_cycles.py's rolling_fit.csv: visualizes how the
chained, window-by-window DMM parameter fits evolve across cycles. The
source CSV is written incrementally by a possibly-still-running
background fit, so this is a live viewer, not a one-shot plot -- a
"Refresh data" button re-reads it from disk on demand (no auto-polling).

Each row in rolling_fit.csv is one WINDOW (see fit_all_cycles.py's own
docstring), and two rows cover one cycle (its own 'set' side gets fit in
one window, its 'reset' side in the next). For all the views here, the
data is first collapsed to ONE ROW PER CYCLE -- the LAST window that
reached each `cycle_c`, i.e. the most up-to-date 12-parameter state once
that cycle's own set AND reset events are both incorporated -- via
collapse_to_per_cycle(). `bad` windows (the retry-then-restart recovery
scheme's own flag, see fit_all_cycles.py) are excluded by default, since
their parameter values are explicitly unreliable; toggle with the
"Exclude bad windows" checkbox. A "Cycle from: / to:" interval selector
(blank = no limit on that end, applied on Enter/focus-out) restricts
both views to a cycle range -- e.g. to zoom into a region of interest
without the full, ever-growing dataset cluttering the view.

Four views (tabs):
  1. "Evolution vs cycle" -- any subset of the 12 parameters, PLUS the
     window's own fit `rmse` (under "Fit quality:", checked by default --
     useful to spot a whole stretch of poor fits at a glance, not just
     one parameter drifting), each independently RESCALED to [-1, 1]
     (min-max over the currently loaded/filtered data) so very
     different-scale parameters (e.g. ioff ~1e-4 vs ron ~50) can be
     compared on one shared axis, plotted against cycle number. A
     slider + "Show actual vs. model" button at
     the bottom picks one cycle (snapping to the nearest one actually
     present in the currently filtered data) and opens a POPUP window
     (a separate matplotlib figure, same convention as gui_view.py's own
     state-change/return-map popups) with real I-V and I(t) against
     dmm_model.simulate(), for that cycle AND N_NEIGHBOR_CYCLES (2)
     cycles before/after it -- each cycle simulated with its OWN fitted
     parameters and color-coded, so cycle-to-cycle evolution is visible
     directly in the overlaid loops, not just a single-cycle snapshot.
     By request (2026-10-11), a SECOND popup opens alongside it: a
     plain table of all 12 DMM parameters (rows) for each plotted
     cycle (columns), column headers color-matched to that cycle's own
     curve in the first popup -- see show_cycle_vs_model() /
     _show_param_table().
  2. "Parameter vs parameter" -- scatter of one parameter against
     another, either from the SAME cycle or the PREVIOUS cycle (a
     return-map-style lag-1 view, same idea as gui_view.py's own
     Vreset(n) vs Vreset(n-1) return maps, generalized to any pair of
     the 12 fitted parameters), with a least-squares regression line and
     its R^2 annotated.
  3. "Distribution" (by request, 2026-10-11) -- a histogram (density)
     of ONE selected parameter (or rmse) over the currently
     loaded/filtered per-cycle data, with any number of distributions
     from PARAM_SPECS overlaid AT ONCE: Gaussian, Lognormal, Gamma,
     Exponential, Logistic, Skew-normal, Weibull -- a mix of "usually
     reasonable for a small, roughly unimodal, physically-bounded
     quantity" (Gaussian) and "usually reasonable for a non-negative,
     right-skewed one like rmse" (Lognormal/Gamma/Weibull, alongside
     Exponential/Logistic/Skew-normal for mild skew/heavy tails). A
     resizable left-hand panel (drag the sash to resize, by request
     2026-10-11) lists all of them, each with its own checkbox (show
     it or not) and its own "k" spinner (1-8): k=1 is that
     distribution's own plain scipy.stats.<dist>.fit(), k>1 the
     corresponding mixture of k components -- one unified representation
     (DistributionTab no longer has a separate "Fit:" dropdown plus a
     "<dist> mixture" entry for each distribution; a single checked
     row with k>1 IS that mixture) fit via scipy.optimize.curve_fit
     against the histogram itself, since scipy.stats has no built-in
     mixture fitter -- see _fit_mixture(). Fitted parameters for every
     currently-shown distribution, plus a couple of basic sample stats
     (n, mean, std), are listed in a scrollable text box in the left
     panel, color-matched to each distribution's own curve.
  4. "Parameter table" (by request, 2026-10-11) -- a live, full-tab
     version of show_cycle_vs_model()'s own small parameter-table
     popup (ParamTableTab): rows are the 12 DMM parameters, columns
     are cycles -- as many as fit the CURRENT WINDOW WIDTH (recomputed
     on resize), centered on an editable "Central cycle" entry (same
     editable-box convention the Evolution tab's slider box uses), with
     single-cycle and full-page Prev/Next buttons to navigate beyond
     what's currently visible.

Usage:
    python3 analyze_rolling_fit.py
    python3 analyze_rolling_fit.py --csv dades/rolling_fit.csv
"""
import argparse
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

import dmm_model as dmm
import extract_fit as ef
import fit_region_pair as frp
from data_utils import DATA_PATH, load_data, cycle_bounds, moving_average, smooth_on_uniform_grid
from fit_all_cycles import SET_SIDE, RESET_SIDE

DEFAULT_CSV = "dades/rolling_fit.csv"
DEFAULT_SELECTED = {"vs", "vr", "rmse"}  # pre-checked on startup, so the first view isn't blank
EXTRA_SERIES = ["rmse"]  # plottable alongside dmm.PARAM_NAMES in EvolutionTab, not a model parameter
N_NEIGHBOR_CYCLES = 2  # cycles shown before/after the selected one in the actual-vs-model popup
SMOOTH_N = 10  # moving_average window for CurveInspectorTab's smoothed-derivative overlay


def read_csv_tolerant(path):
    """Reads `path`, tolerating a torn last line -- the background fit
    writes incrementally, so a refresh can land mid-write. Falls back to
    skipping unparseable lines; raises only if nothing usable is read.

    Also accepts a chain CSV (extract_fit.chain_results_to_dataframe(), as
    written by fit_chain_cli.py or this app's own Chain mode) in place of
    fit_all_cycles.py's own window-based rolling_fit.csv -- this app's
    whole point, from its first use, was "point --csv at a different file
    and hit Refresh"; a chain CSV is already one row per half-cycle with
    the same 12 params + rmse, just under 'cycle'/'kind' instead of
    'cycle_c'/'window'/'bad'. Rather than duplicate the Evolution/Scatter/
    Distribution/Parameter-table views for it, adapt it ONCE here: rename
    cycle->cycle_c, synthesize window=row order (chain rows are already in
    fit order) and bad=False (chain mode has no bad-window detection) --
    then every existing view, and collapse_to_per_cycle's own "last window
    per cycle_c" logic (which for a 2-half-cycles-per-cycle chain means
    "the min/RESET half, once both sides of that cycle are fit" -- exactly
    the same up-to-date-state intent as rolling_fit.csv's own set+reset
    pair), works unchanged."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "cycle" in df.columns and "cycle_c" not in df.columns and "window" not in df.columns:
        df = df.rename(columns={"cycle": "cycle_c"})
        df["window"] = np.arange(len(df))
        df["bad"] = False
    return df


def collapse_to_per_cycle(df, exclude_bad=True):
    """Collapses window-level rows to ONE ROW PER CYCLE (`cycle_c`) --
    the LAST window (by fit order) that reached each cycle -- sorted by
    cycle. See module docstring for why."""
    if df.empty:
        return df
    if exclude_bad and "bad" in df.columns:
        df = df[~df["bad"].astype(bool)]
    if df.empty:
        return df
    df = df.sort_values("window").drop_duplicates(subset="cycle_c", keep="last")
    return df.sort_values("cycle_c").reset_index(drop=True)


def cycle_to_cycle_deltas(df, params):
    """One row per cycle in `df` (already collapsed to per-cycle, bad
    windows already excluded by the caller), with each `params` column
    replaced by value[cycle] - value[cycle-1] -- but ONLY when cycle-1 is
    ITSELF present in `df`. If it isn't (a bad/skipped cycle, or simply
    missing data), the delta is NaN rather than falling back to whatever
    earlier cycle IS present -- a gap, not a leap across it. `df` should
    be the full (not cycle-range-filtered) bad-excluded data, so a delta
    for the first cycle of some later-applied display range can still
    see its true previous cycle even if that one isn't itself displayed."""
    if df.empty:
        return pd.DataFrame(columns=["cycle_c", *params])
    cycles = df["cycle_c"].astype(int).to_numpy()
    idx_by_cycle = {c: i for i, c in enumerate(cycles)}
    out = {"cycle_c": cycles}
    for name in params:
        v = df[name].to_numpy(dtype=float)
        d = np.full(len(cycles), np.nan)
        for i, c in enumerate(cycles):
            prev_i = idx_by_cycle.get(c - 1)
            if prev_i is not None:
                d[i] = v[i] - v[prev_i]
        out[name] = d
    return pd.DataFrame(out)


def rescale_minus1_1(y):
    """Per-series min-max rescale to [-1, 1]. A constant series (max==min)
    maps to all zeros rather than dividing by zero."""
    y = np.asarray(y, dtype=float)
    lo, hi = np.nanmin(y), np.nanmax(y)
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-300:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo) * 2.0 - 1.0


class EvolutionTab:
    """View 1: selected parameters, each rescaled to [-1, 1], vs cycle.
    Also carries the cycle-picker slider + "actual vs model" popup button
    (they belong here, not the scatter tab, since picking a point ON this
    temporal view is the natural way to ask "what does THAT cycle's real
    data vs. the fit actually look like")."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)

        # Packed BEFORE left/right so it claims its height from the
        # bottom first; left/right then fill whatever's left above it.
        bottom = ttk.LabelFrame(self.frame, text="Inspect one cycle")
        bottom.pack(side="bottom", fill="x", padx=6, pady=6)
        slider_row = ttk.Frame(bottom)
        slider_row.pack(fill="x")
        ttk.Label(slider_row, text="Cycle:").pack(side="left")
        self.cycle_slider_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(slider_row, from_=0, to=1, orient="horizontal",
                                 variable=self.cycle_slider_var, command=self._on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.cycle_entry_var = tk.StringVar(value="--")
        self.cycle_entry = ttk.Entry(slider_row, textvariable=self.cycle_entry_var, width=8)
        self.cycle_entry.pack(side="left")
        self.cycle_entry.bind("<Return>", self._on_cycle_entry)
        self.cycle_entry.bind("<FocusOut>", self._on_cycle_entry)
        ttk.Button(bottom, text=f"Show actual vs. model (I-V, I-t): this cycle "
                   f"±{N_NEIGHBOR_CYCLES} cycles", command=self._show_cycle_popup).pack(
                   fill="x", pady=(4, 0))

        left = ttk.Frame(self.frame, width=170)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)
        ttk.Label(left, text="Parameters:").pack(anchor="w")
        self.check_vars = {}
        for name in dmm.PARAM_NAMES:
            v = tk.BooleanVar(value=name in DEFAULT_SELECTED)
            ttk.Checkbutton(left, text=name, variable=v, command=self._redraw).pack(anchor="w")
            self.check_vars[name] = v
        ttk.Label(left, text="Fit quality:").pack(anchor="w", pady=(8, 0))
        for name in EXTRA_SERIES:
            v = tk.BooleanVar(value=name in DEFAULT_SELECTED)
            ttk.Checkbutton(left, text=name, variable=v, command=self._redraw).pack(anchor="w")
            self.check_vars[name] = v
        sel_row = ttk.Frame(left)
        sel_row.pack(fill="x", pady=(8, 0))
        ttk.Button(sel_row, text="All", command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(sel_row, text="None", command=lambda: self._set_all(False)).pack(side="left", padx=4)

        self.show_delta_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Show cycle-to-cycle Δ", variable=self.show_delta_var,
                         command=self._redraw).pack(anchor="w", pady=(12, 0))

        right = ttk.Frame(self.frame)
        right.pack(side="left", fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).pack(fill="x")

    def _set_all(self, value):
        for v in self.check_vars.values():
            v.set(value)
        self._redraw()

    def _nearest_available_cycle(self, value):
        """Snaps a slider value to the nearest cycle actually present in
        self.app.df -- the slider's own from/to span the full min-max
        range, but filtered-out (bad/range-excluded) cycles leave gaps in
        between."""
        df = self.app.df
        if df.empty:
            return None
        cycles = df["cycle_c"].to_numpy()
        return int(cycles[np.argmin(np.abs(cycles - value))])

    def _on_slider_move(self, _value=None):
        c = self._nearest_available_cycle(self.cycle_slider_var.get())
        self.cycle_entry_var.set(str(c) if c is not None else "--")

    def _on_cycle_entry(self, _event=None):
        """Typing a cycle number directly into the (now-editable) box
        next to the slider and pressing Enter/tabbing out (by request,
        2026-10-11 -- it used to be a read-only label) moves the slider
        to it, snapping to the nearest cycle actually present, same as
        dragging the slider there would. An unparseable/out-of-data
        value just reverts the box to the slider's current position,
        rather than leaving stale or invalid text showing."""
        try:
            value = int(self.cycle_entry_var.get().strip())
        except ValueError:
            self._on_slider_move()
            return
        c = self._nearest_available_cycle(value)
        if c is None:
            self._on_slider_move()
            return
        self.cycle_slider_var.set(c)
        self.cycle_entry_var.set(str(c))

    def _show_cycle_popup(self):
        c = self._nearest_available_cycle(self.cycle_slider_var.get())
        if c is None:
            messagebox.showinfo("No data", "No cycle data is currently loaded/shown.")
            return
        self.app.show_cycle_vs_model(c)

    def on_data_changed(self):
        df = self.app.df
        if not df.empty:
            lo, hi = int(df["cycle_c"].min()), int(df["cycle_c"].max())
            self.slider.config(from_=lo, to=max(hi, lo + 1))
            # keep the current slider position if it's still in range, else snap to lo
            if not (lo <= self.cycle_slider_var.get() <= hi):
                self.cycle_slider_var.set(lo)
            self._on_slider_move()
        else:
            self.cycle_entry_var.set("--")
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        df = self.app.df
        selected = [name for name, v in self.check_vars.items() if v.get()]
        if df.empty:
            self.ax.set_title("No data loaded (check the CSV path / press Refresh)")
            self.canvas.draw_idle()
            return
        if not selected:
            self.ax.set_title("No parameters selected")
            self.canvas.draw_idle()
            return

        if self.show_delta_var.get():
            deltas = cycle_to_cycle_deltas(self.app.df_all_good, selected)
            deltas = deltas[deltas["cycle_c"].isin(df["cycle_c"])]
            cycles = deltas["cycle_c"].to_numpy()
            for name in selected:
                y = rescale_minus1_1(deltas[name].to_numpy())
                self.ax.plot(cycles, y, "-", marker=".", ms=3, lw=1, label=name)
            self.ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            self.ax.set_xlabel("cycle")
            self.ax.set_ylabel("Δ (change from previous cycle), rescaled to [-1, 1] per parameter")
            self.ax.set_title(f"{len(df)} cycles shown -- gaps mark a non-consecutive previous "
                               f"cycle (bad/missing), not a leap across it")
        else:
            cycles = df["cycle_c"].to_numpy()
            for name in selected:
                y = rescale_minus1_1(df[name].to_numpy())
                self.ax.plot(cycles, y, "-", marker=".", ms=3, lw=1, label=name)
            self.ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            self.ax.set_xlabel("cycle")
            self.ax.set_ylabel("value, rescaled to [-1, 1] per parameter")
            self.ax.set_title(f"{len(df)} cycles shown")
        self.ax.legend(fontsize=8, ncol=min(4, len(selected)))
        self.fig.tight_layout()
        self.canvas.draw_idle()


class ScatterTab:
    """View 2: one parameter vs another, same cycle or lag-1 (previous
    cycle), with a least-squares regression line and R^2."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)

        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="X:").pack(side="left")
        self.x_var = tk.StringVar(value=dmm.PARAM_NAMES[0])
        xc = ttk.Combobox(top, textvariable=self.x_var, values=dmm.PARAM_NAMES,
                           state="readonly", width=8)
        xc.pack(side="left", padx=4)
        xc.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        ttk.Label(top, text="Y:").pack(side="left", padx=(14, 0))
        self.y_var = tk.StringVar(value=dmm.PARAM_NAMES[1])
        yc = ttk.Combobox(top, textvariable=self.y_var, values=dmm.PARAM_NAMES,
                           state="readonly", width=8)
        yc.pack(side="left", padx=4)
        yc.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        self.lag_var = tk.StringVar(value="same")
        ttk.Radiobutton(top, text="X from the SAME cycle", variable=self.lag_var,
                         value="same", command=self._redraw).pack(side="left", padx=(18, 4))
        ttk.Radiobutton(top, text="X from the PREVIOUS cycle", variable=self.lag_var,
                         value="prev", command=self._redraw).pack(side="left", padx=4)

        self.stats_var = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.stats_var, font=("", 10, "bold")).pack(
            anchor="w", padx=6)

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self.frame).pack(fill="x")

    def on_data_changed(self):
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        df = self.app.df
        xname, yname, lag = self.x_var.get(), self.y_var.get(), self.lag_var.get()

        if len(df) < 3:
            self.ax.set_title("Not enough data (need >=3 cycles)")
            self.canvas.draw_idle()
            self.stats_var.set("")
            return

        y = df[yname].to_numpy(dtype=float)
        if lag == "prev":
            x = df[xname].shift(1).to_numpy(dtype=float)
            xlabel = f"{xname} (previous cycle)"
        else:
            x = df[xname].to_numpy(dtype=float)
            xlabel = f"{xname} (same cycle)"

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 3:
            self.ax.set_title("Not enough overlapping data")
            self.canvas.draw_idle()
            self.stats_var.set("")
            return

        self.ax.scatter(x, y, s=12, alpha=0.5, color="0.3")
        slope, intercept = np.polyfit(x, y, 1)
        xfit = np.array([x.min(), x.max()])
        self.ax.plot(xfit, slope * xfit + intercept, "-", color="crimson", lw=1.5)

        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot > 0:
            yhat = slope * x + intercept
            r2 = 1.0 - np.sum((y - yhat) ** 2) / ss_tot
        else:
            r2 = float("nan")

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(f"{yname} (same cycle)")
        self.ax.set_title(f"{yname} vs {xname}, {'previous' if lag == 'prev' else 'same'} "
                           f"cycle (n={len(x)})")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.stats_var.set(f"y = {slope:.4g}·x + {intercept:.4g}      R² = {r2:.4g}"
                            f"      n = {len(x)}")


# name -> (scipy.stats distribution, [(param_name, kind), ...]) in the
# EXACT positional order both .fit() and .pdf() use for that
# distribution. `kind` drives both a single-component .fit() call's
# display labels AND how _fit_mixture() parameterizes/bounds/seeds that
# parameter across several components (see PARAM_KINDS below) -- one
# table now drives both the plain single-distribution fits and, by
# request (2026-10-11, extending the original Gaussian-only and then
# Gaussian+Weibull-only mixture support to all of PARAM_SPECS), a
# mixture of any number of components of ANY of these distributions.
# Chosen to make physical sense for this kind of data: Lognormal/Gamma
# (non-negative, right-skewed -- natural for a quantity like rmse,
# which can't go below 0 and has a long tail of bad fits), Exponential
# (the simplest non-negative-decay case, a sanity-check floor for
# Gamma/Weibull), Logistic (symmetric but heavier-tailed than Gaussian
# -- a common choice for threshold/breakdown-type quantities),
# Skew-normal (mild asymmetry without committing to a one-sided
# support), Weibull (classic breakdown/failure-time distribution).
#
# kind meanings:
#   "loc_free"     -- real-valued location, unconstrained support (as
#                      in Gaussian/Logistic/Skew-normal) -- free to
#                      differ per mixture component, that's what makes
#                      the mixture multi-modal.
#   "loc_bounded"  -- a location that's also a hard support boundary
#                      (Lognormal/Gamma/Exponential/Weibull all need
#                      x >= loc), bounded to just at-or-below the
#                      data's own min so every component's support
#                      still covers the whole sample.
#   "shape_pos_log" -- a positive shape parameter (Lognormal's s,
#                      Gamma's a, Weibull's k), fit in LOG-SPACE and
#                      exponentiated -- fit directly and abs()'d
#                      instead (first tried, 2026-10-11, for the
#                      Weibull mixture alone), curve_fit would wander
#                      into scale->0 / shape->huge regions near the
#                      support boundary and get stuck there (confirmed
#                      on real data: fits landing on scale=0 or
#                      c~1e5/loc~-1e4); log-space keeps the search
#                      smooth with no boundary to fall off.
#   "shape_free"   -- a real-valued shape parameter with no positivity
#                      need (Skew-normal's a, the skewness).
#   "scale_log"    -- the scale, always positive, same log-space
#                      reasoning as shape_pos_log.
PARAM_SPECS = {
    "gaussian": (stats.norm, [("loc", "loc_free"), ("scale", "scale_log")]),
    "lognormal": (stats.lognorm, [("s", "shape_pos_log"), ("loc", "loc_bounded"),
                                   ("scale", "scale_log")]),
    "gamma": (stats.gamma, [("a", "shape_pos_log"), ("loc", "loc_bounded"),
                             ("scale", "scale_log")]),
    "exponential": (stats.expon, [("loc", "loc_bounded"), ("scale", "scale_log")]),
    "logistic": (stats.logistic, [("loc", "loc_free"), ("scale", "scale_log")]),
    "skew-normal": (stats.skewnorm, [("a", "shape_free"), ("loc", "loc_free"),
                                      ("scale", "scale_log")]),
    "weibull": (stats.weibull_min, [("k", "shape_pos_log"), ("loc", "loc_bounded"),
                                     ("scale", "scale_log")]),
}


def _param_init_bounds(kind, i, k, lo, hi, span):
    """One component's (init, lower, upper) for one parameter `kind`,
    in whatever space that kind is actually optimized in (log-space for
    "*_log" kinds -- see PARAM_SPECS). `i` (this component's index among
    k) only matters for "loc_free", so repeated components start spread
    across the data's own range instead of all stacked on top of each
    other."""
    if kind == "scale_log":
        return np.log(span / (k + 1)), np.log(span / 1000), np.log(span * 10)
    if kind == "shape_pos_log":
        return np.log(1.5), np.log(0.05), np.log(50.0)
    if kind == "shape_free":
        return 0.0, -20.0, 20.0
    if kind == "loc_free":
        return lo + (i + 0.5) * span / k, lo - span, hi + span
    if kind == "loc_bounded":
        return lo - 1e-6 * span, lo - span, lo - 1e-9 * span
    raise ValueError(kind)


def _param_transform(kind, raw):
    """Maps a fitted raw value back to the actual distribution
    parameter -- exp() for the log-space kinds, identity otherwise."""
    return np.exp(raw) if kind in ("scale_log", "shape_pos_log") else raw


def _mixture_pdf(x, dist, param_defs, k, *params):
    """PDF of a k-component mixture of `dist` (one of PARAM_SPECS's
    distributions, `param_defs` its own param kind list). `params` is k
    weight LOGITS (softmax'd back to a [0, 1]-summing-to-1 simplex here,
    so curve_fit has no simplex constraint to respect), followed by k
    groups of len(param_defs) per-component parameters, each in
    whatever space _param_init_bounds/_param_transform use for its own
    kind."""
    x = np.asarray(x, dtype=float)
    n = len(param_defs)
    logits = np.array(params[:k])
    w = np.exp(logits - np.max(logits))
    w = w / np.sum(w)
    comp_raw = params[k:]
    pdf = np.zeros_like(x)
    for i in range(k):
        raw_i = comp_raw[i * n:(i + 1) * n]
        actual = [_param_transform(kind, v) for (_, kind), v in zip(param_defs, raw_i)]
        pdf += w[i] * dist.pdf(x, *actual)
    return pdf


def _fit_mixture(x, fit_kind, k, bins=50):
    """Fits a k-component mixture of `fit_kind` (any PARAM_SPECS key)
    to `x`'s own histogram (density=True) via curve_fit -- scipy.stats
    has no built-in mixture fitter, so the histogram itself (counts vs
    bin center) stands in for the usual (xdata, ydata) curve_fit
    expects. Each parameter's own initial guess/bounds come from
    _param_init_bounds(); a genuinely bad/non-existent extra mode just
    settles on a near-duplicate or near-zero-weight component (a
    legitimate, informative result, not a failure -- verified directly,
    2026-10-11, requesting more components than a synthetic test
    mixture actually had).
    Returns (w, components) -- w a length-k array, components a list of
    k per-component parameter tuples (in `dist.pdf()`'s own positional
    order), both sorted by each component's own `dist.mean()` so
    repeated fits/redraws don't reshuffle which component is "first"."""
    dist, param_defs = PARAM_SPECS[fit_kind]
    n = len(param_defs)
    counts, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lo, hi = float(np.min(x)), float(np.max(x))
    span = max(hi - lo, 1e-12)

    p0, lower, upper = [0.0] * k, [-10.0] * k, [10.0] * k
    for i in range(k):
        for _, kind in param_defs:
            v0, lo_b, hi_b = _param_init_bounds(kind, i, k, lo, hi, span)
            p0.append(v0)
            lower.append(lo_b)
            upper.append(hi_b)

    f = lambda xx, *p: _mixture_pdf(xx, dist, param_defs, k, *p)
    popt, _ = curve_fit(f, centers, counts, p0=p0, bounds=(lower, upper), maxfev=30000)

    logits = np.array(popt[:k])
    w = np.exp(logits - np.max(logits))
    w = w / np.sum(w)
    comp_raw = popt[k:]
    components = [tuple(_param_transform(kind, v) for (_, kind), v in
                         zip(param_defs, comp_raw[i * n:(i + 1) * n]))
                  for i in range(k)]

    means = [dist.mean(*c) for c in components]
    order = np.argsort(means)
    return w[order], [components[i] for i in order]


class DistributionTab:
    """View 3: histogram (density) of one selected parameter (or rmse)
    over the currently loaded/filtered per-cycle data, with any number
    of PARAM_SPECS distributions overlaid AT ONCE (by request,
    2026-10-11, replacing an earlier single "Fit:" dropdown -- each
    row in the left-hand panel is one distribution, a checkbox to show
    it plus its OWN "k" components spinner (1-8); k=1 is just that
    distribution's own plain scipy.stats.<dist>.fit(), k>1 the
    corresponding mixture -- the same single curve-fit-or-mixture
    PARAM_SPECS/_fit_mixture() machinery either way, so there's no
    separate "gaussian" vs "gaussian mixture" entry to pick between
    any more, just one unified row per distribution). The left panel
    is a resizable pane (ttk.PanedWindow, drag the sash between it and
    the plot to resize) rather than a fixed-width sidebar, since with
    up to 8 distributions x up to 8 components each the per-distribution
    fitted-parameter text can get long."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)

        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Parameter:").pack(side="left")
        self.param_var = tk.StringVar(value="rmse")
        names = list(dmm.PARAM_NAMES) + EXTRA_SERIES
        pc = ttk.Combobox(top, textvariable=self.param_var, values=names,
                           state="readonly", width=10)
        pc.pack(side="left", padx=4)
        pc.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        ttk.Label(top, text="Bins:").pack(side="left", padx=(14, 2))
        self.bins_var = tk.StringVar(value="50")
        bins_entry = ttk.Entry(top, textvariable=self.bins_var, width=5)
        bins_entry.pack(side="left")
        bins_entry.bind("<Return>", lambda e: self._redraw())
        bins_entry.bind("<FocusOut>", lambda e: self._redraw())

        paned = ttk.PanedWindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=0)

        ttk.Label(left, text="Distributions:").pack(anchor="w", padx=6, pady=(6, 2))
        self.dist_vars = {}  # name -> (BooleanVar shown, StringVar k)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, dist_name in enumerate(PARAM_SPECS):
            row = ttk.Frame(left)
            row.pack(fill="x", padx=6, pady=1)
            shown = tk.BooleanVar(value=False)
            swatch = "#%02x%02x%02x" % tuple(int(c * 255) for c in colors[i % 10][:3])
            ttk.Checkbutton(row, text=dist_name, variable=shown,
                             command=self._redraw).pack(side="left")
            tk.Label(row, text="█", fg=swatch).pack(side="left", padx=(2, 0))
            k_var = tk.StringVar(value="1")
            spin = ttk.Spinbox(row, from_=1, to=8, textvariable=k_var, width=3,
                                 command=self._redraw)
            spin.pack(side="right")
            spin.bind("<Return>", lambda e: self._redraw())
            spin.bind("<FocusOut>", lambda e: self._redraw())
            self.dist_vars[dist_name] = (shown, k_var)
        ttk.Label(left, text="k = number of mixture\ncomponents (1 = plain fit)",
                   font=("", 8), foreground="gray40").pack(anchor="w", padx=6, pady=(4, 0))

        sel_row = ttk.Frame(left)
        sel_row.pack(fill="x", pady=(8, 4), padx=6)
        ttk.Button(sel_row, text="All", command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(sel_row, text="None", command=lambda: self._set_all(False)).pack(
            side="left", padx=4)

        self.stats_text = tk.Text(left, height=10, width=1, wrap="word", font=("", 8),
                                    state="disabled")
        self.stats_text.pack(fill="both", expand=True, padx=6, pady=(8, 6))

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).pack(fill="x")

        self.frame.after(50, lambda: paned.sashpos(0, 230))

    def _set_all(self, value):
        for shown, _ in self.dist_vars.values():
            shown.set(value)
        self._redraw()

    def on_data_changed(self):
        self._redraw()

    def _set_stats_text(self, text):
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

    def _redraw(self):
        self.ax.clear()
        df = self.app.df
        name = self.param_var.get()
        if df.empty or name not in df.columns:
            self.ax.set_title("No data loaded (check the CSV path / press Refresh)")
            self.canvas.draw_idle()
            self._set_stats_text("")
            return

        x = df[name].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 5:
            self.ax.set_title("Not enough data (need >=5 values)")
            self.canvas.draw_idle()
            self._set_stats_text("")
            return

        try:
            bins = max(5, int(self.bins_var.get()))
        except ValueError:
            bins = 50

        counts, edges, _ = self.ax.hist(x, bins=bins, density=True, color="0.7",
                                         edgecolor="0.4", alpha=0.85)
        lines = [f"n = {len(x)}   mean = {np.mean(x):.4g}   std = {np.std(x):.4g}"]

        selected = [name for name, (shown, _) in self.dist_vars.items() if shown.get()]
        if selected:
            xs = np.linspace(edges[0], edges[-1], 400)
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for i, dist_name in enumerate(PARAM_SPECS):
                if dist_name not in selected:
                    continue
                color = colors[i % 10]
                _, k_var = self.dist_vars[dist_name]
                try:
                    k = max(1, min(8, int(k_var.get())))
                except ValueError:
                    k = 1
                dist, param_defs = PARAM_SPECS[dist_name]
                try:
                    if k == 1:
                        params = dist.fit(x)
                        curve = dist.pdf(xs, *params)
                        param_txt = ", ".join(f"{pname}={v:.4g}" for (pname, _), v in
                                               zip(param_defs, params))
                        lines.append(f"{dist_name} (k=1): {param_txt}")
                    else:
                        w, components = _fit_mixture(x, dist_name, k, bins=bins)
                        curve = sum(w[j] * dist.pdf(xs, *components[j]) for j in range(k))
                        comp_txts = []
                        for j in range(k):
                            param_txt = ", ".join(f"{pname}{j+1}={v:.4g}" for (pname, _), v in
                                                   zip(param_defs, components[j]))
                            comp_txts.append(f"(w{j+1}={w[j]:.3g}, {param_txt})")
                        lines.append(f"{dist_name} (k={k}): " + ", ".join(comp_txts))
                    self.ax.plot(xs, curve, "-", lw=2, color=color,
                                 label=f"{dist_name} (k={k})")
                except Exception as e:
                    lines.append(f"{dist_name}: fit failed -- {e}")
            self.ax.legend(fontsize=8)

        self.ax.set_xlabel(name)
        self.ax.set_ylabel("density")
        self.ax.set_title(f"Distribution of {name} ({len(x)} cycles)")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._set_stats_text("\n".join(lines))


class ParamTableTab:
    """View 4 (by request, 2026-10-11): a live, embedded version of
    show_cycle_vs_model()'s own small ±N_NEIGHBOR_CYCLES parameter
    table popup (see App._show_param_table()), but filling the whole
    tab: as many cycle COLUMNS as fit the current window width
    (recomputed on resize, "Columns:" in the status line), rows fixed
    at the 12 DMM parameters, centered on a user-editable "Central
    cycle" entry (same editable-box convention as the Evolution tab's
    slider box) with both single-cycle and full-page Prev/Next
    navigation buttons."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)
        self._n_cols = 1  # updated by _redraw(), used by the page-step buttons
        self._resize_job = None

        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Central cycle:").pack(side="left")
        self.center_var = tk.StringVar(value="--")
        entry = ttk.Entry(top, textvariable=self.center_var, width=8)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda e: self._redraw())
        entry.bind("<FocusOut>", lambda e: self._redraw())

        ttk.Button(top, text="◀◀ Prev page", command=self._page_prev).pack(
            side="left", padx=(14, 2))
        ttk.Button(top, text="◀ Prev cycle", command=lambda: self._step(-1)).pack(
            side="left", padx=2)
        ttk.Button(top, text="Next cycle ▶", command=lambda: self._step(1)).pack(
            side="left", padx=2)
        ttk.Button(top, text="Next page ▶▶", command=self._page_next).pack(
            side="left", padx=2)

        self.status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=12)

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.get_tk_widget().bind("<Configure>", self._on_resize)

    def _cycles(self):
        df = self.app.df
        return sorted(df["cycle_c"].unique()) if not df.empty else []

    def _nearest_cycle(self, value, cycles):
        arr = np.asarray(cycles)
        return int(arr[np.argmin(np.abs(arr - value))])

    def _parse_center(self):
        try:
            return int(self.center_var.get().strip())
        except ValueError:
            return None

    def _on_resize(self, _event=None):
        # Debounced -- a window drag fires many <Configure> events; only
        # the last one (200ms after dragging stops) actually redraws.
        if self._resize_job is not None:
            self.frame.after_cancel(self._resize_job)
        self._resize_job = self.frame.after(200, self._redraw)

    def _step(self, delta):
        cycles = self._cycles()
        if not cycles:
            return
        center = self._parse_center()
        center = self._nearest_cycle(center, cycles) if center is not None else cycles[len(cycles) // 2]
        idx = cycles.index(center)
        idx = max(0, min(len(cycles) - 1, idx + delta))
        self.center_var.set(str(cycles[idx]))
        self._redraw()

    def _page_prev(self):
        self._step(-max(1, self._n_cols))

    def _page_next(self):
        self._step(max(1, self._n_cols))

    @staticmethod
    def _window_bounds(idx_center, n_cols, n_total):
        """[lo, hi] (inclusive indices into a 0..n_total-1 cycle list)
        for a run of n_cols entries centered on idx_center, clamped to
        stay inside range (shifting the whole window rather than
        truncating it, so a page near either end still shows a FULL
        n_cols columns whenever n_total allows it)."""
        n_cols = min(n_cols, n_total)
        half = n_cols // 2
        lo = idx_center - half
        hi = lo + n_cols - 1
        if lo < 0:
            hi -= lo
            lo = 0
        if hi > n_total - 1:
            lo -= hi - (n_total - 1)
            hi = n_total - 1
        return max(0, lo), hi

    def on_data_changed(self):
        cycles = self._cycles()
        if cycles:
            center = self._parse_center()
            if center is None or self._nearest_cycle(center, cycles) not in cycles:
                center = cycles[len(cycles) // 2]
            self.center_var.set(str(self._nearest_cycle(center, cycles)))
        else:
            self.center_var.set("--")
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.axis("off")
        cycles = self._cycles()
        if not cycles:
            self.ax.set_title("No data loaded (check the CSV path / press Refresh)")
            self.canvas.draw_idle()
            self.status_var.set("")
            return

        center = self._parse_center()
        center_cycle = self._nearest_cycle(center, cycles) if center is not None else \
            cycles[len(cycles) // 2]
        if str(center_cycle) != self.center_var.get():
            self.center_var.set(str(center_cycle))
        idx_center = cycles.index(center_cycle)

        widget_w = self.canvas.get_tk_widget().winfo_width()
        if widget_w <= 1:
            widget_w = 900  # not yet mapped -- first draw, before a real size is known
        width_in = widget_w / self.fig.dpi
        n_cols = max(3, min(len(cycles), int((width_in - 1.4) / 0.85)))
        self._n_cols = n_cols

        lo, hi = self._window_bounds(idx_center, n_cols, len(cycles))
        window_cycles = cycles[lo:hi + 1]

        rows = dmm.PARAM_NAMES
        cell_text = []
        for pname in rows:
            row_vals = []
            for c in window_cycles:
                d = dmm.unpack(self.app._merged_params(c, cycles))
                row_vals.append(f"{d[pname]:.4g}")
            cell_text.append(row_vals)

        table = self.ax.table(cellText=cell_text, rowLabels=rows,
                               colLabels=[str(c) for c in window_cycles],
                               loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        if center_cycle in window_cycles:
            col = window_cycles.index(center_cycle)
            table[(0, col)].set_facecolor("#ffe08a")
            for row in range(len(rows) + 1):
                table[(row, col)].set_edgecolor("crimson")
                table[(row, col)].set_linewidth(1.6)

        self.fig.suptitle(f"Fitted parameters vs cycle (centered on {center_cycle})")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set(f"showing {len(window_cycles)} of {len(cycles)} cycles "
                             f"({window_cycles[0]}-{window_cycles[-1]})")


class CurveInspectorTab:
    """View 5 (by request, 2026-10-12): one selected cycle (+/- an
    adjustable number of neighbors), actual vs. model, as FOUR plots at
    once -- I-V, dI/dV, I(t), dI/dt -- so the shape AND the rate-of-change
    of both are visible side by side without needing the separate "Show
    actual vs. model" popup. Derivatives are plain np.gradient of the
    SAME plotted curves (data and model each use the one method, so the
    two stay directly comparable) -- not dmm_model.py's analytic_didt --
    deliberately, so what's plotted is always literally "the derivative
    of the curve drawn in the panel above it", nothing fancier.

    dI/dV is ill-conditioned right at a sweep's own V turning points
    (dV/dt, and so dV between samples, ~0 there -- division blows up) --
    this is real, not a bug (see the 2026-10-12 session investigation of
    why fitted vr sits so far from the apparent crossing), so rather than
    hide it, the y-axis is just clipped to the 1st-99th percentile of the
    finite values so a couple of turning-point spikes don't flatten the
    whole rest of the curve.

    Dashed vertical lines (color-matched per cycle) mark this cycle's
    EXTRACTED vs/vr (by request, 2026-10-12, cycle 3039 investigation --
    see RULES_DOC.md): the first positive peak of d2I/dt2 on the real
    data, V>0.2 (vs) / V<-0.2 (vr) -- see
    fit_region_pair.estimate_threshold_from_d2idt2()'s own docstring.
    NOT the bare model parameter's own numeric value, which, confirmed
    directly on real data, can sit far from where the current visibly
    transitions (the self-quenching g^gamma term) -- see RULES_DOC.md
    section 9 for how far apart those two were before this fix. Falls
    back to the bare parameter if no positive peak is found. In the
    I-V/dI-dV panels the line sits exactly at this extracted V; in the
    I(t)/dI-dt panels there's no single well-defined "time" for a
    voltage value, so the line is placed at the nearest actual sample
    whose V matches it.

    Star markers (by request, 2026-10-14) sit on the I-V and I(t) panels
    at (vs, Is)/(vr, Ir) -- the real DATA current at that same extracted
    sample. The point of showing it: the raw extracted vs/vr (measured
    V, at the device's external terminals) and the model's own FITTED
    vs/vr (which gate VA, the voltage across just the fast I(VA,g)
    element -- see dmm_model.py's module docstring, V = VA +
    K(ron,roff,g)*I) are expected to differ by roughly that series-
    resistance drop, Ron/Roff * I -- NOT a sign the windows were placed
    on the wrong region. Is/Ir let that drop be eyeballed directly
    (gap ~= Ron/Roff * Is or Ir) instead of only inferred indirectly from
    fit_all_cycles.py's vs_vr_gap column."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)

        top = ttk.Frame(self.frame)
        top.pack(side="top", fill="x", padx=6, pady=6)
        ttk.Label(top, text="Cycle:").pack(side="left")
        self.cycle_slider_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(top, from_=0, to=1, orient="horizontal",
                                 variable=self.cycle_slider_var, command=self._on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.cycle_entry_var = tk.StringVar(value="--")
        self.cycle_entry = ttk.Entry(top, textvariable=self.cycle_entry_var, width=8)
        self.cycle_entry.pack(side="left")
        self.cycle_entry.bind("<Return>", self._on_cycle_entry)
        self.cycle_entry.bind("<FocusOut>", self._on_cycle_entry)

        ttk.Label(top, text="Neighbors:").pack(side="left", padx=(12, 2))
        self.neighbors_var = tk.IntVar(value=0)
        ttk.Spinbox(top, from_=0, to=5, width=3, textvariable=self.neighbors_var,
                    command=self._redraw).pack(side="left")

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.status_var).pack(side="top", anchor="w", padx=6)

        self.fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        (self.ax_iv, self.ax_it), (self.ax_didv, self.ax_didt) = axes
        # twinx() shares the SAME axes rect as its parent -- the plot box
        # itself doesn't resize, only a second y-axis (and its own label/
        # ticks, on the right) is overlaid, so the 4 panels stay the same
        # size (by request, 2026-10-12).
        self.ax_didv2 = self.ax_didv.twinx()
        self.ax_didt2 = self.ax_didt.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self.frame).pack(fill="x")

    def _nearest_available_cycle(self, value):
        df = self.app.df
        if df.empty:
            return None
        cycles = df["cycle_c"].to_numpy()
        return int(cycles[np.argmin(np.abs(cycles - value))])

    def _on_slider_move(self, _value=None):
        c = self._nearest_available_cycle(self.cycle_slider_var.get())
        self.cycle_entry_var.set(str(c) if c is not None else "--")
        self._redraw()

    def _on_cycle_entry(self, _event=None):
        try:
            value = int(self.cycle_entry_var.get().strip())
        except ValueError:
            self._on_slider_move()
            return
        c = self._nearest_available_cycle(value)
        if c is None:
            self._on_slider_move()
            return
        self.cycle_slider_var.set(c)
        self.cycle_entry_var.set(str(c))
        self._redraw()

    def on_data_changed(self):
        df = self.app.df
        if not df.empty:
            lo, hi = int(df["cycle_c"].min()), int(df["cycle_c"].max())
            self.slider.config(from_=lo, to=max(hi, lo + 1))
            if not (lo <= self.cycle_slider_var.get() <= hi):
                self.cycle_slider_var.set(lo)
            c = self._nearest_available_cycle(self.cycle_slider_var.get())
            self.cycle_entry_var.set(str(c) if c is not None else "--")
        else:
            self.cycle_entry_var.set("--")
        self._redraw()

    @staticmethod
    def _clipped_ylim(ax, *series):
        """1st-99th percentile of all finite values across `series` --
        see class docstring for why dI/dV needs this."""
        vals = np.concatenate([s[np.isfinite(s)] for s in series if len(s)])
        if vals.size == 0:
            return
        lo, hi = np.percentile(vals, [1, 99])
        if lo == hi:
            return
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    def _redraw(self):
        for ax in (self.ax_iv, self.ax_didv, self.ax_it, self.ax_didt, self.ax_didv2, self.ax_didt2):
            ax.clear()
        df = self.app.df
        if df.empty:
            self.ax_iv.set_title("No data loaded (check the CSV path / press Refresh)")
            self.canvas.draw_idle()
            return
        cycle = self._nearest_available_cycle(self.cycle_slider_var.get())
        if cycle is None:
            self.canvas.draw_idle()
            return

        try:
            window = self.app.simulate_cycle_window(cycle, int(self.neighbors_var.get()))
        except Exception as e:
            self.ax_iv.set_title(f"Can't load raw data: {e}")
            self.canvas.draw_idle()
            return
        if window is None or not window["results"]:
            self.ax_iv.set_title(f"Cycle {cycle}: no data")
            self.canvas.draw_idle()
            return

        results = window["results"]
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        didv_data_all, didv_model_all = [], []
        didt_data_all, didt_model_all = [], []
        d2idv2_data_all, d2idv2_model_all = [], []
        d2idt2_data_all, d2idt2_model_all = [], []
        t0_global = results[0]["t"][0]

        def _v_crossing_time(t_rel, V, target_v, positive):
            """t_rel at the sample whose V is closest to target_v, among
            samples on the right side of zero (SET's vs>0, RESET's vr<0)
            -- a simple nearest-match used only to place the vs/vr marker
            line in the TIME-domain panels at its own V value; vs/vr
            themselves are now EXTRACTED values (see simulate_cycle_window
            ()'s own docstring), not the bare model parameter."""
            mask = (V > 0) if positive else (V < 0)
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                return None
            j = idx[np.argmin(np.abs(V[idx] - target_v))]
            return t_rel[j]

        for color, r in zip(colors, results):
            c, t, V, I, I_model = r["cycle"], r["t"], r["V"], r["I"], r["I_model"]
            vs, vr = r["vs"], r["vr"]
            Is, Ir = r["Is"], r["Ir"]
            is_selected = c == cycle
            lw = 2.0 if is_selected else 1.0
            label = f"cycle {c}" + (" (selected)" if is_selected else "")
            t_rel = t - t0_global

            didv_data = np.gradient(I, V)
            didv_model = np.gradient(I_model, V)
            didt_data = np.gradient(I, t)
            didt_model = np.gradient(I_model, t)
            # SMOOTHED derivatives (by request, 2026-10-12): moving_average(.,
            # SMOOTH_N) applied to the raw gradient -- same convention
            # fit_region_pair.estimate_cycle_thresholds uses for dI/dV --
            # plotted as a thick line alongside the thin raw one, since the
            # raw derivative is dominated by sample-to-sample noise right
            # where the transition actually matters.
            didv_data_sm = smooth_on_uniform_grid(V, didv_data, SMOOTH_N)
            didv_model_sm = smooth_on_uniform_grid(V, didv_model, SMOOTH_N)
            didt_data_sm = smooth_on_uniform_grid(t, didt_data, SMOOTH_N)
            didt_model_sm = smooth_on_uniform_grid(t, didt_model, SMOOTH_N)
            didv_data_all.append(didv_data_sm); didv_model_all.append(didv_model_sm)
            didt_data_all.append(didt_data_sm); didt_model_all.append(didt_model_sm)
            # SECOND derivative (by request, 2026-10-12): one more
            # gradient+smooth step on top of the already-smoothed 1st
            # derivative (same "smooth before every derivative" principle
            # as RULES_DOC.md section 9.3), shown on a SECOND y-axis
            # (twinx() -- shares the panel's own rect, so the plot box
            # stays the same size) since d2I/dV2 and d2I/dt2 are on a very
            # different scale from dI/dV and dI/dt.
            d2idv2_data = smooth_on_uniform_grid(V, np.gradient(didv_data_sm, V), SMOOTH_N)
            d2idv2_model = smooth_on_uniform_grid(V, np.gradient(didv_model_sm, V), SMOOTH_N)
            d2idt2_data = smooth_on_uniform_grid(t, np.gradient(didt_data_sm, t), SMOOTH_N)
            d2idt2_model = smooth_on_uniform_grid(t, np.gradient(didt_model_sm, t), SMOOTH_N)
            d2idv2_data_all.append(d2idv2_data); d2idv2_model_all.append(d2idv2_model)
            d2idt2_data_all.append(d2idt2_data); d2idt2_model_all.append(d2idt2_model)

            self.ax_iv.plot(V, I, ".", ms=3, color=color, alpha=0.6)
            self.ax_iv.plot(V, I_model, "-", lw=lw, color=color, label=label)
            self.ax_iv.axvline(vs, color=color, lw=lw, ls="--")
            self.ax_iv.axvline(vr, color=color, lw=lw, ls="--")
            if np.isfinite(Is):
                self.ax_iv.plot([vs], [Is], "*", ms=12 if is_selected else 8,
                                 color=color, mec="black", mew=0.5, zorder=5)
            if np.isfinite(Ir):
                self.ax_iv.plot([vr], [Ir], "*", ms=12 if is_selected else 8,
                                 color=color, mec="black", mew=0.5, zorder=5)
            self.ax_didv.plot(V, didv_data, ".", ms=2, color=color, alpha=0.25)
            self.ax_didv.plot(V, didv_model, "-", lw=lw * 0.6, color=color, alpha=0.35)
            self.ax_didv.plot(V, didv_data_sm, "-.", lw=lw, color=color, alpha=0.8)
            self.ax_didv.plot(V, didv_model_sm, "-", lw=lw * 2.0, color=color)
            self.ax_didv.axvline(vs, color=color, lw=lw, ls="--")
            self.ax_didv.axvline(vr, color=color, lw=lw, ls="--")
            self.ax_didv2.plot(V, d2idv2_data, ":", lw=lw, color="red", alpha=0.5)
            self.ax_didv2.plot(V, d2idv2_model, "--", lw=lw, color="red", alpha=0.9)
            self.ax_it.plot(t_rel, I, ".", ms=3, color=color, alpha=0.6)
            self.ax_it.plot(t_rel, I_model, "-", lw=lw, color=color, label=label)
            self.ax_it.axvline(t_rel[0], color=color, lw=0.6, ls=":")
            t_vs = _v_crossing_time(t_rel, V, vs, positive=True)
            t_vr = _v_crossing_time(t_rel, V, vr, positive=False)
            if t_vs is not None:
                self.ax_it.axvline(t_vs, color=color, lw=lw, ls="--")
                if np.isfinite(Is):
                    self.ax_it.plot([t_vs], [Is], "*", ms=12 if is_selected else 8,
                                     color=color, mec="black", mew=0.5, zorder=5)
            if t_vr is not None:
                self.ax_it.axvline(t_vr, color=color, lw=lw, ls="--")
                if np.isfinite(Ir):
                    self.ax_it.plot([t_vr], [Ir], "*", ms=12 if is_selected else 8,
                                     color=color, mec="black", mew=0.5, zorder=5)
            self.ax_didt.plot(t_rel, didt_data, ".", ms=2, color=color, alpha=0.25)
            self.ax_didt.plot(t_rel, didt_model, "-", lw=lw * 0.6, color=color, alpha=0.35)
            self.ax_didt.plot(t_rel, didt_data_sm, "-.", lw=lw, color=color, alpha=0.8)
            self.ax_didt.plot(t_rel, didt_model_sm, "-", lw=lw * 2.0, color=color)
            self.ax_didt2.plot(t_rel, d2idt2_data, ":", lw=lw, color="red", alpha=0.5)
            self.ax_didt2.plot(t_rel, d2idt2_model, "--", lw=lw, color="red", alpha=0.9)
            self.ax_didt.axvline(t_rel[0], color=color, lw=0.6, ls=":")
            if t_vs is not None:
                self.ax_didt.axvline(t_vs, color=color, lw=lw, ls="--")
            if t_vr is not None:
                self.ax_didt.axvline(t_vr, color=color, lw=lw, ls="--")

        self.ax_iv.set_xlabel("V (V)"); self.ax_iv.set_ylabel("I (A)")
        self.ax_iv.set_title("I-V  (dots=actual, lines=model, dashed=extracted vs/vr, "
                              "star=Is/Ir at that point)")
        self.ax_iv.legend(fontsize=7, ncol=2)

        self.ax_didv.set_xlabel("V (V)"); self.ax_didv.set_ylabel("dI/dV (S)")
        self.ax_didv.set_title(f"dI/dV (left axis, per-cycle color) + d2I/dV2 (RED, right axis)\n"
                                f"faint=raw, bold/dashed=smoothed[{SMOOTH_N}], vertical dashed=extracted vs/vr")
        self._clipped_ylim(self.ax_didv, *didv_data_all, *didv_model_all)
        self.ax_didv2.set_ylabel("d2I/dV2 (S/V)", color="red")
        self.ax_didv2.tick_params(axis="y", colors="red")
        self._clipped_ylim(self.ax_didv2, *d2idv2_data_all, *d2idv2_model_all)

        self.ax_it.set_xlabel("t (s)"); self.ax_it.set_ylabel("I (A)")
        self.ax_it.set_title("I(t)  (dotted=cycle start, dashed=extracted vs/vr, "
                              "star=Is/Ir at that point)")
        self.ax_it.legend(fontsize=7, ncol=2)

        self.ax_didt.set_xlabel("t (s)"); self.ax_didt.set_ylabel("dI/dt (A/s)")
        self.ax_didt.set_title(f"dI/dt (left axis, per-cycle color) + d2I/dt2 (RED, right axis)\n"
                                f"faint=raw, bold/dashed=smoothed[{SMOOTH_N}]")
        self._clipped_ylim(self.ax_didt, *didt_data_all, *didt_model_all)
        self.ax_didt2.set_ylabel("d2I/dt2 (A/s2)", color="red")
        self.ax_didt2.tick_params(axis="y", colors="red")
        self._clipped_ylim(self.ax_didt2, *d2idt2_data_all, *d2idt2_model_all)

        rmses = [r["rmse"] for r in results]
        missing_txt = f", {window['n_missing']} cycle(s) missing" if window["n_missing"] else ""
        self.fig.suptitle(f"cycles {window['window_cycles'][0]}-{window['window_cycles'][-1]} "
                           f"(selected: {cycle}): mean rmse={np.mean(rmses):.4g}{missing_txt}")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        sel = next((r for r in results if r["cycle"] == cycle), None)
        is_ir_txt = ""
        if sel is not None:
            is_ir_txt = (f"   |   selected cycle {cycle}: Is={sel['Is']:.4g} A @ vs={sel['vs']:.4g} V, "
                         f"Ir={sel['Ir']:.4g} A @ vr={sel['vr']:.4g} V")
        self.status_var.set(f"{len(results)} cycle(s) simulated{is_ir_txt}")


class FuzzyExtractorTab:
    """View 6 (by request, 2026-10-14; rebuilt same day on extract_fit.py
    -- see that module's own docstring for the full rationale, "forget
    the previous fuzzy" rule-vote approach entirely): visualizes
    extract_fit.py's per-parameter EXTRACTED-VALUE iterative fit on ONE
    selected cycle -- an alternative lens to CurveInspectorTab's own
    least_squares-based view, showing not just the final fit but HOW it
    got there (one entry per actual parameter update: which parameter,
    its own extracted-data-vs-model error, old/new value, rmse as a
    DIAGNOSTIC only -- extract_fit() never uses rmse to decide
    anything, see its own docstring).

    "Run" does two things, self-contained for whichever cycle is
    entered:
      1. Locates that cycle's own window as its RAW logical-cycle bounds
         (data's own 'cycle' column, via cycle_bounds() -- confirmed
         2026-10-14 to already be one clean, complete 0->Vmax->0->
         Vmin->0 pair on its own, no detection needed; NOT
         fit_region_pair.find_fit_window()'s own detection-based
         vreset->vset->vreset window, which spans parts of TWO logical
         cycles and was found to anchor on estimate_cycle_thresholds --
         the SAME detector already confirmed, on cycle 4155, to lock
         onto the wrong reset event). ALSO runs the joint least_squares
         fit_region_pair() on this same window -- kept and shown
         alongside extract_fit's own result purely as a reference
         "answer key" for how close this cruder, interpretable method
         got, not as anything extract_fit() itself uses.
      2. Runs extract_fit.extract_fit() on the REAL (t,V,I) for that
         same window. "Seed from cycle" (optional): when given, seeds
         extract_fit with THAT cycle's own already-fitted row from
         rolling_fit.csv (via App._merged_params(), the same RESET_SIDE
         -staleness-patched merge every other tab uses) instead of
         extract_fit's own NOMINAL+direct-extraction default -- by
         request, "let me pick a cycle whose parameters you can use as
         a guess" (confirmed live, 2026-10-14: a good reference-cycle
         seed is what actually lets the device switch within THIS
         window's own voltage range, which every other fix depends on).
         Typing "bootstrap" instead of a cycle number runs
         extract_fit.bootstrap_half_cycles() on THIS cycle's own window
         instead (RULES_DOC.md section 19) -- a genuine two-stage
         least_squares fit (SET_SIDE over the window's own 0->Vmax->0
         span, RESET_SIDE over 0->Vmin->0), no reference cycle needed;
         validated as a real fix for the "NOMINAL seed never switches"
         problem, with a known, accepted long tail on ion/ioff/etar
         (a structural identifiability ridge, not a bug).

    Four panels: I-V and I(t) (actual vs extract_fit's own model AT THE
    CURRENTLY DISPLAYED STEP, same convention as CurveInspectorTab/
    gui_view.py), rmse-vs-step (diagnostic only, log scale -- NOT what
    extract_fit optimizes against), and every free parameter's own
    value vs step, each independently rescaled to [-1,1] (same
    convention as EvolutionTab) so very different-scale parameters
    share one axis."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)
        self._last_result = None
        self._window = self._joint_fit = self._cycle = None
        self._seed_label = ""
        self._manual_override = None
        # Per-parameter nudge-button state (by request, 2026-10-14, "the
        # +1%/-1% should not make a perfect round trip to avoid
        # oscillations"): same bisection-style damping extract_fit.py's own
        # persistent ratio_cap uses -- clicking the SAME direction again
        # keeps the current step size, clicking the OPPOSITE direction
        # (a reversal) HALVES it first. A perfectly-symmetric +1%/-1% would
        # let repeated clicking loop forever at a fixed amplitude instead of
        # ever homing in; shrinking on reversal turns it into manual
        # bisection instead. Reset on every new extract_fit run.
        self._nudge_pct_state = {name: None for name in dmm.PARAM_NAMES}
        self._nudge_last_sign = {name: 0 for name in dmm.PARAM_NAMES}
        self._history_display = []
        self._round_i = self._best_round_i = 0

        top = ttk.Frame(self.frame)
        top.pack(side="top", fill="x", padx=6, pady=6)
        ttk.Label(top, text="Cycle:").pack(side="left")
        self.cycle_var = tk.StringVar(value="2000")
        entry = ttk.Entry(top, textvariable=self.cycle_var, width=8)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda e: self._on_run())

        # Seed-from-cycle (by request, 2026-10-14): optional -- blank means
        # extract_fit's own default seed (NOMINAL + direct extraction where
        # validated). A reference cycle's own already-fitted row is a much
        # better starting point in practice (see class docstring). Typing
        # "bootstrap" instead of a cycle number (by request, 2026-10-14,
        # "wire it into the GUI's Seed from cycle") runs
        # extract_fit.bootstrap_half_cycles() on THIS cycle's own window
        # instead -- RULES_DOC.md section 19's own least-squares starting-
        # point generator: no reference cycle needed, but a known, accepted
        # long tail on ion/ioff/etar (see that section).
        ttk.Label(top, text="Seed from cycle (or 'bootstrap'):").pack(side="left", padx=(12, 2))
        self.seed_cycle_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.seed_cycle_var, width=10).pack(side="left")

        ttk.Label(top, text="Max passes:").pack(side="left", padx=(12, 2))
        self.max_passes_var = tk.StringVar(value="15")
        ttk.Entry(top, textvariable=self.max_passes_var, width=6).pack(side="left")

        ttk.Button(top, text="Run extract_fit", command=self._on_run).pack(side="left", padx=(12, 0))

        # Step-by-step playback (by request, 2026-10-14): "Run" still computes
        # the FULL history up front (cheap, same as before) -- these buttons
        # just move a read index through the already-computed per-round
        # snapshots and re-simulate+redraw at that one round, so stepping is
        # instant and never re-runs fuzzy_fit itself.
        ttk.Button(top, text="|< Start", command=lambda: self._step_to(0)).pack(side="left", padx=(16, 0))
        ttk.Button(top, text="< Step", command=self._step_back).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Step >", command=self._step_fwd).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="End >|", command=lambda: self._step_to(-1)).pack(side="left", padx=(4, 0))
        self.round_label_var = tk.StringVar(value="(run a fit first)")
        ttk.Label(top, textvariable=self.round_label_var).pack(side="left", padx=(10, 0))

        # Parameter trajectory focus (by request, 2026-10-14, "I'd like to see
        # in the step-by-step the individual parameter variation, not all of
        # them"): "(auto)" shows whichever parameter the CURRENT step actually
        # updated (extract_fit's own history already names it per step) --
        # the natural default, since only one parameter moves at a time.
        # Picking a specific name pins the panel to that parameter regardless
        # of which one is being updated at the current step.
        ttk.Label(top, text="Show param:").pack(side="left", padx=(12, 2))
        self.param_focus_var = tk.StringVar(value="(auto)")
        self.param_focus_combo = ttk.Combobox(top, textvariable=self.param_focus_var, width=8,
                                               values=["(auto)"] + ef.FREE_PARAMS_DEFAULT_ORDER, state="readonly")
        self.param_focus_combo.pack(side="left")
        self.param_focus_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        self.status_var = tk.StringVar(value="Enter a cycle and click \"Run fuzzy fit\".")
        ttk.Label(self.frame, textvariable=self.status_var, wraplength=1000, justify="left").pack(
            side="top", anchor="w", padx=6)

        # Editable parameter table (by request, 2026-10-14, "add a table with
        # the values of the parameters... editable?"): shows the CURRENTLY
        # DISPLAYED step's own values for all 12 dmm.PARAM_NAMES (including
        # gam/g0, which extract_fit holds fixed but still simulates with).
        # Editing a value and clicking "Apply manual edit" (or pressing
        # Enter in any field) re-simulates the I-V/I(t) curves with the
        # EDITED values -- a live, non-destructive "what if" override on top
        # of the current step. It does NOT change extract_fit's own
        # recorded history -- stepping to a different round clears it and
        # reverts to that step's own real values.
        # +-step% nudge buttons (by request, 2026-10-14, "the changes are
        # too big, can you make the changes relative? say increase a 1% or
        # decrease a 1%?"): typing a whole new absolute value makes it easy
        # to overshoot -- "-"/"+" multiply the CURRENT value by (1 -+
        # step_pct/100) and apply immediately, a controlled relative nudge
        # instead of a guessed absolute jump. step_pct itself is editable
        # (default 1%), shared by every parameter's own buttons.
        param_table_frame = ttk.Frame(self.frame)
        param_table_frame.pack(side="top", fill="x", padx=6, pady=(2, 4))
        self.param_entry_vars = {}
        for i, name in enumerate(dmm.PARAM_NAMES):
            r, c = divmod(i, 4)
            base = 4 * c
            ttk.Label(param_table_frame, text=name + ":").grid(
                row=r, column=base, sticky="e", padx=(10 if c else 0, 2), pady=1)
            var = tk.StringVar(value="")
            entry = ttk.Entry(param_table_frame, textvariable=var, width=10)
            entry.grid(row=r, column=base + 1, sticky="w")
            entry.bind("<Return>", lambda e: self._apply_manual_edit())
            self.param_entry_vars[name] = var
            ttk.Button(param_table_frame, text="-", width=2,
                       command=lambda n=name: self._nudge_param(n, -1)).grid(row=r, column=base + 2, padx=(2, 0))
            ttk.Button(param_table_frame, text="+", width=2,
                       command=lambda n=name: self._nudge_param(n, +1)).grid(row=r, column=base + 3, padx=(1, 4))
        ttk.Label(param_table_frame, text="Step %:").grid(row=0, column=16, sticky="e", padx=(14, 2))
        self.nudge_pct_var = tk.StringVar(value="1")
        ttk.Entry(param_table_frame, textvariable=self.nudge_pct_var, width=5).grid(row=0, column=17, sticky="w")
        ttk.Button(param_table_frame, text="Apply manual edit", command=self._apply_manual_edit).grid(
            row=1, column=16, columnspan=2, padx=(14, 0))

        # Real-data chain mode (by request, 2026-10-14): runs
        # extract_fit.chain_half_cycles() -- genuine fit_region_pair()
        # least_squares over N consecutive REAL logical cycles, alternating
        # 0->Vmax->0 (LRS free) / 0->Vmin->0 (HRS free), each half-cycle's
        # own fixed side held at the IMMEDIATELY PRECEDING half-cycle's own
        # value, with a dynamic +-ron_roff_bound_pct% bound on ron/roff
        # (RULES_DOC.md's own "smoking gun" real-data validation). A
        # DIFFERENT mode from the single-cycle extract_fit() view above --
        # switches the SAME 4 panels to a stitched, multi-cycle display
        # (_redraw_chain) instead of the per-step playback view
        # (_redraw/_step_to); the playback buttons are inert in this mode.
        chain_row = ttk.Frame(self.frame)
        chain_row.pack(side="top", fill="x", padx=6, pady=(0, 4))
        ttk.Label(chain_row, text="Chain from cycle:").pack(side="left")
        self.chain_start_var = tk.StringVar(value="2000")
        ttk.Entry(chain_row, textvariable=self.chain_start_var, width=8).pack(side="left", padx=(2, 10))
        ttk.Label(chain_row, text="# cycles:").pack(side="left")
        self.chain_n_var = tk.StringVar(value="1000")
        ttk.Entry(chain_row, textvariable=self.chain_n_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(chain_row, text="ron/roff bound %:").pack(side="left")
        self.chain_bound_pct_var = tk.StringVar(value="25")
        ttk.Entry(chain_row, textvariable=self.chain_bound_pct_var, width=5).pack(side="left", padx=(2, 10))
        ttk.Button(chain_row, text="Run chain (real least_squares)", command=self._on_run_chain).pack(side="left")
        # Save to CSV (by request, 2026-10-14, "can't you share the results and
        # the functions between the cli and the gui? It seems inefficient to
        # have them duplicated"): reuses extract_fit.chain_results_to_dataframe()
        # -- the SAME row-building function fit_chain_cli.py's own CLI uses --
        # so a chain run started in the GUI can be saved and then inspected
        # from the command line (or vice versa) with byte-identical columns.
        ttk.Button(chain_row, text="Save chain to CSV", command=self._on_save_chain_csv).pack(
            side="left", padx=(10, 0))
        self._mode = "single"  # "single" (extract_fit playback) or "chain" (chain_half_cycles)
        self._chain_results = None

        self.fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        (self.ax_iv, self.ax_it), (self.ax_rmse, self.ax_params) = axes
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self.frame).pack(fill="x")

    def on_data_changed(self):
        pass  # independent of rolling_fit.csv -- works directly off the raw measurement data

    def _on_run(self):
        self._mode = "single"
        try:
            cycle = int(self.cycle_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid cycle", "Cycle must be an integer.")
            return
        try:
            max_passes = max(1, int(self.max_passes_var.get().strip()))
        except ValueError:
            max_passes = 15
        seed_cycle_txt = self.seed_cycle_var.get().strip()
        seed_cycle = None
        use_bootstrap = False
        if seed_cycle_txt:
            if seed_cycle_txt.lower() == "bootstrap":
                use_bootstrap = True
            else:
                try:
                    seed_cycle = int(seed_cycle_txt)
                except ValueError:
                    messagebox.showerror("Invalid seed cycle",
                                          "Seed-from-cycle must be an integer, 'bootstrap', or blank.")
                    return

        try:
            (curr, cyc, time_, volt), bounds = self.app._get_raw_data()
        except Exception as e:
            messagebox.showerror("Can't load raw data", f"{self.app.data_path}:\n{e}")
            return
        # The window shown/fit is now the RAW logical cycle's own bounds
        # (by request, 2026-10-14, "rewire it to show the 0-Vmax-0 and
        # 0-Vmin-0 cycle... somewhere it's still using something old"):
        # find_fit_window()'s own detection-based vreset->vset->vreset
        # window (spanning parts of TWO logical cycles, anchored to the
        # SAME estimate_cycle_thresholds detector already found to lock
        # onto the wrong event on cycle 4155) is no longer used here at
        # all. A raw logical cycle (data's own 'cycle' column, via
        # cycle_bounds()) is confirmed to ALREADY be one clean, complete
        # 0->Vmax->0->Vmin->0 pair on its own, essentially starting and
        # ending at V~=0 -- no detection needed.
        if cycle not in bounds:
            messagebox.showwarning("Can't build window", f"Cycle {cycle} not present in this data.")
            return
        i_a, i_end_excl = bounds[cycle]
        i_c = i_end_excl - 1
        i_b = i_a + int(np.argmax(volt[i_a:i_c + 1]))  # Vmax index -- only for fit_region_pair()'s own region1/region2 API
        window = dict(i_vreset1=i_a, i_vset1=i_b, i_vreset2=i_c)

        i_a, i_b, i_c = window["i_vreset1"], window["i_vset1"], window["i_vreset2"]
        self._nudge_pct_state = {name: None for name in dmm.PARAM_NAMES}
        self._nudge_last_sign = {name: 0 for name in dmm.PARAM_NAMES}

        x0_dict = None
        seed_label = "extract_fit's own default (NOMINAL + direct extraction)"
        if use_bootstrap:
            self.status_var.set(f"Cycle {cycle}: running bootstrap_half_cycles() (two least_squares fits)...")
            self.canvas.draw_idle()
            self.frame.update_idletasks()
            boot = ef.bootstrap_half_cycles(time_, volt, curr, i_a, i_b, i_c)
            x0_dict = boot["params_dict"]
            seed_label = "bootstrap_half_cycles() least-squares starting point (RULES_DOC.md section 19 -- a real, accepted long tail on ion/ioff/etar)"
        elif seed_cycle is not None:
            cycles_available = sorted(self.app.df["cycle_c"].unique()) if not self.app.df.empty else []
            if seed_cycle not in cycles_available:
                messagebox.showwarning("Seed cycle not found",
                                        f"Cycle {seed_cycle} has no row in the currently loaded/filtered "
                                        f"rolling_fit.csv -- can't use it as a seed.")
                return
            x0_dict = dmm.unpack(self.app._merged_params(seed_cycle, cycles_available))
            seed_label = f"cycle {seed_cycle}'s own fitted row"

        self.status_var.set(f"Cycle {cycle}: fitting (joint least_squares, for reference) and running extract_fit "
                             f"(seed: {seed_label})...")
        self.canvas.draw_idle()
        self.frame.update_idletasks()

        joint_fit = frp.fit_region_pair(time_, volt, curr, i_a, i_b, i_c, max_nfev=2000)
        result = ef.extract_fit(time_, volt, curr, i_a, i_b, i_c, x0_dict=x0_dict, max_outer_passes=max_passes)

        self._last_result = result
        self._cycle, self._window, self._joint_fit = cycle, window, joint_fit
        self._seed_label = seed_label
        history = result["history"]
        if not history:
            # No update was ever applied (e.g. the seed was already a good
            # match, or the device never switched within max_inner_iters) --
            # synthesize a single display-only entry from the seed/final
            # state so the playback UI still has something to show.
            history = [dict(round=0, params=dict(result["params_dict"]), rmse=result["rmse"])]
        self._history_display = history
        self._best_round_i = min(range(len(history)), key=lambda i: history[i]["rmse"])
        self._step_to(-1)  # land on the FINAL state (what params_dict actually is)

    def _step_fwd(self):
        if self._mode != "single" or self._last_result is None:
            return
        self._step_to(min(self._round_i + 1, len(self._history_display) - 1))

    def _step_back(self):
        if self._mode != "single" or self._last_result is None:
            return
        self._step_to(max(self._round_i - 1, 0))

    def _step_to(self, idx):
        """Jumps the playback cursor to history-index `idx` (-1 = last) and
        redraws AT THAT STEP -- never re-runs extract_fit, just re-simulates
        dmm.simulate() once with that step's own params snapshot (already in
        `history`, cheap) so stepping is instant (by request, 2026-10-14,
        "add a step-by-step button, so I see how the fitting evolves")."""
        if self._mode != "single" or self._last_result is None:
            return
        history = self._history_display
        if idx < 0:
            idx = len(history) - 1
        idx = max(0, min(idx, len(history) - 1))
        self._round_i = idx
        self._manual_override = None  # stepping reverts to that step's own real values
        self._redraw()

    def _nudge_param(self, name, sign):
        """Multiplies `name`'s own CURRENT entry value by (1 +- pct/100) and
        applies immediately -- a controlled relative nudge (by request,
        "the changes are too big... make the changes relative, say increase
        a 1% or decrease a 1%"). `pct` is `name`'s own CURRENT step size
        (starts at the shared "Step %" field) -- clicking the SAME
        direction again keeps it, clicking the OPPOSITE direction (a
        reversal) HALVES it first (by request, "should not make a perfect
        round trip to avoid oscillations" -- bisection, not a fixed-
        amplitude loop)."""
        if self._mode != "single" or self._last_result is None:
            return
        pct = self._nudge_pct_state[name]
        if pct is None:
            try:
                pct = float(self.nudge_pct_var.get())
            except ValueError:
                messagebox.showerror("Invalid step %", "Step % must be a number.")
                return
        last_sign = self._nudge_last_sign[name]
        if last_sign != 0 and sign != last_sign:
            pct *= 0.5
        self._nudge_pct_state[name] = pct
        self._nudge_last_sign[name] = sign
        try:
            old = float(self.param_entry_vars[name].get())
        except ValueError:
            return  # field doesn't hold a number yet -- nothing to nudge from
        new = old * (1.0 + sign * pct / 100.0)
        self.param_entry_vars[name].set(f"{new:.6g}")
        self._apply_manual_edit()

    def _apply_manual_edit(self):
        """Reads the parameter table's own entries and re-simulates the
        I-V/I(t) curves with them -- a live override on top of the
        currently-displayed step, kept until the next Step/Start/End click
        clears it back to that step's own real values."""
        if self._mode != "single" or self._last_result is None:
            return
        try:
            p = {name: float(self.param_entry_vars[name].get()) for name in dmm.PARAM_NAMES}
        except ValueError:
            messagebox.showerror("Invalid value", "Every parameter field must be a number.")
            return
        self._manual_override = p
        self._redraw()

    def _redraw(self):
        result, joint_fit, cycle = self._last_result, self._joint_fit, self._cycle
        history = self._history_display
        idx = self._round_i
        snap = history[idx]
        display_params = self._manual_override if self._manual_override is not None else snap["params"]
        for name in dmm.PARAM_NAMES:
            self.param_entry_vars[name].set(f"{display_params[name]:.6g}")

        for ax in (self.ax_iv, self.ax_it, self.ax_rmse, self.ax_params):
            ax.clear()

        t_w, V_w, I_w = result["t"], result["V"], result["I"]
        t_rel = t_w - t_w[0]
        with np.errstate(over="ignore"):
            I_model, _, _ = dmm.simulate(t_w, V_w, dmm.pack(display_params))

        # Fitting-mask overlay (by request, 2026-10-14, "mark in different
        # colors each fitting mask"; fixed THREE times same day -- first
        # after "the masks are absolutely wrong... you're using Vmax and
        # Vmin to freeze things" (region1/region2 time-index split), again
        # after "you're still using the old masks, not the 0-vmax-0 and
        # 0-vmin-0 new ones" (frozen ON/OFF became the window's own
        # 0->Vmax->0 / 0->Vmin->0 half-cycle spans, SAME geometry
        # bootstrap_half_cycles() uses), and once more the same day after
        # confirming etas/etar were STILL using the OLD detected-threshold
        # +-60-sample anchor, independent of mask_on/mask_off, in BOTH the
        # extraction (extract_fit.extract_eta_width_masked now fixes this)
        # AND this drawing: SET-active/RESET-active below are ONLY vs/vr's
        # own detection now (etas/etar moved to the full frozen ON/OFF
        # spans) -- still drawn as a narrower sub-region on TOP of frozen
        # ON/OFF (wide regions first) so both remain visible. Unmasked
        # points stay plain gray.
        v_rising = np.concatenate(([True], np.diff(V_w) >= 0))
        set_active = (V_w > 0) & v_rising
        reset_active = (V_w < 0) & ~v_rising
        mask_on = ef._frozen_on_mask(V_w)
        mask_off = ef._frozen_off_mask(V_w)
        mask_specs = [("frozen ON (ron/aon/ion/etas)", mask_on, "dodgerblue"),
                      ("frozen OFF (roff/aoff/ioff/etar)", mask_off, "orangered"),
                      ("SET-active (vs only)", set_active, "goldenrod"),
                      ("RESET-active (vr only)", reset_active, "mediumorchid")]

        # frozen ON/OFF (wide regions) drawn LARGER, SET/RESET-active (narrow
        # sub-regions within them) drawn SMALLER on top with alpha -- both
        # stay visible where they overlap instead of one hiding the other.
        self.ax_iv.plot(V_w, I_w, ".", ms=3, color="0.85", label="actual (unmasked)")
        for label, mask, color in mask_specs:
            ms = 7 if "frozen" in label else 4
            self.ax_iv.plot(V_w[mask], I_w[mask], ".", ms=ms, color=color, alpha=0.7, label=label)
        self.ax_iv.plot(V_w, I_model, "-", lw=1.5, color="crimson", label=f"extract_fit model (step {snap['round']}{' manual' if self._manual_override is not None else ''})")
        self.ax_iv.set_xlabel("V (V)"); self.ax_iv.set_ylabel("I (A)")
        self.ax_iv.set_title(f"I-V -- cycle {cycle}")
        self.ax_iv.legend(fontsize=6, ncol=2)

        self.ax_it.plot(t_rel, I_w, ".", ms=3, color="0.85", label="actual (unmasked)")
        for label, mask, color in mask_specs:
            ms = 7 if "frozen" in label else 4
            self.ax_it.plot(t_rel[mask], I_w[mask], ".", ms=ms, color=color, alpha=0.7, label=label)
        self.ax_it.plot(t_rel, I_model, "-", lw=1.5, color="crimson", label=f"extract_fit model (step {snap['round']}{' manual' if self._manual_override is not None else ''})")
        self.ax_it.set_xlabel("t (s)"); self.ax_it.set_ylabel("I (A)")
        self.ax_it.set_title("I(t)")
        self.ax_it.legend(fontsize=6, ncol=2)

        rounds = [h["round"] for h in history]
        rmses = [h["rmse"] for h in history]
        best_round = history[self._best_round_i]["round"]
        self.ax_rmse.plot(rounds, rmses, "-", lw=1.2, color="darkorange")
        self.ax_rmse.axvline(best_round, color="0.5", lw=1, ls="--",
                              label=f"lowest rmse seen (step {best_round}, diagnostic only)")
        self.ax_rmse.axvline(snap["round"], color="steelblue", lw=1.5, ls="-",
                              label=f"playback cursor (step {snap['round']})")
        self.ax_rmse.axhline(joint_fit["rmse"], color="0.4", lw=1, ls=":",
                              label=f"joint least_squares rmse ({joint_fit['rmse']:.4g})")
        self.ax_rmse.set_yscale("log")
        self.ax_rmse.set_xlabel("step"); self.ax_rmse.set_ylabel("rmse (log scale, diagnostic only)")
        self.ax_rmse.set_title("Goodness of fit (NOT the optimization target)")
        self.ax_rmse.legend(fontsize=7)

        free_names = ef.FREE_PARAMS_DEFAULT_ORDER
        focus = self.param_focus_var.get()
        if focus == "(auto)":
            focus_name = snap.get("param") or free_names[0]
        else:
            focus_name = focus
        y = np.array([h["params"][focus_name] for h in history])
        self.ax_params.plot(rounds, y, "-", lw=1.3, color="darkviolet", label=focus_name)
        # Mark the steps where THIS parameter was the one actually being
        # updated (every other step just carries its value forward unchanged).
        own_steps = [h["round"] for h in history if h.get("param") == focus_name]
        own_vals = [h["params"][focus_name] for h in history if h.get("param") == focus_name]
        self.ax_params.plot(own_steps, own_vals, "o", ms=4, color="darkviolet")
        self.ax_params.axhline(joint_fit["params_dict"][focus_name], color="0.4", lw=1, ls=":",
                                label=f"joint least_squares ({joint_fit['params_dict'][focus_name]:.4g})")
        self.ax_params.axvline(best_round, color="0.5", lw=1, ls="--")
        self.ax_params.axvline(snap["round"], color="steelblue", lw=1.5, ls="-")
        self.ax_params.set_xlabel("step"); self.ax_params.set_ylabel(f"{focus_name} (own units)")
        self.ax_params.set_title(f"Parameter trajectory -- {focus_name}"
                                  + (" (auto: step's own param)" if focus == "(auto)" else ""))
        self.ax_params.legend(fontsize=7)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        p, jp = display_params, joint_fit["params_dict"]
        param_txt = "  ".join(f"{n}={p[n]:.4g}(joint={jp[n]:.4g})" for n in free_names)
        snap_rmse = float(np.sqrt(np.mean((I_model - I_w) ** 2)))
        which_param = snap.get("param")
        step_detail = (f" (updated {which_param}: {snap['extracted_data']:.4g} vs {snap['extracted_model']:.4g} "
                        f"-> {which_param} {snap['old_value']:.4g}->{snap['new_value']:.4g})"
                        if which_param is not None else "")
        manual_tag = " [MANUAL EDIT -- not part of extract_fit's own history]" if self._manual_override is not None else ""
        self.round_label_var.set(f"Step {snap['round']}/{len(history) - 1}  (lowest rmse at step {best_round}){manual_tag}")
        self.status_var.set(
            f"Cycle {cycle}: seeded from {self._seed_label}  |  "
            f"step {snap['round']} rmse={snap_rmse:.4g} (diagnostic only){step_detail}{manual_tag}  |  "
            f"final rmse={result['rmse']:.4g}  vs. joint rmse={joint_fit['rmse']:.4g}\n{param_txt}")

    def _on_run_chain(self):
        """Runs extract_fit.chain_half_cycles() over N consecutive REAL
        logical cycles -- genuine fit_region_pair() least_squares per
        half-cycle, not extract_fit's own per-parameter correction (by
        request, 2026-10-14, the "smoking gun" real-data validation)."""
        self._mode = "chain"
        try:
            start_cycle = int(self.chain_start_var.get().strip())
            n_cycles = max(1, int(self.chain_n_var.get().strip()))
            bound_pct = float(self.chain_bound_pct_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid chain settings", "Start cycle, # cycles, and bound % must be numbers.")
            return
        try:
            (curr, cyc, time_, volt), bounds = self.app._get_raw_data()
        except Exception as e:
            messagebox.showerror("Can't load raw data", f"{self.app.data_path}:\n{e}")
            return
        cycles_available = sorted(bounds)
        if start_cycle not in cycles_available:
            messagebox.showwarning("Cycle not found", f"Cycle {start_cycle} not present in this data.")
            return
        self.status_var.set(f"Running chain_half_cycles() over {n_cycles} real cycles starting at "
                             f"{start_cycle} (genuine least_squares per half-cycle, this takes a while)...")
        self.canvas.draw_idle()
        self.frame.update_idletasks()
        t0 = time.time()
        try:
            results = ef.chain_half_cycles(time_, volt, curr, bounds, cycles_available, start_cycle, n_cycles,
                                            ron_roff_max_change_frac=bound_pct / 100.0)
        except Exception as e:
            messagebox.showerror("Chain failed", str(e))
            return
        elapsed = time.time() - t0
        self._chain_results = results
        self._redraw_chain(elapsed)

    def _on_save_chain_csv(self):
        """Saves the last chain run via extract_fit.chain_results_to_dataframe()
        -- the SAME function fit_chain_cli.py's own CLI uses, so a GUI-run
        chain and a CLI-run chain produce identical CSV columns."""
        if self._mode != "chain" or not self._chain_results:
            messagebox.showwarning("No chain result", "Run a chain first (\"Run chain (real "
                                    "least_squares)\").")
            return
        default_name = f"chain_{self._chain_results[0]['cycle']}_{self._chain_results[-1]['cycle']}.csv"
        path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(self.app.data_path), initialfile=default_name,
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        df = ef.chain_results_to_dataframe(self._chain_results)
        df.to_csv(path, index=False)
        self.status_var.set(f"Saved {len(df)} half-cycle rows to {path}")

    def _redraw_chain(self, elapsed):
        results = self._chain_results
        for ax in (self.ax_iv, self.ax_it, self.ax_rmse, self.ax_params):
            ax.clear()

        t0g = results[0]["t_w"][0]
        for r in results:
            t_rel = r["t_w"] - t0g
            color = "crimson" if r["kind"] == "max" else "dodgerblue"
            self.ax_iv.plot(r["V_w"], r["I_w"], ".", ms=2, color="0.5")
            self.ax_iv.plot(r["V_w"], r["I_model"], "-", lw=1, color=color)
            self.ax_it.plot(t_rel, r["I_w"], ".", ms=2, color="0.5")
            self.ax_it.plot(t_rel, r["I_model"], "-", lw=1, color=color)
        self.ax_iv.set_xlabel("V (V)"); self.ax_iv.set_ylabel("I (A)")
        self.ax_iv.set_title(f"I-V -- {len(results)} half-cycles, actual (gray) vs model "
                              f"(red=LRS/max, blue=HRS/min)")
        self.ax_it.set_xlabel("t (s)"); self.ax_it.set_ylabel("I (A)")
        self.ax_it.set_title("I(t), stitched across the chain")

        rounds = list(range(len(results)))
        rmses = [r["rmse"] for r in results]
        self.ax_rmse.plot(rounds, rmses, "o-", ms=3, color="darkorange")
        self.ax_rmse.set_yscale("log")
        self.ax_rmse.set_xlabel("half-cycle step (chain order)"); self.ax_rmse.set_ylabel("rmse (log)")
        self.ax_rmse.set_title("Per-half-cycle rmse (genuine least_squares, real data)")

        focus = self.param_focus_var.get()
        if focus == "(auto)":
            ron_vals = [r["params_dict"]["ron"] for r in results]
            roff_vals = [r["params_dict"]["roff"] for r in results]
            self.ax_params.plot(rounds, ron_vals, "o-", ms=3, label="ron")
            self.ax_params.plot(rounds, roff_vals, "s-", ms=3, label="roff")
            self.ax_params.set_title("ron/roff trajectory (auto -- pick a specific param above for any other one)")
        else:
            vals = [r["params_dict"][focus] for r in results]
            self.ax_params.plot(rounds, vals, "o-", ms=3, color="darkviolet", label=focus)
            self.ax_params.set_title(f"{focus} trajectory across the chain")
        self.ax_params.set_yscale("log")
        self.ax_params.set_xlabel("half-cycle step (chain order)"); self.ax_params.set_ylabel("value (log)")
        self.ax_params.legend(fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        n_logical = len({r["cycle"] for r in results})
        rate = len(results) / elapsed if elapsed > 0 else float("inf")
        self.round_label_var.set(f"Chain: {n_logical} real cycles, {len(results)} half-cycles")
        self.status_var.set(
            f"Chain from cycle {results[0]['cycle']}: {n_logical} real experimental cycles "
            f"({len(results)} half-cycles) fit in {elapsed:.1f}s ({rate:.1f} half-cycles/s, "
            f"{rate / 2:.1f} cycles/s)  |  rmse median={np.median(rmses):.4g}  "
            f"min={min(rmses):.4g}  max={max(rmses):.4g}")

class App:
    def __init__(self, root, csv_path, data_path=DATA_PATH):
        self.root = root
        self.csv_path = csv_path
        self.data_path = data_path
        root.title("rolling_fit.csv analyzer")

        self.df_raw = pd.DataFrame()
        self.df_all_good = pd.DataFrame()
        self.df = pd.DataFrame()
        self._raw = None     # lazy-loaded (curr, cyc, time_, volt), see _get_raw_data()
        self._bounds = None  # lazy-loaded cycle_bounds(cyc), paired with self._raw

        top = ttk.Frame(root)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Refresh data", command=self._refresh).pack(side="left")
        self.exclude_bad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Exclude bad windows", variable=self.exclude_bad_var,
                         command=self._apply_filter).pack(side="left", padx=12)

        ttk.Label(top, text="Cycle from:").pack(side="left", padx=(12, 2))
        self.cycle_min_var = tk.StringVar(value="")
        e_min = ttk.Entry(top, textvariable=self.cycle_min_var, width=8)
        e_min.pack(side="left")
        e_min.bind("<Return>", lambda e: self._apply_filter())
        e_min.bind("<FocusOut>", lambda e: self._apply_filter())

        ttk.Label(top, text="to:").pack(side="left", padx=(8, 2))
        self.cycle_max_var = tk.StringVar(value="")
        e_max = ttk.Entry(top, textvariable=self.cycle_max_var, width=8)
        e_max.pack(side="left")
        e_max.bind("<Return>", lambda e: self._apply_filter())
        e_max.bind("<FocusOut>", lambda e: self._apply_filter())

        ttk.Button(top, text="Reset range", command=self._reset_range).pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=12)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.tab1 = EvolutionTab(notebook, self)
        self.tab2 = ScatterTab(notebook, self)
        self.tab3 = DistributionTab(notebook, self)
        self.tab4 = ParamTableTab(notebook, self)
        self.tab5 = CurveInspectorTab(notebook, self)
        self.tab6 = FuzzyExtractorTab(notebook, self)
        notebook.add(self.tab1.frame, text="Evolution vs cycle")
        notebook.add(self.tab2.frame, text="Parameter vs parameter")
        notebook.add(self.tab3.frame, text="Distribution")
        notebook.add(self.tab4.frame, text="Parameter table")
        notebook.add(self.tab5.frame, text="Curve inspector")
        notebook.add(self.tab6.frame, text="Fuzzy extractor")

        self._refresh()

    def _refresh(self):
        try:
            self.df_raw = read_csv_tolerant(self.csv_path)
        except Exception as e:
            messagebox.showerror("Can't read CSV", f"{self.csv_path}:\n{e}")
            return
        self._apply_filter()

    def _reset_range(self):
        self.cycle_min_var.set("")
        self.cycle_max_var.set("")
        self._apply_filter()

    @staticmethod
    def _parse_int_or_none(s):
        s = s.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None  # blank/invalid both just mean "no limit on this end"

    def _apply_filter(self):
        # Bad-excluded (or not, per the checkbox) but NOT cycle-range-filtered --
        # cycle_to_cycle_deltas() needs this to see a cycle's true previous
        # cycle even when that one falls just outside the display range below.
        self.df_all_good = collapse_to_per_cycle(self.df_raw, self.exclude_bad_var.get())
        df = self.df_all_good
        lo = self._parse_int_or_none(self.cycle_min_var.get())
        hi = self._parse_int_or_none(self.cycle_max_var.get())
        if lo is not None and not df.empty:
            df = df[df["cycle_c"] >= lo]
        if hi is not None and not df.empty:
            df = df[df["cycle_c"] <= hi]
        self.df = df.reset_index(drop=True)

        n_bad = int(self.df_raw["bad"].sum()) if "bad" in self.df_raw.columns else 0
        range_txt = f" (cycle {lo if lo is not None else 'min'}-{hi if hi is not None else 'max'})" \
            if (lo is not None or hi is not None) else ""
        self.status_var.set(f"{len(self.df_raw)} windows loaded ({n_bad} bad) "
                             f"-> {len(self.df)} cycles shown{range_txt} from {self.csv_path}")
        self.tab1.on_data_changed()
        self.tab2.on_data_changed()
        self.tab3.on_data_changed()
        self.tab4.on_data_changed()
        self.tab5.on_data_changed()
        self.tab6.on_data_changed()

    def _get_raw_data(self):
        """Lazily loads+caches the real measurement data and its cycle
        index -- only needed once "Show actual vs. model" is first
        clicked, so a plain CSV-browsing session never pays for it."""
        if self._raw is None:
            curr, cyc, time_, volt = load_data(self.data_path)
            self._raw = (curr, cyc, time_, volt)
            self._bounds = cycle_bounds(cyc)
        return self._raw, self._bounds

    def _merged_params(self, cycle, cycles_available):
        """Params for `cycle`'s own whole-cycle simulation, PATCHING its
        row's own RESET_SIDE with the NEXT cycle's row, when available
        AND that next cycle is truly ADJACENT (cycle+1 -- see the
        2026-10-12 fix below for why this guard matters).

        Why: fit_all_cycles.py tags each window's row with the CYCLE OF
        ITS THIRD POINT. For cycle N, the window that freshly fits
        SET_SIDE (using data up to N's own reset) is tagged cycle_c=N --
        correct. But the window that freshly fits RESET_SIDE for N's OWN
        reset event only finishes at N's own reset -> (N+1)'s own set,
        so it's tagged cycle_c=N+1, one cycle LATER. So "cycle N"'s own
        row (collapse_to_per_cycle's pick) carries RESET_SIDE from the
        PREVIOUS reset's fit, not N's own -- verified directly against
        the CSV (2026-10-09): a real, if usually small (~1-2%), staleness.
        Patching in (N+1)'s row's RESET_SIDE fixes it; SET_SIDE needs no
        such patch, it's already correctly fresh in row N itself.

        Fixed (2026-10-12, after the user spotted cycle 2173's vr looking
        wrong): this used to patch from cycles_available[idx+1]
        unconditionally -- "the next AVAILABLE cycle", not necessarily
        N+1 itself. When N's own SET fit fails and the chain's skip-ahead
        recovery discards one or more cycles entirely (none of them get
        a row at all, or get one that's bad and excluded), "the next
        available cycle" can be several cycles later -- confirmed
        directly on cycle 2173: its own SET fit capped out, cycles 2174
        and 2175 were both lost to cascading restart failures, so the
        next available cycle was 2176, and its RESET_SIDE (vr=-0.413,
        etar=148.25, actually fit for CYCLE 2175's own reset) was being
        silently presented as if it were cycle 2173's own -- a value
        that was never fit against cycle 2173's own data at all. Now
        only patches when cycles_available[idx+1] == cycle+1 (a real,
        adjacent reset fit available); otherwise falls back to row's own
        (one-cycle-stale, same as the pre-2026-10-09 behavior) RESET_SIDE,
        which is at least actually in the right neighborhood."""
        idx = cycles_available.index(cycle)
        row = self.df[self.df["cycle_c"] == cycle].iloc[-1]
        d = {name: float(row[name]) for name in dmm.PARAM_NAMES}
        if idx + 1 < len(cycles_available) and cycles_available[idx + 1] == cycle + 1:
            next_row = self.df[self.df["cycle_c"] == cycles_available[idx + 1]].iloc[-1]
            for name in RESET_SIDE:
                d[name] = float(next_row[name])
        return dmm.pack(d)

    def simulate_cycle_window(self, cycle, n_neighbors):
        """Shared by show_cycle_vs_model() and CurveInspectorTab: simulates
        `cycle` PLUS `n_neighbors` cycles before/after (by position in the
        currently filtered self.df's own cycle list), with g walked
        CONTINUOUSLY across them via one extra, unplotted ANCHOR cycle
        just before the window (see the long explanation below -- moved
        here unchanged from show_cycle_vs_model's own former docstring).

        By request (2026-10-09), after the user reported the FIRST PART
        of each plotted cycle looking wrong: each row's own `g0` column
        is NOT the device's g at that whole cycle's own start (V~=0, per
        cycle_bounds()) -- it's "g carried over from the CHAIN's own
        window-boundary point" (see fit_all_cycles.py), which for most
        rows is an EARLIER point in real time than this cycle's own
        start. Using it directly as this cycle's own g0 skips real
        intervening relaxation and starts the simulation from the wrong
        state -- verified directly (2026-10-09): using the stored g0
        gave RMSE=2.8e-3 over a cycle's first 30 samples; properly
        carrying g forward from the END of the PREVIOUS cycle's own
        simulation gave RMSE=5.5e-5, ~50x better, and 44% better over the
        WHOLE cycle too. So here, g is walked CONTINUOUSLY: the anchor
        cycle's own row's stored g0 is the only starting point available
        that far back, but any error in it decays well within one full
        cycle's worth of relaxation -- and every subsequent cycle --
        anchor included -- hands its own simulation's FINAL g to the next
        one via g0_override, instead of each cycle re-reading its own
        (chain-bookkeeping, not whole-cycle-boundary) g0 column. See
        _merged_params() for the smaller, separate RESET_SIDE-staleness
        fix applied to each cycle's parameters.

        Returns None if `cycle` has no row in self.df at all, else a dict
        with cycles_available, window_cycles, anchor_cycle, n_missing, and
        results -- a list of dicts {cycle, t, V, I, I_model, g_hist, rmse,
        vs, vr, Is, Ir}, one per successfully-simulated cycle in
        window_cycles (in order, g-continuous)."""
        cycles_available = sorted(self.df["cycle_c"].unique())
        if cycle not in cycles_available:
            return None
        idx = cycles_available.index(cycle)
        lo = max(0, idx - n_neighbors)
        hi = min(len(cycles_available) - 1, idx + n_neighbors)
        window_cycles = cycles_available[lo:hi + 1]
        anchor_cycle = cycles_available[lo - 1] if lo > 0 else None

        (curr, cyc, time_, volt), bounds = self._get_raw_data()
        is_chain = "kind" in self.df_raw.columns

        def simulate_cycle_chain(c, g0_override):
            """Chain-CSV version of simulate_cycle() (by request 2026-10-17,
            "a single call cannot work, because we're changing the values
            of the parameters! We have, in fact, three different sets:
            {S0,R0}, {S1,R0}, {S1,R1}" -- confirmed directly: a single
            dmm.simulate() call over the whole cycle using ONE merged
            param set applies the FINAL (post-RESET-fit) RESET_SIDE value
            R1 for the ENTIRE cycle, including the SET portion, where the
            real chain fit actually used the OLDER R0 (RESET_SIDE held
            FIXED at its pre-this-cycle value while SET_SIDE was being
            refit) -- a real physical difference (confirmed: g at the
            SET/RESET boundary came out 0.483 from the wrong merged-call
            reconstruction vs. the chain's own true 0.260), not a g0/
            anchor-continuity artifact. _merged_params()/the old
            simulate_cycle() below are correct ONLY for fit_all_cycles.py's
            own one-row-per-cycle CSVs, where there IS only one state
            snapshot per cycle; chain CSVs have TWO (one row per half-
            cycle, each with its own self-consistent {S,R} pair exactly as
            fit), so reconstruction here runs the SAME TWO dmm.simulate()
            calls the real extraction did -- 'max' row's own full
            params_dict over its own i_lo:i_hi window, then 'min' row's
            own full params_dict over ITS window, continuing g from the
            first call's end into the second's start -- instead of one
            merged call over the combined window."""
            max_rows = self.df_raw[(self.df_raw["cycle_c"] == c) & (self.df_raw["kind"] == "max")]
            min_rows = self.df_raw[(self.df_raw["cycle_c"] == c) & (self.df_raw["kind"] == "min")]
            if max_rows.empty or min_rows.empty:
                return None
            max_row, min_row = max_rows.iloc[-1], min_rows.iloc[-1]
            i_lo1, i_hi1 = int(max_row["i_lo"]), int(max_row["i_hi"])
            i_lo2, i_hi2 = int(min_row["i_lo"]), int(min_row["i_hi"])
            t1, V1, I1 = time_[i_lo1:i_hi1 + 1], volt[i_lo1:i_hi1 + 1], curr[i_lo1:i_hi1 + 1]
            t2, V2, I2 = time_[i_lo2:i_hi2 + 1], volt[i_lo2:i_hi2 + 1], curr[i_lo2:i_hi2 + 1]
            params1 = dmm.pack({name: float(max_row[name]) for name in dmm.PARAM_NAMES})
            params2 = dmm.pack({name: float(min_row[name]) for name in dmm.PARAM_NAMES})
            # No anchor available (very first cycle shown): fall back to THIS
            # half's own recorded 'g0' column -- which, for a chain CSV, IS the
            # true g0_override the real fit started this exact half from (see
            # extract_fit.chain_half_cycles()) -- not a merged/guessed value.
            g0_1 = g0_override if g0_override is not None else float(max_row["g0"])
            with np.errstate(over="ignore"):
                I_model1, g_hist1, _ = dmm.simulate(t1, V1, params1, g0_override=g0_1)
                I_model2, g_hist2, _ = dmm.simulate(t2, V2, params2, g0_override=float(g_hist1[-1]))
            t = np.concatenate([t1, t2]); V = np.concatenate([V1, V2]); I = np.concatenate([I1, I2])
            I_model = np.concatenate([I_model1, I_model2]); g_hist = np.concatenate([g_hist1, g_hist2])
            vs_param, vr_param = float(max_row["vs"]), float(min_row["vr"])
            i_vs = frp.estimate_threshold_from_d2idt2(t, V, I, 0.2, "set")
            i_vr = frp.estimate_threshold_from_d2idt2(t, V, I, -0.2, "reset")
            vs = float(V[i_vs]) if i_vs is not None else vs_param
            vr = float(V[i_vr]) if i_vr is not None else vr_param
            Is = float(I[i_vs]) if i_vs is not None else float("nan")
            Ir = float(I[i_vr]) if i_vr is not None else float("nan")
            return t, V, I, I_model, g_hist, vs, vr, Is, Ir

        def simulate_cycle(c, g0_override):
            """One cycle's own (t, V, I, I_model, g_hist, vs, vr, Is, Ir)
            -- None if `c` isn't in the raw data at all. vs/vr are this
            cycle's own EXTRACTED values (by request, 2026-10-12, cycle
            3039 investigation -- see RULES_DOC.md): the first positive
            peak of d2I/dt2 on the real DATA curve, V<-0.2 (vr) / V>0.2
            (vs) -- NOT the bare model parameter's own numeric value,
            which (confirmed on real data) can sit far from where the
            current visibly transitions, for physically real reasons (the
            self-quenching g^gamma term) that still don't make it a
            sensible thing to mark on a plot. Falls back to the bare
            model parameter if no positive peak is found (e.g. too few
            samples on one side) -- Is/Ir are then NaN too, since there's
            no real sample index to read a current off of.

            Is/Ir (by request, 2026-10-14): the real DATA current AT that
            same extracted sample (I[i_vs]/I[i_vr]) -- not a modeled
            quantity. The model's own vs/vr gate VA (the voltage ACROSS
            just the fast element), not the externally measured V, which
            includes the series-resistance drop V = VA + K(ron,roff,g)*I
            (see dmm_model.py's own module docstring) -- so the raw
            extracted vs/vr (measured V) and the fitted model's vs/vr
            (VA) are EXPECTED to differ by roughly that IR drop, not a
            sign of mismatched windows. Is/Ir let that drop be read
            directly off the plot (drop ~= Ron/Roff * I at the marked
            point) instead of only inferred from the vs_vr_gap column."""
            if c not in bounds:
                return None
            i0, i1 = bounds[c]
            t, V, I = time_[i0:i1], volt[i0:i1], curr[i0:i1]
            params = self._merged_params(c, cycles_available)
            vs_param = float(params[dmm.PARAM_NAMES.index("vs")])
            vr_param = float(params[dmm.PARAM_NAMES.index("vr")])
            i_vs = frp.estimate_threshold_from_d2idt2(t, V, I, 0.2, "set")
            i_vr = frp.estimate_threshold_from_d2idt2(t, V, I, -0.2, "reset")
            vs = float(V[i_vs]) if i_vs is not None else vs_param
            vr = float(V[i_vr]) if i_vr is not None else vr_param
            Is = float(I[i_vs]) if i_vs is not None else float("nan")
            Ir = float(I[i_vr]) if i_vr is not None else float("nan")
            with np.errstate(over="ignore"):
                I_model, g_hist, _ = dmm.simulate(t, V, params, g0_override=g0_override)
            return t, V, I, I_model, g_hist, vs, vr, Is, Ir

        simulate_cycle_fn = simulate_cycle_chain if is_chain else simulate_cycle

        g_carry = None  # None => let the first simulated cycle use its own row's g0
        if anchor_cycle is not None:
            anchor = simulate_cycle_fn(anchor_cycle, None)
            if anchor is not None:
                g_carry = float(anchor[4][-1])  # end-of-anchor g, carried into the plotted window

        results, n_missing = [], 0
        for c in window_cycles:
            result = simulate_cycle_fn(c, g_carry)
            if result is None:
                n_missing += 1
                continue
            t, V, I, I_model, g_hist, vs, vr, Is, Ir = result
            g_carry = float(g_hist[-1])  # hand off to the NEXT plotted cycle
            rmse = float(np.sqrt(np.mean((I_model - I) ** 2)))
            results.append(dict(cycle=c, t=t, V=V, I=I, I_model=I_model, g_hist=g_hist, rmse=rmse,
                                 vs=vs, vr=vr, Is=Is, Ir=Ir))

        return dict(cycles_available=cycles_available, window_cycles=window_cycles,
                     anchor_cycle=anchor_cycle, n_missing=n_missing, results=results)

    def show_cycle_vs_model(self, cycle):
        """Opens a new popup window (a plain matplotlib figure, same
        convention as gui_view.py's own state-change/return-map popups)
        with I-V and I(t), actual vs. model, for `cycle` PLUS
        N_NEIGHBOR_CYCLES cycles before/after -- color-coded, the selected
        cycle's curve drawn bolder. See simulate_cycle_window() for the
        actual simulation/g-continuity details."""
        try:
            window = self.simulate_cycle_window(cycle, N_NEIGHBOR_CYCLES)
        except Exception as e:
            messagebox.showerror("Can't load raw data", f"{self.data_path}:\n{e}")
            return
        if window is None:
            messagebox.showwarning("Cycle not found",
                                    f"Cycle {cycle} has no fitted-parameter row in the currently "
                                    f"shown data (check filters / Refresh).")
            return
        window_cycles, anchor_cycle = window["window_cycles"], window["anchor_cycle"]
        results, n_missing = window["results"], window["n_missing"]

        fig, (ax_iv, ax_it) = plt.subplots(1, 2, figsize=(13, 5.5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(window_cycles)))
        color_by_cycle = dict(zip(window_cycles, colors))
        rmses, t0_global = [], None
        plotted_cycles, plotted_colors = [], []
        for r in results:
            c, t, V, I, I_model = r["cycle"], r["t"], r["V"], r["I"], r["I_model"]
            color = color_by_cycle[c]
            plotted_cycles.append(c)
            plotted_colors.append(color)
            rmses.append(r["rmse"])
            if t0_global is None:
                t0_global = t[0]
            t_rel = t - t0_global
            is_selected = c == cycle
            lw = 2.0 if is_selected else 1.0
            label = f"cycle {c}" + (" (selected)" if is_selected else "")

            ax_iv.plot(V, I, ".", ms=3, color=color, alpha=0.6)
            ax_iv.plot(V, I_model, "-", lw=lw, color=color, label=label)
            ax_it.plot(t_rel, I, ".", ms=3, color=color, alpha=0.6)
            ax_it.plot(t_rel, I_model, "-", lw=lw, color=color, label=label)
            ax_it.axvline(t_rel[0], color=color, lw=0.6, ls=":")

        ax_iv.set_xlabel("V (V)"); ax_iv.set_ylabel("I (A)")
        ax_iv.set_title("I-V"); ax_iv.legend(fontsize=7, ncol=2)
        ax_it.set_xlabel("t (s)"); ax_it.set_ylabel("I (A)")
        ax_it.set_title("I(t)  (dots=actual, lines=model; dotted verticals mark cycle starts)")
        ax_it.legend(fontsize=7, ncol=2)

        missing_txt = f", {n_missing} cycle(s) missing from raw data" if n_missing else ""
        anchor_txt = f", g primed from cycle {anchor_cycle}" if anchor_cycle is not None else \
            " (no anchor cycle available -- first cycle shown uses its own stored g0)"
        fig.suptitle(f"cycles {window_cycles[0]}-{window_cycles[-1]} (selected: {cycle}): actual "
                     f"vs. model, mean rmse={np.mean(rmses):.4g}{missing_txt}{anchor_txt}")
        fig.tight_layout()
        fig.show()

        if plotted_cycles:
            self._show_param_table(plotted_cycles, plotted_colors, cycle, window["cycles_available"])

    def _show_param_table(self, plotted_cycles, plotted_colors, selected_cycle, cycles_available):
        """Opens a SECOND popup window (by request, 2026-10-11, alongside
        show_cycle_vs_model()'s own I-V/I(t) popup) with a plain table of
        all 12 DMM parameters (rows) for each of that popup's own
        plotted cycles (columns) -- the exact same MERGED params
        (_merged_params()'s RESET_SIDE-staleness fix included) each
        cycle was actually simulated with, so the table and the curves
        it accompanies never disagree. Each column header is
        color-matched to that cycle's own curve color in the other
        popup; the selected cycle's column is additionally bolded and
        outlined."""
        rows = dmm.PARAM_NAMES
        cell_text = []
        for pname in rows:
            row_vals = []
            for c in plotted_cycles:
                d = dmm.unpack(self._merged_params(c, cycles_available))
                row_vals.append(f"{d[pname]:.4g}")
            cell_text.append(row_vals)

        n_cols = len(plotted_cycles)
        fig2, ax2 = plt.subplots(figsize=(max(5.0, 1.3 * n_cols + 1.5), 0.42 * len(rows) + 1.3))
        ax2.axis("off")
        table = ax2.table(cellText=cell_text, rowLabels=rows,
                           colLabels=[f"cycle {c}" for c in plotted_cycles],
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        for col, (c, color) in enumerate(zip(plotted_cycles, plotted_colors)):
            header_cell = table[(0, col)]
            header_cell.set_facecolor(color)
            header_cell.set_alpha(0.55)
            if c == selected_cycle:
                header_cell.set_text_props(weight="bold")
                for row in range(len(rows) + 1):
                    table[(row, col)].set_edgecolor("black")
                    table[(row, col)].set_linewidth(1.8)

        fig2.suptitle(f"Fitted parameters: cycles {plotted_cycles[0]}-{plotted_cycles[-1]} "
                      f"(selected: {selected_cycle})")
        fig2.tight_layout()
        fig2.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--data", default=DATA_PATH, help="real measurement data, for the "
                     "'actual vs. model' popup (lazy-loaded on first use)")
    args = ap.parse_args()

    root = tk.Tk()
    App(root, args.csv, args.data)
    root.mainloop()


if __name__ == "__main__":
    main()
