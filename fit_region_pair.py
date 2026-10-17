"""Fits dmm_model.py's 12 parameters to real measurement data over a
single window spanning TWO CONSECUTIVE Vreset events -- the selected
cycle's own Vreset plus the FOLLOWING cycle's Vset and Vreset -- as one
joint nonlinear least_squares fit, by request (2026-10-09).

Within one logical cycle (see data_utils.cycle_bounds), V sweeps a full
loop starting and ending near 0: 0 -> +Vmax (SET happens on this rising
branch) -> 0 -> -Vmin (RESET happens on this falling branch) -> 0. So in
time order across cycle n and cycle n+1: Vset(n), Vreset(n), Vset(n+1),
Vreset(n+1) -- i.e. exactly ONE Vset sits between two consecutive Vreset
events, namely the NEXT cycle's. The window [Vreset(n) .. Vreset(n+1)]
therefore splits into exactly the two regions asked for:
    region 1 "reset_to_set": Vreset(n) -> Vset(n+1)
        device stays OFF/relaxing (RESET-active while V<0, frozen while
        V>=0 & falling) through Vmin, back up through V=0, up to the
        next SET threshold.
    region 2 "set_to_reset": Vset(n+1) -> Vreset(n+1)
        device stays ON (SET-active while V>=0 & rising, frozen while
        V<0 & rising... -- see dmm_model/gui_view's own region-labeling
        docstrings) through Vmax, back down through V=0, down to the
        next RESET threshold.
This is why "fitting the selected cycle plus the next one" and "a region
from vreset1 to vset1, another from vset1 to vreset2" describe the SAME
thing: cycle n contributes only its tail (Vreset(n) onward); essentially
all of the window's data comes from cycle n+1.

The two regions are fit JOINTLY (one simultaneous least_squares call),
not one after another, because g (the device's internal state) is a
single continuous variable across the Vset1 boundary -- there's no real
discontinuity there for two independent fits to each get their own
starting g for. g0 (one of the 12 parameters) plays that role: it's the
device's state right at the window's own start (Vreset1), not
necessarily pinned to exactly roff's own target.

Usage:
    python3 fit_region_pair.py --cycle 2000
    python3 fit_region_pair.py --cycle 2000 --plot --out dades/region_pair_fits.csv
    python3 fit_region_pair.py --cycle 2000 --fix gam etar   # hold these at NOMINAL
"""
import argparse
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

import dmm_model as dmm
from data_utils import (DATA_PATH, DEFAULT_BOUNDS, load_data, cycle_bounds,
                         moving_average, _fill_nonfinite, smooth_on_uniform_grid)
from scipy.signal import find_peaks

REGION1_COLOR = "#1f77b4"  # reset_to_set
REGION2_COLOR = "#ff7f0e"  # set_to_reset


def estimate_cycle_thresholds(t_seg, V_seg, I_seg, smooth_n=10):
    """Vset/Vreset LOCAL indices within one cycle's own segment, via the
    max(d2I/dV2) estimator (same algorithm as gui_view.App's threshold
    panel: max of the real data's d2I/dV2, restricted to SET-active
    V>0&rising samples for Vset, RESET-active V<0&falling samples for
    Vreset -- see gui_view._estimate_thresholds's own docstring for why
    restricting to each region matters). Returns (i_set, i_reset), either
    None if this cycle has no SET-active or no RESET-active samples."""
    I_sm = moving_average(I_seg, smooth_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        dIdV = np.gradient(I_sm, V_seg)
    dIdV = moving_average(_fill_nonfinite(dIdV), smooth_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2IdV2 = np.gradient(dIdV, V_seg)
    d2IdV2 = _fill_nonfinite(d2IdV2)

    v_rising = np.concatenate(([True], np.diff(V_seg) >= 0))
    set_idx = np.flatnonzero((V_seg > 0) & v_rising)
    reset_idx = np.flatnonzero((V_seg < 0) & ~v_rising)
    i_set = int(set_idx[np.argmax(d2IdV2[set_idx])]) if len(set_idx) else None
    i_reset = int(reset_idx[np.argmax(d2IdV2[reset_idx])]) if len(reset_idx) else None
    return i_set, i_reset


def estimate_cycle_thresholds_extracted(t_seg, V_seg, I_seg, smooth_n=10, v_threshold=0.2):
    """Alternative to `estimate_cycle_thresholds` above, for use as
    `build_threshold_sequence`'s `threshold_fn` (RULES_DOC.md section
    13, by request 2026-10-13): uses the newer, synthetic-validated
    extractors instead of the production max(d2I/dV2) one --
    `estimate_threshold_from_d2idt2` for vs (first positive d2I/dt2
    peak, V>v_threshold) and `estimate_threshold_from_d3_before_d2` for
    vr (max d3I/dt3 at/before max d2I/dt2, V<-v_threshold; nearly
    DOUBLES the synthetic correlation with the true model vr vs. the
    production detector, 0.513 vs 0.445 -- see RULES_DOC.md section 10).

    Motivation: `build_threshold_sequence`'s window boundaries ARE
    already "extracted" thresholds, just from the weaker production
    detector -- and that detector's own peak-of-curvature location
    need not coincide with where the device's own transition truly
    starts. A window boundary placed slightly off from the true onset
    means a few samples on the wrong side of the split get fit with
    the WRONG side's free parameters held fixed (that region's own
    free knobs forced to explain dynamics that actually belong to the
    other side) -- a real, if usually small, model-misspecification at
    every single window boundary, not just noise. Swapping in the
    better-validated extractor here is the direct fix.

    Same (i_set, i_reset) LOCAL-index return convention as
    `estimate_cycle_thresholds` -- a drop-in replacement, either value
    None if that polarity's mask has too few samples or no qualifying
    peak is found in this cycle segment."""
    i_set = estimate_threshold_from_d2idt2(t_seg, V_seg, I_seg, v_threshold, "set", smooth_n)
    i_reset = estimate_threshold_from_d3_before_d2(t_seg, V_seg, I_seg, -v_threshold, "reset", smooth_n)
    return i_set, i_reset


def estimate_threshold_from_d2idt2(t_seg, V_seg, I_seg, v_threshold, side, smooth_n=10):
    """Alternative vs/vr estimator (by request, 2026-10-12, cycle 3039
    investigation -- see RULES_DOC.md): the FIRST positive peak of
    d2I/dt2 (smoothed via smooth_on_uniform_grid, NOT plain moving_average
    over sample count -- real acquisition dt can vary several-fold within
    one window, badly distorting a naive sample-count smoothing of a
    SECOND time-derivative) within V<v_threshold (side="reset",
    v_threshold<0) or V>v_threshold (side="set", v_threshold>0), in time
    order. v_threshold exists specifically to exclude the region right
    around V=0 -- confirmed directly (cycle 3039): the raw current there
    is perfectly LINEAR in V (pure Ohmic, no real kink at all) even
    though the acquisition's own dt happens to jump 2-4x right there
    (instrument dwell/settling time after the device switches), which
    otherwise produces a small, purely-artifactual peak in d2I/dt2 that
    the FIRST-peak rule would latch onto -- v_threshold must be chosen
    far enough from 0 to clear that artifact region (confirmed at
    V<-0.2 / V>0.2 on real data; not otherwise tuned/validated as a
    general threshold).

    Returns a LOCAL index into t_seg/V_seg/I_seg, or None if no positive
    peak is found within the mask (e.g. too few masked samples, or the
    curve never actually accelerates there)."""
    didt = np.gradient(I_seg, t_seg)
    didt_sm = smooth_on_uniform_grid(t_seg, didt, smooth_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2idt2 = np.gradient(didt_sm, t_seg)
    d2idt2_sm = smooth_on_uniform_grid(t_seg, _fill_nonfinite(d2idt2), smooth_n)

    mask = (V_seg < v_threshold) if side == "reset" else (V_seg > v_threshold)
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        return None
    seg = d2idt2_sm[idx]
    rng = seg.max() - seg.min()
    if rng <= 0:
        return None
    pk, _ = find_peaks(seg, prominence=0.02 * rng)
    pk_sorted = pk[np.argsort(idx[pk])]
    for p in pk_sorted:
        if seg[p] > 0:
            return int(idx[p])
    return None


def estimate_threshold_from_d3_before_d2(t_seg, V_seg, I_seg, v_threshold, side, smooth_n=10):
    """Second alternative vs/vr estimator (by request, 2026-10-12): the
    max of d3I/dt3 at or before the max of d2I/dt2 (both smoothed via
    smooth_on_uniform_grid, same convention as
    estimate_threshold_from_d2idt2), within the SAME V<v_threshold
    (side="reset") / V>v_threshold (side="set") mask. d3 (the "jerk")
    tends to peak slightly BEFORE d2 (acceleration) itself peaks, at
    the point where the transition is accelerating FASTEST rather than
    where it's already at its sharpest -- validated via synthetic
    ground truth (RULES_DOC.md): for vr specifically, this nearly
    DOUBLES the correlation with the true model parameter vs. max(d2)
    alone (0.247->0.513, also beating the production max(d2I/dV2)
    detector's own 0.445), though its RAW (uncalibrated) absolute
    accuracy is still worse than that production detector's -- a
    linear calibration (fit vr_pred = a*feature+b against known truth)
    is needed to make the ABSOLUTE value usable, the correlation alone
    isn't enough (see RULES_DOC.md's own calibration discussion for
    why these are different things). NOT separately validated for vs
    -- used there only by symmetry, unconfirmed.

    Returns a LOCAL index, or None if max(d2I/dt2) itself can't be
    found (too few masked samples) or there's nothing before it."""
    didt = np.gradient(I_seg, t_seg)
    didt_sm = smooth_on_uniform_grid(t_seg, didt, smooth_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2idt2 = np.gradient(didt_sm, t_seg)
    d2idt2_sm = smooth_on_uniform_grid(t_seg, _fill_nonfinite(d2idt2), smooth_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        d3idt3 = np.gradient(d2idt2_sm, t_seg)
    d3idt3_sm = smooth_on_uniform_grid(t_seg, _fill_nonfinite(d3idt3), smooth_n)

    mask = (V_seg < v_threshold) if side == "reset" else (V_seg > v_threshold)
    idx = np.flatnonzero(mask)
    if len(idx) < 5:
        return None
    j2 = idx[np.argmax(d2idt2_sm[idx])]
    idx_before = idx[idx <= j2]  # "before" = earlier in TIME, same for either side
    if len(idx_before) < 3:
        return None
    return int(idx_before[np.argmax(d3idt3_sm[idx_before])])


def estimate_vr_from_half_recovery(t_seg, V_seg, I_seg, v_threshold=-0.2):
    """Third alternative vr estimator, RESET side only (by request,
    2026-10-14): finds I_min, the current's own minimum within V<
    v_threshold (the self-quenching "dip" -- reset switches the device
    OFF, raising its resistance, so |I| temporarily overshoots then
    recovers even as |V| keeps growing, see RULES_DOC.md section 9.4),
    then I_at_Vmin, the current at that SAME mask's own voltage minimum
    (argmin(V) -- the sweep's own most-negative point). Di = I_at_Vmin
    - I_min is the RECOVERY actually achieved, within this window, by
    the time V bottoms out -- the target crossing level is the
    MIDPOINT of that observed span (I_min + Di/2), not an arbitrary
    zero-referenced half of I_min (that naive version was checked too
    and is worse, see RULES_DOC.md section 9.4). Walking FORWARD in
    time from I_min, returns the first local index where I_seg
    reaches that midpoint.

    Returns None if V<v_threshold has too few samples, OR if Di<=0 --
    no real recovery is visible by the time V reaches its own minimum
    (common: ~48% of even cleanly-switching synthetic trials, since
    recovery often continues PAST that point, while V is already
    rising back toward 0) -- the caller should fall back to another
    detector in that case, see estimate_vr_seed().

    Validated against synthetic ground truth (dmm.simulate's own true
    VA, not raw V -- see section 13.2 for why that distinction
    matters): corr=0.562 against true vr when it resolves (the best
    correlation found anywhere in this whole investigation, beating
    even section 9.3's calibrated d3I/dV3 feature), but only resolves
    ~52% of the time and has a much worse median relative error
    (153%) than the simpler `estimate_threshold_from_d2idt2` (61%) --
    a better RANK-tracker, not a more accurate one, same trade-off
    pattern documented in section 9.1/9.2."""
    idx = np.flatnonzero(V_seg < v_threshold)
    if len(idx) < 3:
        return None
    j_min_pos = int(np.argmin(I_seg[idx]))
    i_min = I_seg[idx[j_min_pos]]
    i_at_vmin = I_seg[idx[int(np.argmin(V_seg[idx]))]]
    di = i_at_vmin - i_min
    if di <= 0:
        return None
    target = i_min + di / 2.0
    for k in range(j_min_pos, len(idx)):
        if I_seg[idx[k]] >= target:
            return int(idx[k])
    return None


def estimate_vr_seed(t_seg, V_seg, I_seg, v_threshold=-0.2, smooth_n=10):
    """Combined vr SEED for `--seed-vr-vs` (by request, 2026-10-14,
    RULES_DOC.md section 9.4/13.3): `estimate_vr_from_half_recovery`
    first (best rank-correlation against true vr, corr=0.562, but only
    resolves ~52% of the time), falling back to
    `estimate_threshold_from_d2idt2` (corr=0.412, always resolves on a
    clean switch) when the half-recovery one finds no crossing.
    A SEED only -- the caller should set x0_dict['vr'] from this and
    leave 'vr' in the FREE parameter set, never add it to `fixed`
    (that was `fix_vr_vs`'s own mistake -- confirmed catastrophic,
    section 10 -- forcing the fit to satisfy a threshold that can sit
    beyond where the internal VA can ever reach).

    Returns a local index, or None if neither detector resolves
    (caller should then leave vr at its own existing x0 default)."""
    j = estimate_vr_from_half_recovery(t_seg, V_seg, I_seg, v_threshold)
    if j is not None:
        return j
    return estimate_threshold_from_d2idt2(t_seg, V_seg, I_seg, v_threshold, "reset", smooth_n)


def estimate_global_on_fast_params(points, time_, volt, curr, n_segments=200, stride=3):
    """Estimates GLOBAL (ron, roff, aon, aoff, ion, ioff) by pooling the
    frozen ON segment (V>=0 & falling, where g is PROVABLY exactly
    constant) from MANY windows into ONE joint least_squares call --
    all six SHARED across all segments, each segment's own g a free
    SCALAR (not a trajectory, since it's frozen within that one
    segment). The ON segment alone still constrains ALL SIX: this
    segment's own g is NOT always close to 1 (incomplete SET
    saturation is common -- median g at reset onset was found to be
    ~0.41 earlier this session), so pooling segments whose g happens
    to range from near 0 up to 1 gives real leverage on BOTH the ON
    and OFF endpoints, via Rs(g)=ktrans(ron,roff,g) etc.

    By request (2026-10-12, after the "device doesn't care if we call
    it a window, a cycle, or an apple" methodological correction --
    see RULES_DOC.md section 12): independently fitting (Rs,Ka,Ki) per
    window is HIGHLY unstable (Ka/Ki individually swing across their
    ENTIRE bound range between windows, confirmed directly on 480 real
    windows) -- not because of insufficient data, but a genuine
    amplitude/curvature degeneracy within any ONE window's own frozen
    segment. Pooling many segments with g allowed to differ per
    segment (exploiting REAL g variation across windows, instead of
    assuming Ka/Ki themselves are constant) breaks this degeneracy:
    validated on 200 genuinely non-overlapping real segments, this
    converges to sensible, non-bound-hugging values for ALL SIX
    params with only a modest ~37% cost in fit quality vs. giving
    every segment its own fully independent 3 free parameters (pooled
    scaled rmse 3.89 vs. 2.84) -- a world apart from either (a)
    fitting ron/aon/ion SHARED via the full ODE simulation across
    multiple windows (tried first, confirmed catastrophic: 5-15x
    worse, section 11 of this file's own earlier investigation) or
    (b) the unstable independent-per-window version this replaces.

    A SEPARATE attempt pooling the narrow near-V=0 OFF-side window
    instead (with ron/aon/ion held at rough guesses, fitting only
    roff/aoff/ioff) was tried and does NOT work -- pooled rmse ~94
    (vs. this function's own 3.89), g collapsing to near-0 everywhere
    (median 0.066, max 0.24) -- consistent with the OFF side's own
    weaker physical justification. Superseded by this function, which
    gets usable roff/aoff/ioff values from the ON segment alone
    instead, where g ranges widely enough to see both endpoints.

    points: build_threshold_sequence()'s own output. n_segments: how
    many windows (spaced `stride` apart, to avoid pooling the SAME
    physical segment twice from overlapping windows) to pool. Returns
    a dict {ron,roff,aon,aoff,ion,ioff} -- plain floats, meant to be
    held FIXED for an entire chain, not re-estimated per window."""
    segs = []
    for w in range(0, len(points) - 2, stride):
        i_a, _, _ = points[w]
        i_b, _, _ = points[w + 1]
        i_c, _, _ = points[w + 2]
        V_w, I_w = np.asarray(volt[i_a:i_c + 1]), np.asarray(curr[i_a:i_c + 1])
        v_rising_w = np.concatenate(([True], np.diff(V_w) >= 0))
        on_mask = (V_w >= 0) & (~v_rising_w)
        if on_mask.sum() < 30:
            continue
        segs.append((V_w[on_mask], I_w[on_mask]))
        if len(segs) >= n_segments:
            break

    def unpack(x):
        ron_, roff_, aon_, aoff_, ion_, ioff_ = 10 ** x[:6]
        gs_ = np.clip(x[6:], 1e-4, 1 - 1e-4)
        return ron_, roff_, aon_, aoff_, ion_, ioff_, gs_

    def residuals(x):
        ron_, roff_, aon_, aoff_, ion_, ioff_, gs_ = unpack(x)
        all_r = []
        for (V_seg, I_seg), g in zip(segs, gs_):
            Rs = dmm.ktrans(ron_, roff_, g)
            Ka = dmm.ktrans(aon_, aoff_, g)
            Ki = dmm.ktrans(ion_, ioff_, g)
            VA = np.arcsinh(I_seg / Ki) / Ka
            r = VA + Rs * I_seg - V_seg
            scale = 0.02 * np.max(np.abs(I_seg)) + 1e-15
            all_r.append(r / scale)
        return np.concatenate(all_r)

    N = len(segs)
    x0 = np.array([np.log10(25.0), np.log10(30.0), np.log10(2.0), np.log10(2.0),
                   np.log10(0.5), np.log10(0.01)] + [0.5] * N)
    lo = [np.log10(1.0), np.log10(1.0), np.log10(0.3), np.log10(0.3),
          np.log10(1e-4), np.log10(1e-6)] + [1e-4] * N
    hi = [np.log10(500.0), np.log10(500.0), np.log10(10.0), np.log10(10.0),
          np.log10(100.0), np.log10(10.0)] + [1 - 1e-4] * N
    res = least_squares(residuals, x0, bounds=(lo, hi), method="trf", max_nfev=30000)
    ron, roff, aon, aoff, ion, ioff = (10 ** res.x[:6]).tolist()
    return dict(ron=ron, roff=roff, aon=aon, aoff=aoff, ion=ion, ioff=ioff)


def find_fit_window(bounds_by_cycle, cycles, selected_cycle, time_, volt, curr, smooth_n=10):
    """Locates the three GLOBAL sample indices (i_vreset1, i_vset1,
    i_vreset2) bounding the fit window for `selected_cycle` -- its own
    Vreset, then the NEXT cycle's Vset and Vreset (see module docstring).
    Raises ValueError with a specific reason if `selected_cycle` has no
    following cycle, either cycle is missing a threshold point, or the
    three points don't come out in time order (an unexpected cycle
    shape -- not silently fit anyway)."""
    if selected_cycle not in bounds_by_cycle:
        raise ValueError(f"cycle {selected_cycle} not present in this data")
    pos = cycles.index(selected_cycle)
    if pos + 1 >= len(cycles):
        raise ValueError(f"cycle {selected_cycle} has no following cycle in this data")
    next_cycle = cycles[pos + 1]

    i0_sel, i1_sel = bounds_by_cycle[selected_cycle]
    i0_next, i1_next = bounds_by_cycle[next_cycle]

    _, i_reset_sel = estimate_cycle_thresholds(time_[i0_sel:i1_sel], volt[i0_sel:i1_sel],
                                                curr[i0_sel:i1_sel], smooth_n)
    i_set_next, i_reset_next = estimate_cycle_thresholds(time_[i0_next:i1_next], volt[i0_next:i1_next],
                                                          curr[i0_next:i1_next], smooth_n)
    if i_reset_sel is None:
        raise ValueError(f"cycle {selected_cycle} has no RESET-active samples -- can't find Vreset1")
    if i_set_next is None or i_reset_next is None:
        raise ValueError(f"cycle {next_cycle} (the one after {selected_cycle}) is missing a "
                          "SET-active or RESET-active sample -- can't find Vset1/Vreset2")

    i_vreset1 = i0_sel + i_reset_sel
    i_vset1 = i0_next + i_set_next
    i_vreset2 = i0_next + i_reset_next
    if not (i_vreset1 < i_vset1 < i_vreset2):
        raise ValueError("threshold points came out of time order "
                          f"(Vreset1={i_vreset1}, Vset1={i_vset1}, Vreset2={i_vreset2}) -- "
                          "unexpected cycle shape, inspect this cycle pair by hand before fitting.")
    return dict(selected_cycle=selected_cycle, next_cycle=next_cycle,
                i_vreset1=i_vreset1, i_vset1=i_vset1, i_vreset2=i_vreset2)


def _param_transform(x0_dict, bounds_dict):
    """{name: value}/{name: (lo,hi)} -> (x0, lo, hi) arrays in FIT space
    -- log10 for dmm.LOG_PARAMS (positive, multi-decade), linear for
    dmm.LINEAR_PARAMS -- PARAM_NAMES order, matching the convention
    dmm_model.py documents for any fitting code built on it."""
    x0 = np.empty(len(dmm.PARAM_NAMES))
    lo = np.empty(len(dmm.PARAM_NAMES))
    hi = np.empty(len(dmm.PARAM_NAMES))
    for i, name in enumerate(dmm.PARAM_NAMES):
        b_lo, b_hi = bounds_dict[name]
        if name in dmm.LOG_PARAMS:
            lo[i], hi[i], x0[i] = np.log10(b_lo), np.log10(b_hi), np.log10(x0_dict[name])
        else:
            lo[i], hi[i], x0[i] = b_lo, b_hi, x0_dict[name]
    x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)
    return x0, lo, hi


def _from_fit_space(x):
    """FIT space -> real params (length-12 array, PARAM_NAMES order):
    undoes _param_transform's log10 on dmm.LOG_PARAMS."""
    params = np.empty(len(dmm.PARAM_NAMES))
    for i, name in enumerate(dmm.PARAM_NAMES):
        params[i] = 10.0 ** x[i] if name in dmm.LOG_PARAMS else x[i]
    return params


def _measure_near_zero_slope(V_seg, I_seg):
    """Median dI/dV over the 7 samples nearest V=0 within this segment
    (V_seg sorted first) -- a plain, model-independent measurement of
    the near-V=0 I-V slope, used by `derive_ron_roff` below. Returns
    None if there aren't enough samples.

    De-duplicates by V BEFORE sorting/gradient (fixed 2026-10-17, by
    request, after `extract_fit.extract_fit()` was found silently
    freezing ron/aon/ion/aoff/ioff for an entire 250-cycle Monte Carlo
    chain -- traced to THIS function returning None every single time,
    on every tested window: when `V_seg` spans a FULL 0->Vmax->0 (or
    0->Vmin->0) half-cycle, the rising and falling edges revisit the
    SAME quantized near-zero voltage levels (confirmed directly, e.g.
    V=[0.01, 0.01, 0.02, 0.02, 0.03] right at the sorted array's own
    center), so np.diff(Vs)==0 between the duplicates and np.gradient
    divides by that zero spacing -- every one of the 7 nearest-to-zero
    samples came back inf/nan, `seg` filtered to empty, None every time.
    Averaging I across duplicate V's first removes the zero-spacing
    case entirely while leaving the slope estimate itself unchanged on
    any already-working input (strictly monotonic V has no duplicates
    to average away)."""
    if len(V_seg) < 7:
        return None
    order = np.argsort(V_seg)
    Vs, Is = V_seg[order], I_seg[order]
    Vs_u, inverse = np.unique(Vs, return_inverse=True)
    if len(Vs_u) < 7:
        return None
    Is_u = np.bincount(inverse, weights=Is) / np.bincount(inverse)
    with np.errstate(divide="ignore", invalid="ignore"):
        dIdV = np.gradient(Is_u, Vs_u)
    j0 = int(np.argmin(np.abs(Vs_u)))
    lo, hi = max(0, j0 - 3), min(len(Vs_u), j0 + 4)
    seg = dIdV[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.median(seg)) if len(seg) else None


def estimate_rs_half(V_seg, I_seg):
    """Crude, trial-param-free Rs estimate (by request, 2026-10-14):
    HALF of the near-V=0 small-signal resistance (1/slope0, via
    `_measure_near_zero_slope`). From the SAME exact relation
    derive_ron/derive_roff use (see fit_region_pair()'s own docstring):
    dI/dV|_0 = Ka*Ki/(1+Rs*Ka*Ki) => 1/slope0 = Rs + 1/(Ka*Ki) -- the
    measured near-V=0 resistance is the SUM of the series resistance
    Rs=K(ron,roff,g) and the fast element's own internal resistance
    1/(Ka*Ki). derive_ron/derive_roff split this sum EXACTLY, but need
    a TRIAL (Ka,Ki) each fit iteration to do it. This instead just
    ASSUMES the split is 50/50 and returns Rs=0.5/slope0 directly, no
    trial params needed -- motivated by an earlier scratch investigation
    (joint_shared_fast.py) that jointly fit ron/aon/ion across 15 real
    SET windows and found the ron/(ron+1/(aon*ion)) split clusters near
    50/50 on average (not Rs- or element-dominated), making this a
    reasonable, if crude, point estimate -- intended as a SEED (x0),
    not a hard constraint (same lesson as section 10's fix_vr_vs vs.
    section 9.4's seed_vr_vs).

    Returns None if slope0 isn't measurable or comes out non-positive
    (a non-positive near-V=0 slope isn't physically a resistance at
    all -- noise or too few samples)."""
    slope0 = _measure_near_zero_slope(V_seg, I_seg)
    if slope0 is None or slope0 <= 0:
        return None
    return 0.5 / slope0


def fit_region_pair(t, V, I, i_vreset1, i_vset1, i_vreset2,
                     x0_dict=None, bounds_dict=None, fixed=None, max_nfev=2000,
                     g0_override=None, fit_derivative=True, deriv_smooth_n=10,
                     derive_ron=False, derive_roff=False, turning_point_weight=0.0,
                     third_deriv_weight=0.0):
    """Jointly fits all 12 dmm_model parameters (minus any named in
    `fixed`, held at their x0 value) to the real (t, V, I) data over
    [i_vreset1, i_vreset2] inclusive -- BOTH switching regions in ONE
    simultaneous least_squares call (see module docstring for why not
    region-by-region).

    x0_dict: starting guess, {name: value}, defaults to dmm.NOMINAL --
    pass the GUI's current slider values here for a warm start.
    bounds_dict: optimizer bounds, {name: (lo, hi)}, defaults to
    data_utils.DEFAULT_BOUNDS.
    fixed: {name: value} held fixed (not varied) at that value --
    e.g. {"gam": dmm.NOMINAL["gam"]} to fit everything else first.
    g0_override: if given, the window's starting g is pinned to this
    value (passed straight through to dmm.simulate) instead of being the
    'g0' parameter -- for chaining fits across overlapping windows
    (fit_all_cycles.py), where g should continue exactly from the
    previous window's own trajectory rather than be refit from scratch.
    'g0' is then NOT varied (silently excluded from the free set,
    regardless of `fixed`) and the returned params/params_dict report
    g0_override itself as "g0", for an honest record of what was
    actually simulated.

    fit_derivative/deriv_smooth_n (by request, 2026-10-12, after the user
    spotted cycle 8887's fit looking "clearly wrong, with the derivatives
    totally different" despite a passable I(t) rmse): when
    fit_derivative=True (the new default), the residual vector ALSO
    includes dI/dt, not just I itself -- both the real and the model's
    own dI/dt are smoothed with a `deriv_smooth_n`-wide moving_average
    FIRST (same convention as the threshold estimator's own smoothing)
    before differencing, since a raw sample-to-sample finite difference
    is dominated by measurement noise on the real side. The two residual
    groups are each normalized the same way `scale` already normalizes
    the plain-I residual (0.02*max(|signal|) in this window), so they
    contribute comparably rather than one swamping the other by raw
    magnitude. This is a genuine trade-off, not a free improvement:
    verified directly on cycle 8887's own (already-bad) window, adding
    it moved I's own rmse from 0.000766 to 0.000836 (+9%) while dI/dt's
    rmse improved from 0.00307 to 0.00255 (-17%) -- i.e. it really does
    pull the fit toward matching the CURVE'S SHAPE's rate of change, at
    some cost to the raw current match, exactly as intended. Pass
    fit_derivative=False to recover the old I-only behavior exactly.

    Residuals are normalized by 0.02*max(|I| in this window), the same
    scale convention the archived initial_guess.py used, so `ftol`/`xtol`
    mean roughly "2% of this window's own current scale" regardless of
    the device's actual current range. Set to 1e-9 (was 1e-12 until
    2026-10-11): scipy's own `result.nfev` only counts "major"
    evaluations, NOT the extra calls each finite-difference Jacobian
    needs per iteration (confirmed directly: a toy 3-free-param problem
    did 28 real calls for a reported nfev of 7) -- so the real cost of
    grinding to 1e-12 on a handful of ill-conditioned windows was far
    higher than --max-nfev's own number suggested, and was the dominant
    remaining cost even after dmm_model.simulate() itself got Numba-
    jitted (2026-10-11, ~64x faster per call in isolation, but barely
    moved end-to-end wall time until this). 1e-9 is still three orders
    of magnitude tighter than the ~1e-4 rmse scale this is ever actually
    judged against; cut a representative 60-window profiling run's real
    simulate() call count by 37% (713742 -> 451238) and wall time by 38%
    (97.3s -> 60.2s), with rmse on the same windows matching to displayed
    precision (no quality loss observed).

    derive_ron/derive_roff (default False each, by request 2026-10-12):
    instead of fitting ron (or roff) as an independent free parameter,
    DERIVE it each iteration from (aon,ion) [or (aoff,ioff)]'s own
    current trial values plus a near-V=0 slope MEASURED directly from
    this window's own real data (model-independent, computed once up
    front) -- the near-V=0 I-V slope is exactly Ka*Ki/(1+Rs*Ka*Ki)
    (linearizing I=Ki*sinh(Ka*VA) at VA=0; confirmed numerically exact
    earlier this session), so given a measured slope0 and trial
    (Ka,Ki), Rs is fully determined: Rs = 1/slope0 - 1/(Ka*Ki). This
    removes the parameter from the free set entirely (1 fewer free
    param), using a directly-measured constraint rather than letting
    the optimizer explore the Rs/(Ka*Ki) degeneracy (see RULES_DOC.md
    section 6 for why that degeneracy exists and how it was first
    characterized, and section 11 for the validation behind keeping
    these as two SEPARATE flags rather than one combined one).

    derive_ron measures slope0 on the V>=0 & falling "frozen" segment,
    where g is PROVABLY exactly constant -- validated as a genuine,
    costless win: on a real test window, rmse improved slightly
    (0.00024->0.000197) with one fewer free parameter, ron landing
    close to the free fit's own value (24.0 vs 27.5).

    derive_roff is NOT recommended -- CONFIRMED BROKEN on real data
    (rmse 36x worse, sometimes g_range collapsing to 0), kept only so
    the mechanism/finding stays reproducible. No equivalent exact
    -freeze segment exists on the OFF side (RESET is only
    APPROXIMATELY frozen near V=0, never exactly, since it's active
    -- if exponentially suppressed -- for ANY V<0); the measured
    slope0_off can come out badly inconsistent with NOMINAL-range
    (aoff,ioff) starting guesses, driving the derived roff negative,
    clipped to its bound, creating a zero-gradient dead zone the
    optimizer can't escape -- confirmed this isn't just a bad-x0 issue
    (tried a better-guessed x0 too, still broke, worse in fact).
    Falls back to the ordinary free fit for whichever side has too few
    samples to measure a slope at all.

    turning_point_weight (default 0.0 = off, by request 2026-10-12):
    adds TWO extra weighted residual entries comparing model vs. data
    dI/dt at the window's own Vmin (after reset) and Vmax (after set)
    -- specifically at whichever nearby sample has the SMALLEST |dV/dt|
    (searched within +-5 samples of the plain argmin/argmax(V), not
    assumed to BE that sample), since this is where the comparison is
    cleanest. Why: the fast I-V relation I=F(V,g) gives
    dI/dt=(dF/dV)*(dV/dt)+(dF/dg)*(dg/dt) by the chain rule -- at the
    EXACT voltage turning point dV/dt=0, so dI/dt there is PURELY
    (dF/dg)*(dg/dt), with zero contribution from the fast/instantaneous
    Rs/Ka/Ki relation. Confirmed numerically (synthetic ground truth,
    incomplete-saturation case so dg/dt isn't itself already zero from
    switching having finished): at the sample nearest Vmax, the "fast"
    contribution came out to -1e-5 against a total dI/dt of 1.76e-3 --
    essentially 100% slow-dynamics signal, decoupled from the Rs/Ka/Ki
    ambiguities documented in section 6. This ONLY holds cleanly when
    the discrete sample's own |dV/dt| is genuinely small, which isn't
    guaranteed by construction (confirmed: the Vmin side of the SAME
    synthetic trial had a much larger residual dV/dt, purely from how
    this particular sweep happens to be sampled near that extremum, and
    the decoupling was muddier there, ~65% fast-term). Requires
    fit_derivative=True (reuses its own didt_real_smooth/scale_didt).

    third_deriv_weight (default 0.0 = off, by request 2026-10-12):
    another extra weighted residual, this time exploiting WHERE d2I/dt2
    peaks rather than a special VALUE. At the real data's own d2I/dt2
    peak (separately located within the SET-active and RESET-active
    masks -- the same kind of location fuzzy_fit.py's own peak-width
    features look at, but used differently here), the first-order
    condition for a maximum means d3I/dt3=0 there EXACTLY, in the
    continuous limit -- an algebraic constraint that doesn't need a
    hand-derived closed form (the chain-rule expansion of I=F(V,g)'s
    own third time-derivative is messy) since it only needs the
    MODEL's own simulated d3I/dt3, evaluated numerically at that index,
    to be close to zero. No measured/matched VALUE is needed (unlike
    turning_point_weight, which matches model to data) -- this residual
    is just the model's own d3I/dt3 there, driven toward 0.

    Returns a dict: params (length-12 array), params_dict, rmse (whole
    window), rmse_region1/rmse_region2 (same fitted params, scored
    separately per region -- lets you see whether one region is dragging
    the joint fit away from the other), didt_rmse (smoothed dI/dt rmse,
    whole window, same fitted params -- computed and reported regardless
    of fit_derivative, so it's directly comparable whether or not it was
    actually optimized against), I_model/g_hist/va_hist/t/V/I (the
    window's own simulate() output and inputs, for plotting), split
    (the region1/region2 boundary as a LOCAL index into those arrays),
    result (the raw scipy.optimize.OptimizeResult)."""
    if i_vreset1 >= i_vreset2 or not (i_vreset1 <= i_vset1 <= i_vreset2):
        raise ValueError(f"indices must satisfy i_vreset1({i_vreset1}) <= "
                          f"i_vset1({i_vset1}) <= i_vreset2({i_vreset2}), with "
                          f"i_vreset1 < i_vreset2")
    fixed = dict(fixed or {})
    if g0_override is not None:
        fixed["g0"] = g0_override  # excluded from the free set either way; value here is cosmetic
    x0_dict = {**dmm.NOMINAL, **(x0_dict or {})}
    bounds_dict = {**DEFAULT_BOUNDS, **(bounds_dict or {})}

    sl = slice(i_vreset1, i_vreset2 + 1)
    t_w, V_w, I_w = np.asarray(t[sl]), np.asarray(V[sl]), np.asarray(I[sl])
    split = i_vset1 - i_vreset1  # region1 = [0:split), region2 = [split:]

    slope0_on = slope0_off = None
    if derive_ron:
        v_rising_w = np.concatenate(([True], np.diff(V_w) >= 0))
        on_mask = (V_w >= 0) & (~v_rising_w)
        slope0_on = _measure_near_zero_slope(V_w[on_mask], I_w[on_mask])
        derive_ron = slope0_on is not None and slope0_on > 0
    if derive_roff:
        # +-overlap margin past `split` (by request, 2026-10-14, "each set or
        # reset cycle should be able to include a small part from before...
        # if we don't use that, we lose many points in the transition"):
        # `split` is where the DETECTOR placed the Vset1 crossing, not
        # necessarily exactly where the device itself stopped being OFF --
        # without this margin a slightly-early detection silently excludes
        # genuinely-still-OFF samples just past the boundary. NOT
        # independently re-validated against derive_roff's own already-
        # CONFIRMED-BROKEN status (this docstring's own warning above) --
        # same narrow/noisy-segment problem either way, this only widens
        # where that noisy segment is drawn from.
        lo_off, hi_off = 0, min(len(V_w), split + 15)
        off_seg = V_w[lo_off:hi_off]
        if len(off_seg):
            j = lo_off + int(np.argmin(np.abs(off_seg)))
            off_lo2, off_hi2 = max(lo_off, j - 20), min(hi_off, j + 20)
            slope0_off = _measure_near_zero_slope(V_w[off_lo2:off_hi2], I_w[off_lo2:off_hi2])
        derive_roff = slope0_off is not None and slope0_off > 0

    x0, lo, hi = _param_transform(x0_dict, bounds_dict)
    free_names = [n for n in dmm.PARAM_NAMES if n not in fixed
                  and not (n == "ron" and derive_ron) and not (n == "roff" and derive_roff)]
    free_idx = [dmm.PARAM_NAMES.index(n) for n in free_names]
    x0_free, lo_free, hi_free = x0[free_idx], lo[free_idx], hi[free_idx]
    i_ron, i_roff = dmm.PARAM_NAMES.index("ron"), dmm.PARAM_NAMES.index("roff")
    i_aon, i_ion = dmm.PARAM_NAMES.index("aon"), dmm.PARAM_NAMES.index("ion")
    i_aoff, i_ioff = dmm.PARAM_NAMES.index("aoff"), dmm.PARAM_NAMES.index("ioff")
    ron_bounds = bounds_dict["ron"]
    roff_bounds = bounds_dict["roff"]

    scale = 0.02 * np.max(np.abs(I_w)) + 1e-15

    # Real dI/dt, smoothed FIRST (see fit_derivative's own docstring above
    # for why) -- fixed for the whole fit, computed once outside residuals().
    didt_real_smooth = moving_average(np.gradient(I_w, t_w), deriv_smooth_n)
    scale_didt = 0.02 * np.max(np.abs(didt_real_smooth)) + 1e-15

    j_vmin = j_vmax = None
    if turning_point_weight > 0.0 and fit_derivative:
        dVdt_w = np.gradient(V_w, t_w)
        def _cleanest(extremum_idx, half_window=5):
            lo_, hi_ = max(0, extremum_idx - half_window), min(len(V_w), extremum_idx + half_window + 1)
            return lo_ + int(np.argmin(np.abs(dVdt_w[lo_:hi_])))
        j_vmin = _cleanest(int(np.argmin(V_w)))
        j_vmax = _cleanest(int(np.argmax(V_w)))

    j_peak_set = j_peak_reset = None
    if third_deriv_weight > 0.0 and fit_derivative:
        # d2I/dt2 of the REAL data, "smooth before every derivative"
        # (RULES_DOC.md section 9.3) -- the peak LOCATION is what
        # matters here, not its value, so this can reuse plain
        # moving_average (sample-count smoothing), not the uniform
        # -grid version -- ok since the window's own dt irregularity
        # (section 10) is far from either SET or RESET's peak region.
        d2idt2_real = moving_average(np.gradient(didt_real_smooth, t_w), deriv_smooth_n)
        v_rising_w2 = np.concatenate(([True], np.diff(V_w) >= 0))
        set_mask_w = (V_w >= 0) & v_rising_w2
        reset_mask_w = V_w < 0
        if set_mask_w.sum() >= 5:
            idx_s = np.flatnonzero(set_mask_w)
            j_peak_set = int(idx_s[np.argmax(np.abs(d2idt2_real[idx_s]))])
        if reset_mask_w.sum() >= 5:
            idx_r = np.flatnonzero(reset_mask_w)
            j_peak_reset = int(idx_r[np.argmax(np.abs(d2idt2_real[idx_r]))])
        scale_d3 = 0.02 * np.max(np.abs(np.gradient(d2idt2_real, t_w))) + 1e-15

    def residuals(x_free):
        x_full = x0.copy()
        x_full[free_idx] = x_free
        params = _from_fit_space(x_full)
        if derive_ron:
            KaKi = params[i_aon] * params[i_ion]
            params[i_ron] = np.clip(1.0 / slope0_on - 1.0 / KaKi, *ron_bounds)
        if derive_roff:
            KaKi = params[i_aoff] * params[i_ioff]
            params[i_roff] = np.clip(1.0 / slope0_off - 1.0 / KaKi, *roff_bounds)
        with np.errstate(over="ignore"):
            I_model, _, _ = dmm.simulate(t_w, V_w, params, g0_override=g0_override)
        r_I = (I_model - I_w) / scale
        if not fit_derivative:
            return r_I
        didt_model_smooth = moving_average(np.gradient(I_model, t_w), deriv_smooth_n)
        r_deriv = (didt_model_smooth - didt_real_smooth) / scale_didt
        extra = []
        if j_vmin is not None:
            extra.append(turning_point_weight * (didt_model_smooth[j_vmin] - didt_real_smooth[j_vmin]) / scale_didt)
            extra.append(turning_point_weight * (didt_model_smooth[j_vmax] - didt_real_smooth[j_vmax]) / scale_didt)
        if j_peak_set is not None or j_peak_reset is not None:
            d2idt2_model = moving_average(np.gradient(didt_model_smooth, t_w), deriv_smooth_n)
            d3idt3_model = np.gradient(d2idt2_model, t_w)
            if j_peak_set is not None:
                extra.append(third_deriv_weight * d3idt3_model[j_peak_set] / scale_d3)
            if j_peak_reset is not None:
                extra.append(third_deriv_weight * d3idt3_model[j_peak_reset] / scale_d3)
        if not extra:
            return np.concatenate([r_I, r_deriv])
        return np.concatenate([r_I, r_deriv, np.array(extra)])

    result = least_squares(residuals, x0_free, bounds=(lo_free, hi_free), method="trf",
                            xtol=1e-9, ftol=1e-9, max_nfev=max_nfev)

    x_full = x0.copy()
    x_full[free_idx] = result.x
    params = _from_fit_space(x_full)
    if derive_ron:
        params[i_ron] = np.clip(1.0 / slope0_on - 1.0 / (params[i_aon] * params[i_ion]), *ron_bounds)
    if derive_roff:
        params[i_roff] = np.clip(1.0 / slope0_off - 1.0 / (params[i_aoff] * params[i_ioff]), *roff_bounds)
    if g0_override is not None:
        params[dmm.PARAM_NAMES.index("g0")] = g0_override
    with np.errstate(over="ignore"):
        I_model, g_hist, va_hist = dmm.simulate(t_w, V_w, params, g0_override=g0_override)

    rmse = float(np.sqrt(np.mean((I_model - I_w) ** 2)))
    rmse_r1 = float(np.sqrt(np.mean((I_model[:split] - I_w[:split]) ** 2))) if split > 0 else float("nan")
    rmse_r2 = float(np.sqrt(np.mean((I_model[split:] - I_w[split:]) ** 2))) if split < len(I_w) else float("nan")
    didt_model_smooth = moving_average(np.gradient(I_model, t_w), deriv_smooth_n)
    didt_rmse = float(np.sqrt(np.mean((didt_model_smooth - didt_real_smooth) ** 2)))

    return dict(params=params, params_dict=dmm.unpack(params),
                rmse=rmse, rmse_region1=rmse_r1, rmse_region2=rmse_r2, didt_rmse=didt_rmse,
                I_model=I_model, g_hist=g_hist, va_hist=va_hist,
                t=t_w, V=V_w, I=I_w, split=split, result=result)


class _GACost:
    """Picklable stand-in for fit_region_pair_ga()'s own cost(x_free) closure
    (by request 2026-10-16, "are you parallelizing the GA as much as
    possible?" -- answer at the time: no, differential_evolution defaulted to
    workers=1/serial, and simply passing workers=-1 crashed outright with
    "Can't get local object 'fit_region_pair_ga.<locals>.cost'" -- scipy's
    multiprocessing path pickles the cost callable to ship it to worker
    processes, and a function closure defined INSIDE another function can't
    be pickled, only a module-level one). Holds exactly what the old closure
    captured, as plain attributes instead of captured variables, so an
    instance pickles fine. See fit_region_pair_ga() for what each one means."""
    __slots__ = ("x0", "free_idx", "scale", "scale_didt", "didt_real_smooth",
                 "t_w", "V_w", "I_w", "g0_override", "fit_derivative", "deriv_smooth_n")

    def __init__(self, x0, free_idx, scale, scale_didt, didt_real_smooth,
                 t_w, V_w, I_w, g0_override, fit_derivative, deriv_smooth_n):
        self.x0, self.free_idx = x0, free_idx
        self.scale, self.scale_didt, self.didt_real_smooth = scale, scale_didt, didt_real_smooth
        self.t_w, self.V_w, self.I_w = t_w, V_w, I_w
        self.g0_override, self.fit_derivative, self.deriv_smooth_n = g0_override, fit_derivative, deriv_smooth_n

    def __call__(self, x_free):
        x_full = self.x0.copy()
        x_full[self.free_idx] = x_free
        params = _from_fit_space(x_full)
        with np.errstate(over="ignore"):
            I_model, _, _ = dmm.simulate(self.t_w, self.V_w, params, g0_override=self.g0_override)
        r_I = (I_model - self.I_w) / self.scale
        if not self.fit_derivative:
            return float(np.sum(r_I ** 2))
        didt_model_smooth = moving_average(np.gradient(I_model, self.t_w), self.deriv_smooth_n)
        r_deriv = (didt_model_smooth - self.didt_real_smooth) / self.scale_didt
        return float(np.sum(r_I ** 2) + np.sum(r_deriv ** 2))


def fit_region_pair_ga(t, V, I, i_vreset1, i_vset1, i_vreset2,
                        x0_dict=None, bounds_dict=None, fixed=None, g0_override=None,
                        fit_derivative=True, deriv_smooth_n=10,
                        lock_values=None, lock_margin_v=0.05,
                        popsize=12, maxiter=100, seed=None, polish=True, tol=1e-9,
                        workers=1):
    """Alternative to fit_region_pair()'s gradient-based trf least_squares:
    a genetic algorithm (scipy.optimize.differential_evolution) over the
    SAME free-side parameters and the SAME scalar objective (sum of
    squared normalized residuals -- I and, by default, dI/dt), added by
    request 2026-10-16 ("try to use a genetic algorithm to perform the
    extraction, but mix it with the direct extraction of vr and vs, so
    we don't get too far away") after fit_rolling_gui15oct.py's own
    per-parameter sequential chain was diagnosed as giving bad vr/vs
    peak-current quality (see memory memristor_model_chain_damping.md).

    A GA explores the whole bounded box with no gradient to guide it --
    good for escaping the compensating-parameter local minima this
    project has repeatedly found (ron/etar bound-hugging while g never
    truly switches, etc.) -- but with no gradient either, nothing stops
    vs/vr (this project's own most failure-sensitive, "most important"
    params) from wandering just as freely as every other parameter,
    with no bias toward the one genuinely reliable direct signal
    available (extract_fit.extract_vs_vr()'s own max(d2I/dV2) threshold
    detector). `lock_values` is the "mix": the caller passes
    {"vs": <direct value>} or {"vr": <direct value>} (whichever one is
    this call's free side -- the other is already held fixed via
    `fixed`), and THAT one parameter's own GA search bounds are narrowed
    to +-lock_margin_v (volts, linear space, default 0.05 V) around the
    direct estimate, intersected with its ordinary bounds_dict span --
    every other free parameter keeps the full bounds_dict range. The GA
    still explores vs/vr a little (so a genuinely wrong direct estimate
    isn't set in stone), but can't wander it across the whole physical
    range the way the harder-to-directly-measure parameters are allowed
    to.

    Reuses fit_region_pair()'s own LOG_PARAMS/LINEAR_PARAMS fit-space
    convention (_param_transform/_from_fit_space) -- this matters even
    more for a GA than for a gradient optimizer: a population sampled
    UNIFORMLY in raw linear space over e.g. ioff's 1e-7..1 range would
    waste nearly all its samples several orders of magnitude away from
    any physically plausible value (the same linear-vs-log gotcha
    already documented for gui_15october.py's own bounded scalar
    search, memory memristor_model_gui_fitting_buttons.md).

    popsize/maxiter control DE's own population size (popsize *
    n_free_params individuals) and generation budget -- deliberately
    small defaults (12/100) since a GA needs far more residual
    evaluations than trf least_squares for the same window, and this
    runs once per half-cycle, potentially thousands of times in a
    chain; `polish=True` (DE's own default) runs a final L-BFGS-B local
    polish from DE's best point.

    Returns the SAME dict shape as fit_region_pair() (params,
    params_dict, rmse, rmse_region1, rmse_region2, didt_rmse, I_model,
    g_hist, va_hist, t, V, I, split, result) so both are interchangeable
    everywhere fit_region_pair()'s return value is used -- `result` here
    is scipy's DE OptimizeResult instead of least_squares'; nothing
    downstream (chain_half_cycles, chain_results_to_dataframe) touches
    that field's internals, only the dict's other keys."""
    if i_vreset1 >= i_vreset2 or not (i_vreset1 <= i_vset1 <= i_vreset2):
        raise ValueError(f"indices must satisfy i_vreset1({i_vreset1}) <= "
                          f"i_vset1({i_vset1}) <= i_vreset2({i_vreset2}), with "
                          f"i_vreset1 < i_vreset2")
    fixed = dict(fixed or {})
    if g0_override is not None:
        fixed["g0"] = g0_override
    x0_dict = {**dmm.NOMINAL, **(x0_dict or {})}
    bounds_dict = {**DEFAULT_BOUNDS, **(bounds_dict or {})}

    sl = slice(i_vreset1, i_vreset2 + 1)
    t_w, V_w, I_w = np.asarray(t[sl]), np.asarray(V[sl]), np.asarray(I[sl])
    split = i_vset1 - i_vreset1

    x0, lo, hi = _param_transform(x0_dict, bounds_dict)
    free_names = [n for n in dmm.PARAM_NAMES if n not in fixed]
    free_idx = [dmm.PARAM_NAMES.index(n) for n in free_names]
    x0_free, lo_free, hi_free = x0[free_idx].copy(), lo[free_idx].copy(), hi[free_idx].copy()

    for name, center in (lock_values or {}).items():
        if name not in free_names:
            continue  # this call's free side doesn't include it -- nothing to lock
        j = free_names.index(name)
        # vs/vr's fit space IS linear space (not in dmm.LOG_PARAMS), so the
        # fit-space bounds and the volt-space lock margin are the same units.
        lo_lock = max(lo_free[j], center - lock_margin_v)
        hi_lock = min(hi_free[j], center + lock_margin_v)
        if lo_lock < hi_lock:
            lo_free[j], hi_free[j] = lo_lock, hi_lock

    # Re-clip ALL free params (not just locked ones) strictly INSIDE their own
    # bounds with a small relative margin, same spirit as _param_transform's own
    # 1e-12 clip -- needed here because differential_evolution's own x0-in-bounds
    # check (_unscale_parameters: (x0-lo)/(hi-lo)) can report a value sitting
    # exactly AT an edge as "outside [0,1]" from float roundoff in that division,
    # which a plain x0<=hi/x0>=lo check (true here) doesn't catch. Confirmed live
    # (2026-10-16): a 0.1V-wide locked vs/vr window raised scipy's own "x0 lay
    # outside the specified bounds" ValueError despite x0 satisfying lo<=x0<=hi
    # exactly, on real chained data within the first 20 cycles.
    eps = np.maximum(1e-9, 1e-6 * (hi_free - lo_free))
    x0_free = np.clip(x0_free, lo_free + eps, hi_free - eps)

    scale = 0.02 * np.max(np.abs(I_w)) + 1e-15
    didt_real_smooth = moving_average(np.gradient(I_w, t_w), deriv_smooth_n)
    scale_didt = 0.02 * np.max(np.abs(didt_real_smooth)) + 1e-15

    cost = _GACost(x0, free_idx, scale, scale_didt, didt_real_smooth,
                    t_w, V_w, I_w, g0_override, fit_derivative, deriv_smooth_n)

    # updating='deferred' is REQUIRED by scipy whenever workers != 1 (the
    # default 'immediate' updating evaluates/accepts trial vectors one at a
    # time within a generation, which is inherently serial -- 'deferred'
    # batches a whole generation's trials together so they can actually be
    # map()'d across worker processes).
    result = differential_evolution(cost, bounds=list(zip(lo_free, hi_free)), x0=x0_free,
                                     popsize=popsize, maxiter=maxiter, seed=seed,
                                     polish=polish, tol=tol, init="latinhypercube",
                                     workers=workers, updating="immediate" if workers == 1 else "deferred")

    x_full = x0.copy()
    x_full[free_idx] = result.x
    params = _from_fit_space(x_full)
    if g0_override is not None:
        params[dmm.PARAM_NAMES.index("g0")] = g0_override
    with np.errstate(over="ignore"):
        I_model, g_hist, va_hist = dmm.simulate(t_w, V_w, params, g0_override=g0_override)

    rmse = float(np.sqrt(np.mean((I_model - I_w) ** 2)))
    rmse_r1 = float(np.sqrt(np.mean((I_model[:split] - I_w[:split]) ** 2))) if split > 0 else float("nan")
    rmse_r2 = float(np.sqrt(np.mean((I_model[split:] - I_w[split:]) ** 2))) if split < len(I_w) else float("nan")
    didt_model_smooth = moving_average(np.gradient(I_model, t_w), deriv_smooth_n)
    didt_rmse = float(np.sqrt(np.mean((didt_model_smooth - didt_real_smooth) ** 2)))

    return dict(params=params, params_dict=dmm.unpack(params),
                rmse=rmse, rmse_region1=rmse_r1, rmse_region2=rmse_r2, didt_rmse=didt_rmse,
                I_model=I_model, g_hist=g_hist, va_hist=va_hist,
                t=t_w, V=V_w, I=I_w, split=split, result=result)


def fit_region_pair_ga_seeded_lsq(t, V, I, i_vreset1, i_vset1, i_vreset2,
                                   x0_dict=None, bounds_dict=None, fixed=None, g0_override=None,
                                   fit_derivative=True, deriv_smooth_n=10,
                                   lock_values=None, lock_margin_v=0.05,
                                   popsize=12, maxiter=100, seed=None, tol=1e-9,
                                   top_frac=1.0, polish_max_nfev=200):
    """"Desperate test" hybrid, added 2026-10-17 by explicit request
    ("let's use GA to generate a population, and then run lsq on these
    population, and use the best as the new seed"), after both plain GA
    (fit_region_pair_ga) and the Jacobian-PCA "directed GA"
    (fit_region_pair_directed_ga) failed to clearly beat plain lsq on the
    Monte Carlo ground-truth recovery test (see memory
    memristor_model_chain_damping.md): classic multi-start local
    refinement, using differential_evolution's own POPULATION (not just
    its single best point) as the set of starting points for a genuine
    bounded least_squares polish, keeping whichever polished candidate
    ends up with the lowest cost.

    Rationale: a GA's population, even only partially converged, is a
    cheap way to sample several different basins of this project's
    "sloppy" (multi-local-minimum, compensating-parameter) cost
    landscape at once -- far more diverse than lsq's own single
    carried-forward starting point, which can only ever find whichever
    basin it started closest to. Each population member gets its own
    REAL local refinement (lsq's own trust-region descent, not GA's
    coarser global search) instead of relying on DE's built-in single-
    point L-BFGS-B polish (polish=True elsewhere in this file) -- this
    is why `DifferentialEvolutionSolver` is used directly here instead
    of the public `differential_evolution()` wrapper fit_region_pair_ga()
    calls: the wrapper doesn't expose the final population, only the
    single best point, and multi-starting from 60-75 points needs them
    all.

    top_frac (default 1.0 = the WHOLE final population, matching the
    request literally): fraction of the population (by lowest DE energy
    first) to actually lsq-polish -- COST WARNING: at 1.0 this runs
    popsize*n_free separate bounded least_squares calls per half-cycle
    (each up to polish_max_nfev iterations) ON TOP OF the GA phase's own
    cost -- strictly more expensive than EITHER fit_region_pair_ga() or
    fit_region_pair_directed_ga() alone, intended for small-scale
    exploratory testing first (not casually run over a full chain)
    before deciding whether a much smaller top_frac is "good enough".

    Returns the SAME dict shape as fit_region_pair()/fit_region_pair_ga()
    -- `result` here is the WINNING candidate's own scipy least_squares
    OptimizeResult (or None if every polish attempt raised, in which
    case the GA's own best unpolished point is used as a fallback)."""
    if i_vreset1 >= i_vreset2 or not (i_vreset1 <= i_vset1 <= i_vreset2):
        raise ValueError(f"indices must satisfy i_vreset1({i_vreset1}) <= "
                          f"i_vset1({i_vset1}) <= i_vreset2({i_vreset2}), with "
                          f"i_vreset1 < i_vreset2")
    fixed = dict(fixed or {})
    if g0_override is not None:
        fixed["g0"] = g0_override
    x0_dict = {**dmm.NOMINAL, **(x0_dict or {})}
    bounds_dict = {**DEFAULT_BOUNDS, **(bounds_dict or {})}

    sl = slice(i_vreset1, i_vreset2 + 1)
    t_w, V_w, I_w = np.asarray(t[sl]), np.asarray(V[sl]), np.asarray(I[sl])
    split = i_vset1 - i_vreset1

    x0, lo, hi = _param_transform(x0_dict, bounds_dict)
    free_names = [n for n in dmm.PARAM_NAMES if n not in fixed]
    free_idx = [dmm.PARAM_NAMES.index(n) for n in free_names]
    x0_free, lo_free, hi_free = x0[free_idx].copy(), lo[free_idx].copy(), hi[free_idx].copy()

    for name, center in (lock_values or {}).items():
        if name not in free_names:
            continue
        j = free_names.index(name)
        lo_lock = max(lo_free[j], center - lock_margin_v)
        hi_lock = min(hi_free[j], center + lock_margin_v)
        if lo_lock < hi_lock:
            lo_free[j], hi_free[j] = lo_lock, hi_lock

    eps = np.maximum(1e-9, 1e-6 * (hi_free - lo_free))
    x0_free = np.clip(x0_free, lo_free + eps, hi_free - eps)

    scale = 0.02 * np.max(np.abs(I_w)) + 1e-15
    didt_real_smooth = moving_average(np.gradient(I_w, t_w), deriv_smooth_n)
    scale_didt = 0.02 * np.max(np.abs(didt_real_smooth)) + 1e-15

    cost = _GACost(x0, free_idx, scale, scale_didt, didt_real_smooth,
                    t_w, V_w, I_w, g0_override, fit_derivative, deriv_smooth_n)

    def residual_fn(x_free):
        return _residual_vector(x_free, x0, free_idx, t_w, V_w, I_w, scale, scale_didt,
                                 didt_real_smooth, g0_override, fit_derivative, deriv_smooth_n)

    # polish=False: we do our OWN (multi-start) polish below instead of DE's
    # single-point one.
    solver = DifferentialEvolutionSolver(cost, bounds=list(zip(lo_free, hi_free)), x0=x0_free,
                                          popsize=popsize, maxiter=maxiter, rng=seed,
                                          polish=False, tol=tol, init="latinhypercube")
    solver.solve()
    population = solver._scale_parameters(solver.population)
    energies = solver.population_energies

    order = np.argsort(energies)
    n_top = max(1, int(np.ceil(top_frac * len(order))))

    best_x_free, best_cost, best_result = None, np.inf, None
    for idx in order[:n_top]:
        seed_x = np.clip(population[idx], lo_free, hi_free)
        try:
            lsq_result = least_squares(residual_fn, seed_x, bounds=(lo_free, hi_free),
                                        method="trf", xtol=1e-9, ftol=1e-9, max_nfev=polish_max_nfev)
            c = float(np.sum(lsq_result.fun ** 2))
        except Exception:
            continue
        if c < best_cost:
            best_cost, best_x_free, best_result = c, lsq_result.x, lsq_result

    if best_x_free is None:
        # every polish attempt raised -- fall back to the GA's own best,
        # unpolished, rather than crash the whole half-cycle fit.
        best_x_free = population[order[0]]

    x_full = x0.copy()
    x_full[free_idx] = best_x_free
    params = _from_fit_space(x_full)
    if g0_override is not None:
        params[dmm.PARAM_NAMES.index("g0")] = g0_override
    with np.errstate(over="ignore"):
        I_model, g_hist, va_hist = dmm.simulate(t_w, V_w, params, g0_override=g0_override)

    rmse = float(np.sqrt(np.mean((I_model - I_w) ** 2)))
    rmse_r1 = float(np.sqrt(np.mean((I_model[:split] - I_w[:split]) ** 2))) if split > 0 else float("nan")
    rmse_r2 = float(np.sqrt(np.mean((I_model[split:] - I_w[split:]) ** 2))) if split < len(I_w) else float("nan")
    didt_model_smooth = moving_average(np.gradient(I_model, t_w), deriv_smooth_n)
    didt_rmse = float(np.sqrt(np.mean((didt_model_smooth - didt_real_smooth) ** 2)))

    return dict(params=params, params_dict=dmm.unpack(params),
                rmse=rmse, rmse_region1=rmse_r1, rmse_region2=rmse_r2, didt_rmse=didt_rmse,
                I_model=I_model, g_hist=g_hist, va_hist=va_hist,
                t=t_w, V=V_w, I=I_w, split=split, result=best_result)


def _residual_vector(x_free, x0, free_idx, t_w, V_w, I_w, scale, scale_didt,
                      didt_real_smooth, g0_override, fit_derivative, deriv_smooth_n):
    """Shared residual-VECTOR builder for fit_region_pair_directed_ga()'s
    own Jacobian estimation and population cost evaluation -- the SAME
    residual definition _GACost.__call__ sums the squares of (I and, by
    default, dI/dt, each scale-normalized exactly as fit_region_pair()
    itself does), kept as the raw vector here (not pre-summed) since a
    Jacobian needs it per-residual, not just the scalar cost."""
    x_full = x0.copy()
    x_full[free_idx] = x_free
    params = _from_fit_space(x_full)
    with np.errstate(over="ignore"):
        I_model, _, _ = dmm.simulate(t_w, V_w, params, g0_override=g0_override)
    r_I = (I_model - I_w) / scale
    if not fit_derivative:
        return r_I
    didt_model_smooth = moving_average(np.gradient(I_model, t_w), deriv_smooth_n)
    r_deriv = (didt_model_smooth - didt_real_smooth) / scale_didt
    return np.concatenate([r_I, r_deriv])


def _fd_jacobian(residual_fn, x, lo, hi, rel_step=1e-4):
    """Central-difference Jacobian of residual_fn at x, IN FIT-SPACE
    units (matching the log10/linear convention _param_transform
    already applies everywhere else in this file, so the eigenvectors
    fit_region_pair_directed_ga() derives from this respect that same
    per-parameter metric). One column per free parameter; each probe is
    clipped to [lo, hi] so a parameter already sitting at its own bound
    still gets a one-sided estimate instead of silently coming out
    zero."""
    n = len(x)
    r_probe = residual_fn(x)
    J = np.zeros((len(r_probe), n))
    for j in range(n):
        h = max(rel_step * max(abs(x[j]), 1.0), 1e-8)
        xp, xm = x.copy(), x.copy()
        xp[j] = min(x[j] + h, hi[j])
        xm[j] = max(x[j] - h, lo[j])
        dx = xp[j] - xm[j]
        if dx <= 0:
            continue
        J[:, j] = (residual_fn(xp) - residual_fn(xm)) / dx
    return J


def fit_region_pair_directed_ga(t, V, I, i_vreset1, i_vset1, i_vreset2,
                                 x0_dict=None, bounds_dict=None, fixed=None, g0_override=None,
                                 fit_derivative=True, deriv_smooth_n=10,
                                 lock_values=None, lock_margin_v=0.05,
                                 popsize=12, maxiter=100, seed=None,
                                 sigma0=0.25, sigma_floor_frac=0.05,
                                 jac_refresh_every=1, jac_rel_step=1e-4,
                                 random_frac=0.3, elite_frac=0.2,
                                 patience=15, tol=1e-9, polish=True, max_nfev_polish=200):
    """"Directed GA": a genetic-algorithm-style POPULATION search, added
    2026-10-17 after a Monte Carlo ground-truth recovery test (see memory
    memristor_model_chain_damping.md's "MONTE CARLO GROUND-TRUTH RECOVERY
    TEST" section) found that even on perfectly clean synthetic data with
    the TRUE generating parameters known exactly, neither plain lsq
    (fit_region_pair) nor plain GA (fit_region_pair_ga, isotropic/uniform
    random mutation) reliably recovers them.

    This reimplements the user's OWN earlier published method (confirmed
    directly, 2026-10-17: "It's the first one, and uses the method I
    described to you"): E. Garcia-Moreno, B. Iniguez, R. Picos, "Directed
    genetic algorithms for OTFT model parameter extraction," IEEE ICSICT
    2008, pp. 424-427, DOI 10.1109/ICSICT.2008.4734565 -- full text not
    yet obtained (only the bibliographic record), so this is this
    project's own best-effort reconstruction of "use the Jacobian to
    generate a PCA and run the variations there, with more weight given
    to the principal directions," not a verified match to the paper's
    exact algorithm. If the paper's own details ever get reviewed
    against this, correct this docstring/implementation to match -- see
    memory memristor_model_directed_ga_reference.md.

    PROJECT-SPECIFIC MOTIVATION: this project's parameter space is
    genuinely "sloppy" (Sethna's term): a few directions are
    tightly constrained by the data ("stiff") while several others barely
    affect the fit at all ("sloppy" -- the compensating ron/aon/ion and
    gam/etar-style ridges this project has repeatedly documented). A
    plain isotropic GA wastes nearly all its mutation budget on sloppy
    combinations that don't matter, while rarely landing an informative
    move along the few stiff directions that actually do.

    By request ("The usual GA generates a random variation. The directed
    GA generates the variations of each generation in the line of the
    steepest descent... Don't use a single line: use the jacobian to
    generate a PCA and run the variations there, with more weight given
    to the principal directions"), this replaces ISOTROPIC random
    mutation with mutation DIRECTED along the local Jacobian's own
    principal axes:

    1. Each generation (or every `jac_refresh_every` generations),
       finite-difference the residual vector at the current best point to
       get J (d residual / d free param, IN FIT SPACE).
    2. Eigendecompose M = J^T @ J (the Gauss-Newton Hessian approximation
       of the sum-of-squares cost) -- its eigenvectors are exactly the
       "stiff" (large eigenvalue) / "sloppy" (small eigenvalue) principal
       directions of the local cost landscape, i.e. a PCA of the local
       sensitivity, not of any sampled population.
    3. Generate each new individual as a Gaussian step in THIS eigenbasis
       (deliberately NOT a single steepest-descent line -- a whole
       weighted subspace instead): per-direction step size
       sigma_i = sigma0 * typical_bound_span * sqrt(eigval_i/max_eigval)
       -- bigger steps along the top/"principal" (most-influential)
       directions, the ones a literal steepest-descent step would be
       dominated by, smaller steps along the lesser ones -- floored at
       sigma_floor_frac*sigma0*typical_bound_span so no direction is ever
       starved entirely.

    CAVEAT worth keeping in mind (not acted on here, since it would
    contradict the explicit design request above -- flagged for a future
    experiment instead): this intentionally biases search TOWARD the
    stiff/well-constrained directions, mirroring what plain steepest
    descent (not Newton's method) does. The opposite weighting --
    bigger steps along the SLOPPY (small-eigenvalue) directions, as
    Newton's method and the "sloppy model ensemble" sampling literature
    both do -- is sometimes argued to explore degenerate ridges more
    efficiently; that would just mean inverting the sigma_i formula
    above (sigma_i ~ 1/sqrt(eigval_i) instead of ~sqrt(eigval_i)) if it's
    ever worth comparing head-to-head.

    `random_frac` of each new generation is instead drawn as a plain
    uniform-random individual within bounds (not Jacobian-directed) --
    NOT a compromise of the "directed" design, kept deliberately: a
    purely gradient-aligned population can get trapped crawling along a
    sloppy valley floor exactly like a plain gradient optimizer would.
    `random_frac` keeps some isotropic global-exploration budget so the
    population can still escape a bad local region the directed moves
    alone might not find a way out of. Each new individual's directed
    anchor point is a RANDOMLY CHOSEN member of the current top
    `elite_frac` fraction of the population (not always the single best),
    for additional population diversity.

    popsize/maxiter/seed: same meaning as fit_region_pair_ga's own
    (population = popsize*n_free individuals; maxiter is a generation
    cap, with an early stop if the best cost hasn't improved by more than
    `tol` relative for `patience` consecutive generations).
    lock_values/lock_margin_v: the SAME "mix in the direct vs/vr
    extraction" mechanism fit_region_pair_ga() already uses.
    polish=True (default): finishes with a short bounded least_squares
    local polish from the population's own best point, same spirit as
    differential_evolution's own polish=True.

    Returns the SAME dict shape as fit_region_pair()/fit_region_pair_ga()
    (params, params_dict, rmse, rmse_region1, rmse_region2, didt_rmse,
    I_model, g_hist, va_hist, t, V, I, split, result), interchangeable
    everywhere those are used -- `result` here is a plain SimpleNamespace
    (x, fun, nit, best_cost_history), not a scipy OptimizeResult; nothing
    downstream touches its internals, only the dict's other keys."""
    if i_vreset1 >= i_vreset2 or not (i_vreset1 <= i_vset1 <= i_vreset2):
        raise ValueError(f"indices must satisfy i_vreset1({i_vreset1}) <= "
                          f"i_vset1({i_vset1}) <= i_vreset2({i_vreset2}), with "
                          f"i_vreset1 < i_vreset2")
    fixed = dict(fixed or {})
    if g0_override is not None:
        fixed["g0"] = g0_override
    x0_dict = {**dmm.NOMINAL, **(x0_dict or {})}
    bounds_dict = {**DEFAULT_BOUNDS, **(bounds_dict or {})}

    sl = slice(i_vreset1, i_vreset2 + 1)
    t_w, V_w, I_w = np.asarray(t[sl]), np.asarray(V[sl]), np.asarray(I[sl])
    split = i_vset1 - i_vreset1

    x0, lo, hi = _param_transform(x0_dict, bounds_dict)
    free_names = [n for n in dmm.PARAM_NAMES if n not in fixed]
    free_idx = [dmm.PARAM_NAMES.index(n) for n in free_names]
    x0_free, lo_free, hi_free = x0[free_idx].copy(), lo[free_idx].copy(), hi[free_idx].copy()
    n_free = len(free_idx)

    for name, center in (lock_values or {}).items():
        if name not in free_names:
            continue
        j = free_names.index(name)
        lo_lock = max(lo_free[j], center - lock_margin_v)
        hi_lock = min(hi_free[j], center + lock_margin_v)
        if lo_lock < hi_lock:
            lo_free[j], hi_free[j] = lo_lock, hi_lock

    eps = np.maximum(1e-9, 1e-6 * (hi_free - lo_free))
    x0_free = np.clip(x0_free, lo_free + eps, hi_free - eps)

    scale = 0.02 * np.max(np.abs(I_w)) + 1e-15
    didt_real_smooth = moving_average(np.gradient(I_w, t_w), deriv_smooth_n)
    scale_didt = 0.02 * np.max(np.abs(didt_real_smooth)) + 1e-15

    def residual_fn(x_free):
        return _residual_vector(x_free, x0, free_idx, t_w, V_w, I_w, scale, scale_didt,
                                 didt_real_smooth, g0_override, fit_derivative, deriv_smooth_n)

    def cost_fn(x_free):
        r = residual_fn(x_free)
        return float(np.sum(r ** 2))

    rng = np.random.default_rng(seed)
    pop_size = max(4, popsize * n_free)
    population = rng.uniform(lo_free, hi_free, size=(pop_size, n_free))
    population[0] = x0_free
    costs = np.array([cost_fn(ind) for ind in population])

    best_idx = int(np.argmin(costs))
    best_x, best_cost = population[best_idx].copy(), float(costs[best_idx])
    best_cost_history = [best_cost]

    eigvecs = np.eye(n_free)
    typical_span = float(np.median(hi_free - lo_free)) or 1.0
    sigmas = np.full(n_free, sigma0 * typical_span)

    n_elite = max(1, int(np.ceil(elite_frac * pop_size)))
    n_random = int(round(random_frac * pop_size))
    stall = 0
    gen = 0
    for gen in range(maxiter):
        if gen % max(1, jac_refresh_every) == 0:
            J = _fd_jacobian(residual_fn, best_x, lo_free, hi_free, rel_step=jac_rel_step)
            M = J.T @ J
            M = M + 1e-12 * max(1.0, float(np.max(np.abs(M)))) * np.eye(n_free)
            eigvals, eigvecs = np.linalg.eigh(M)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = np.clip(eigvals[order], 0.0, None), eigvecs[:, order]
            max_eig = eigvals[0] if eigvals[0] > 0 else 1.0
            sigmas = sigma0 * typical_span * np.sqrt(eigvals / max_eig)
            sigmas = np.maximum(sigmas, sigma_floor_frac * sigma0 * typical_span)

        order_by_cost = np.argsort(costs)
        elite_pool = population[order_by_cost[:n_elite]]

        new_population = population.copy()
        new_population[0] = best_x  # elitism: carry the best-ever forward unconditionally
        for i in range(1, pop_size):
            if i <= n_random:
                cand = rng.uniform(lo_free, hi_free)
            else:
                anchor = elite_pool[rng.integers(len(elite_pool))]
                z = rng.normal(size=n_free)
                delta = eigvecs @ (z * sigmas)
                cand = np.clip(anchor + delta, lo_free, hi_free)
            new_population[i] = cand

        population = new_population
        costs = np.array([cost_fn(ind) for ind in population])

        gen_best_idx = int(np.argmin(costs))
        if costs[gen_best_idx] < best_cost:
            improvement = (best_cost - costs[gen_best_idx]) / max(abs(best_cost), 1e-300)
            best_x, best_cost = population[gen_best_idx].copy(), float(costs[gen_best_idx])
            stall = 0 if improvement > tol else stall + 1
        else:
            stall += 1
        best_cost_history.append(best_cost)
        if stall >= patience:
            break

    x_best_free = best_x
    if polish:
        try:
            polish_result = least_squares(residual_fn, x_best_free, bounds=(lo_free, hi_free),
                                            method="trf", xtol=1e-9, ftol=1e-9, max_nfev=max_nfev_polish)
            if float(np.sum(polish_result.fun ** 2)) <= best_cost:
                x_best_free = polish_result.x
        except Exception:
            pass

    x_full = x0.copy()
    x_full[free_idx] = x_best_free
    params = _from_fit_space(x_full)
    if g0_override is not None:
        params[dmm.PARAM_NAMES.index("g0")] = g0_override
    with np.errstate(over="ignore"):
        I_model, g_hist, va_hist = dmm.simulate(t_w, V_w, params, g0_override=g0_override)

    rmse = float(np.sqrt(np.mean((I_model - I_w) ** 2)))
    rmse_r1 = float(np.sqrt(np.mean((I_model[:split] - I_w[:split]) ** 2))) if split > 0 else float("nan")
    rmse_r2 = float(np.sqrt(np.mean((I_model[split:] - I_w[split:]) ** 2))) if split < len(I_w) else float("nan")
    didt_model_smooth = moving_average(np.gradient(I_model, t_w), deriv_smooth_n)
    didt_rmse = float(np.sqrt(np.mean((didt_model_smooth - didt_real_smooth) ** 2)))

    result = SimpleNamespace(x=x_best_free, fun=residual_fn(x_best_free), nit=gen + 1,
                              best_cost_history=best_cost_history)
    return dict(params=params, params_dict=dmm.unpack(params),
                rmse=rmse, rmse_region1=rmse_r1, rmse_region2=rmse_r2, didt_rmse=didt_rmse,
                I_model=I_model, g_hist=g_hist, va_hist=va_hist,
                t=t_w, V=V_w, I=I_w, split=split, result=result)


SET_SIDE = ["vs", "etas", "ron", "aon", "ion"]
RESET_SIDE = ["vr", "etar", "gam", "roff", "aoff", "ioff"]


def fit_region_pair_coordinate(t, V, I, i_vreset1, i_vset1, i_vreset2, active_side,
                                x0_dict=None, bounds_dict=None, fixed=None, g0_override=None,
                                fit_derivative=True, deriv_smooth_n=10, max_nfev=500,
                                max_outer_rounds=20, tol_frac=1e-4,
                                fix_gam=True, gam_fixed_value=None):
    """BLOCK-COORDINATE-DESCENT alternative to `fit_region_pair()`'s one
    joint least_squares call (by request, 2026-10-14): instead of
    fitting every free parameter of the newly-active side AT ONCE,
    cycles through THREE blocks in a fixed order --
      1. the threshold itself (vs if active_side=='set', vr if 'reset')
      2. its matching series resistance (ron / roff)
      3. the remaining free parameters of that side, jointly
    -- each block optimized BY ITSELF (every other parameter held fixed
    at its current value) until that block's own least_squares call
    converges, then the next block starts from there. The whole 3-block
    cycle repeats until a FULL PASS fails to improve rmse by more than
    `tol_frac` (relative), or `max_outer_rounds` is reached.

    Each block is just a call to `fit_region_pair()` itself with every
    PARAM_NAME outside that block's own list added to `fixed` -- no new
    optimizer, no new residual code; this reuses the exact same bounds/
    log-space handling/derivative-fitting machinery the main joint
    fitter already has, just restricts which names are free each call.

    Model-constraint reductions applied by default (by request, "always
    keep the constraints from our model in mind, so we can reduce the
    number of actual parameters to be extracted" -- reusing this
    project's own already-validated findings rather than re-deriving
    them): `gam` is held FIXED at `gam_fixed_value` (default
    dmm.NOMINAL['gam']) when fix_gam=True (the default) -- section 8's
    own finding that gam sits on a genuine flat ridge, not reliably
    identifiable from one window's data no matter the search method.
    The INACTIVE side's own parameters (RESET_SIDE when active_side==
    'set', and vice versa) are always held fixed at their incoming
    x0_dict value -- the same SET_SIDE/RESET_SIDE partition
    fit_all_cycles.py's own chain already uses, since a single
    switching event underdetermines both sides of the model at once.

    Returns the LAST block's own `fit_region_pair()` result dict
    (same shape as that function's own return value) from whichever
    outer round the loop stopped at -- not necessarily the lowest-rmse
    round seen (unlike fuzzy_fit's own best-checkpoint behavior), since
    each block here is itself a proper converged least_squares call,
    not a small undirected step -- a later block/round should not make
    rmse WORSE except by a genuine different local optimum, which this
    function does not currently guard against."""
    if active_side not in ("set", "reset"):
        raise ValueError("active_side must be 'set' or 'reset'")
    x0_dict = {**dmm.NOMINAL, **(x0_dict or {})}
    fixed_base = dict(fixed or {})

    if active_side == "set":
        thresh_name, rs_name = "vs", "ron"
        other_names = ["etas", "aon", "ion"]
        inactive_names = list(RESET_SIDE)
    else:
        thresh_name, rs_name = "vr", "roff"
        other_names = ["etar", "aoff", "ioff"]
        inactive_names = list(SET_SIDE)
        if fix_gam:
            gam_val = dmm.NOMINAL["gam"] if gam_fixed_value is None else gam_fixed_value
            fixed_base.setdefault("gam", gam_val)
        else:
            other_names = other_names + ["gam"]

    for n in inactive_names:
        fixed_base.setdefault(n, x0_dict[n])

    blocks = [[thresh_name], [rs_name], other_names]

    params = dict(x0_dict)
    fit = None
    prev_rmse = None
    for _outer in range(max_outer_rounds):
        for block_names in blocks:
            block_fixed = dict(fixed_base)
            for n in dmm.PARAM_NAMES:
                if n not in block_names and n not in block_fixed:
                    block_fixed[n] = params[n]
            fit = fit_region_pair(t, V, I, i_vreset1, i_vset1, i_vreset2,
                                   x0_dict=params, bounds_dict=bounds_dict, fixed=block_fixed,
                                   max_nfev=max_nfev, g0_override=g0_override,
                                   fit_derivative=fit_derivative, deriv_smooth_n=deriv_smooth_n)
            params = fit["params_dict"]
        this_round_rmse = fit["rmse"]
        if prev_rmse is not None and (prev_rmse - this_round_rmse) < tol_frac * prev_rmse:
            break
        prev_rmse = this_round_rmse
    return fit


def _cycle_set_reset_split(V, i0, i1):
    """The EXACT global index where this cycle's own V crosses from >=0
    to <0 -- i.e. the real electrical SET-active/RESET-active boundary
    dmm_model.step_g() itself uses (not a curvature estimate). The cycle
    spans [i0, i1): SET-governed sub-span is [i0, i_cross), RESET-
    governed is [i_cross, i1)."""
    rel_neg = np.flatnonzero(V[i0:i1] < 0)
    if len(rel_neg) == 0:
        raise ValueError(f"cycle [{i0},{i1}) never goes negative -- can't locate its RESET phase")
    return i0 + int(rel_neg[0])


def fit_cycle_block(t, V, I, block_cycles, bounds_by_cycle, x0_each=None, x0_per_cycle=None,
                     prev_reset_side=None, bounds_dict=None, max_nfev=2000, g0_override=None,
                     fit_derivative=True, deriv_smooth_n=10):
    """By request (2026-10-12), after the user pointed out a real
    incoherence in fit_region_pair()'s own windowing (it picks window
    BOUNDARIES via a purely geometric curvature detector, then freezes
    one side's params while freeing the other based on which polarity
    the detector labeled the boundary -- but a single switching event's
    worth of data underdetermines 5-6 free params, AND the detector's
    boundary doesn't actually line up with where vs/vr "activate";
    confirmed repeatedly this session via the vr/apparent gap, self-
    quenching, and incomplete-SET-saturation findings) -- then corrected
    again (2026-10-12, same day) after an all-12-params-constant-across-
    the-block first attempt turned out to NOT be what was asked: "each
    vs-vr or vr-vs has DIFFERENT parameters, because the parameters
    change during the set or reset process."

    So EVERY cycle in `block_cycles` gets its OWN independent SET_SIDE
    (vs/etas/ron/aon/ion) and RESET_SIDE (vr/etar/gam/roff/aoff/ioff) --
    exactly like fit_all_cycles.py's per-event granularity -- but ALL of
    them, across the WHOLE block, are fit JOINTLY in ONE least_squares
    call, instead of one event at a time with the other side frozen.
    The SET/RESET split within each cycle is the EXACT sign of V
    (_cycle_set_reset_split) -- not a curvature estimate -- since that's
    literally the condition dmm_model.step_g() itself gates on; this is
    also what directly fixes the original "boundaries don't match
    physics" complaint.

    Mechanics: the block's N cycles give N SET segments and N RESET
    segments, chronologically interleaved (SET_0, RESET_0, SET_1,
    RESET_1, ...). dmm.simulate() is called ONCE PER SEGMENT, in order,
    each with its own event's "active" params but the OTHER side's
    CURRENTLY-IN-FORCE values (the most recent event of that side --
    this block's own earlier event if there is one, else `prev_reset_side`
    for the very first SET segment) -- ron/roff/aon/aoff/ion/ioff are
    needed by EVERY segment's own I(VA,g) relation regardless of which
    side is kinetically active, so "the other side's values" always
    means ALL SIX of its names, not just the kinetic ones. g is carried
    via g0_override from each segment's end to the next's start, so one
    continuous trajectory runs through the whole block exactly as
    dmm.simulate() already guarantees within a single call.

    x0_each: {name: value} warm-start applied to EVERY cycle's own 11
    free values (defaults to dmm.NOMINAL) -- a single shared starting
    guess, since each cycle's own optimum is unknown a priori; letting
    the optimizer differentiate them from there is the whole point.
    prev_reset_side: {name: value} (RESET_SIDE names only) "in force"
    for the block's very first SET segment -- the previous block's own
    LAST reset event's parameters when chaining; dmm.NOMINAL's own
    RESET_SIDE if this is a fresh restart.
    g0_override: starting g for the block's very first sample (None ->
    dmm.NOMINAL's own g0).

    x0_per_cycle: optional list of N {name: value} dicts, one per cycle
    -- a DIFFERENT warm start per cycle (e.g. seeded from an existing
    per-event fit), overriding `x0_each` for that cycle when given.

    Returns a dict: per_cycle_params (list of N {name: value} dicts, one
    per cycle, SET_SIDE+RESET_SIDE keys only -- that cycle's own fresh
    values), rmse/didt_rmse (whole block), I_model/g_hist/t/V/I (whole
    block, concatenated across all segments in order), result (scipy's
    OptimizeResult)."""
    n = len(block_cycles)
    if n < 1:
        raise ValueError("block_cycles must have at least one cycle")
    x0_each = {**dmm.NOMINAL, **(x0_each or {})}
    x0_list = [{**x0_each, **(x0_per_cycle[k] if x0_per_cycle else {})} for k in range(n)]
    prev_reset_side = {**{k: dmm.NOMINAL[k] for k in RESET_SIDE}, **(prev_reset_side or {})}
    bounds_dict = {**DEFAULT_BOUNDS, **(bounds_dict or {})}
    g_start = float(g0_override) if g0_override is not None else float(dmm.NOMINAL["g0"])

    # Segment boundaries (fixed, computed ONCE from raw V -- doesn't depend on fit params at all).
    segments = []  # list of (kind, i_a, i_b) in chronological order, kind in {"set","reset"}
    for k, c in enumerate(block_cycles):
        i0, i1 = bounds_by_cycle[c]
        i_cross = _cycle_set_reset_split(V, i0, i1)
        i_reset_end = bounds_by_cycle[block_cycles[k + 1]][0] if k + 1 < n else i1
        segments.append(("set", i0, i_cross))
        segments.append(("reset", i_cross, i_reset_end))

    # Free-parameter layout: per cycle k, 11 values (SET_SIDE then RESET_SIDE), fit-space (log/linear).
    def _to_fit_space(name, value):
        return np.log10(value) if name in dmm.LOG_PARAMS else value

    def _from_fit_space_1(name, value):
        return 10.0 ** value if name in dmm.LOG_PARAMS else value

    names_per_cycle = SET_SIDE + RESET_SIDE  # 11 names
    x0_free = np.concatenate([[_to_fit_space(nm, x0_list[k][nm]) for nm in names_per_cycle]
                               for k in range(n)])
    lo_free = np.concatenate([[_to_fit_space(nm, bounds_dict[nm][0]) for nm in names_per_cycle]
                               for _ in range(n)])
    hi_free = np.concatenate([[_to_fit_space(nm, bounds_dict[nm][1]) for nm in names_per_cycle]
                               for _ in range(n)])
    x0_free = np.clip(x0_free, lo_free + 1e-12, hi_free - 1e-12)

    i_first, i_last = segments[0][1], segments[-1][2]
    t_full, V_full, I_full = np.asarray(t[i_first:i_last]), np.asarray(V[i_first:i_last]), np.asarray(I[i_first:i_last])
    scale = 0.02 * np.max(np.abs(I_full)) + 1e-15
    didt_real_full_smooth = moving_average(np.gradient(I_full, t_full), deriv_smooth_n)
    scale_didt = 0.02 * np.max(np.abs(didt_real_full_smooth)) + 1e-15

    def _unpack_events(x_free):
        """x_free -> list of N {name: value} dicts (real units), one per cycle."""
        events = []
        for k in range(n):
            chunk = x_free[k * 11:(k + 1) * 11]
            events.append({nm: _from_fit_space_1(nm, v) for nm, v in zip(names_per_cycle, chunk)})
        return events

    def _simulate_block(x_free):
        """Runs every segment in order with the right per-event params,
        g carried continuously. Returns (I_model_full, g_hist_full,
        events) all aligned to [i_first, i_last)."""
        events = _unpack_events(x_free)
        I_chunks, g_chunks = [], []
        g_carry = g_start
        reset_side_in_force = dict(prev_reset_side)
        for k in range(n):
            set_side_k = {nm: events[k][nm] for nm in SET_SIDE}
            reset_side_k = {nm: events[k][nm] for nm in RESET_SIDE}

            i_a, i_b = segments[2 * k][1], segments[2 * k][2]
            params_set = dmm.pack({**set_side_k, **reset_side_in_force, "g0": g_carry})
            with np.errstate(over="ignore"):
                Im, gh, _ = dmm.simulate(t[i_a:i_b], V[i_a:i_b], params_set, g0_override=g_carry)
            I_chunks.append(Im); g_chunks.append(gh)
            g_carry = float(gh[-1]) if len(gh) else g_carry

            i_a, i_b = segments[2 * k + 1][1], segments[2 * k + 1][2]
            params_reset = dmm.pack({**set_side_k, **reset_side_k, "g0": g_carry})
            with np.errstate(over="ignore"):
                Im, gh, _ = dmm.simulate(t[i_a:i_b], V[i_a:i_b], params_reset, g0_override=g_carry)
            I_chunks.append(Im); g_chunks.append(gh)
            g_carry = float(gh[-1]) if len(gh) else g_carry
            reset_side_in_force = reset_side_k

        return np.concatenate(I_chunks), np.concatenate(g_chunks), events

    def residuals(x_free):
        I_model, _, _ = _simulate_block(x_free)
        r_I = (I_model - I_full) / scale
        if not fit_derivative:
            return r_I
        didt_model_smooth = moving_average(np.gradient(I_model, t_full), deriv_smooth_n)
        r_deriv = (didt_model_smooth - didt_real_full_smooth) / scale_didt
        return np.concatenate([r_I, r_deriv])

    result = least_squares(residuals, x0_free, bounds=(lo_free, hi_free), method="trf",
                            xtol=1e-9, ftol=1e-9, max_nfev=max_nfev)

    I_model, g_hist, events = _simulate_block(result.x)
    rmse = float(np.sqrt(np.mean((I_model - I_full) ** 2)))
    didt_model_smooth = moving_average(np.gradient(I_model, t_full), deriv_smooth_n)
    didt_rmse = float(np.sqrt(np.mean((didt_model_smooth - didt_real_full_smooth) ** 2)))

    return dict(per_cycle_params=events, rmse=rmse, didt_rmse=didt_rmse,
                I_model=I_model, g_hist=g_hist, t=t_full, V=V_full, I=I_full, result=result)


def fit_cycle_pair(data_path=DATA_PATH, selected_cycle=2000, smooth_n=10, **fit_kwargs):
    """Convenience one-call wrapper: loads the data, locates the window
    for `selected_cycle` (its own Vreset through the next cycle's Vreset),
    and fits it. Returns (fit_result_dict, window_dict)."""
    curr, cyc, time_, volt = load_data(data_path)
    bounds_by_cycle = cycle_bounds(cyc)
    cycles = sorted(bounds_by_cycle)
    window = find_fit_window(bounds_by_cycle, cycles, selected_cycle, time_, volt, curr, smooth_n)
    fit = fit_region_pair(time_, volt, curr, window["i_vreset1"], window["i_vset1"],
                           window["i_vreset2"], **fit_kwargs)
    return fit, window


def plot_fit(fit, window, save_path=None):
    """Diagnostic figure: I(t) (actual vs. model, V(t) on a twin axis,
    region1/region2 background-shaded, split marked) and I-V (actual vs.
    model), for one fit_region_pair() result."""
    t_rel = fit["t"] - fit["t"][0]
    split = fit["split"]

    fig, (ax_t, ax_iv) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_t.axvspan(t_rel[0], t_rel[split], color=REGION1_COLOR, alpha=0.08, lw=0)
    ax_t.axvspan(t_rel[split], t_rel[-1], color=REGION2_COLOR, alpha=0.08, lw=0)
    ax_t.axvline(t_rel[split], color="k", lw=1, ls="--")
    ax_t.plot(t_rel, fit["I"], ".", ms=3, color="0.5", label="actual")
    ax_t.plot(t_rel, fit["I_model"], "-", lw=1.2, color="crimson", label="model")
    ax_v = ax_t.twinx()
    ax_v.plot(t_rel, fit["V"], "-", lw=1.0, color="tab:blue", alpha=0.6)
    ax_v.set_ylabel("V (V)", color="tab:blue")
    ax_v.tick_params(axis="y", labelcolor="tab:blue")
    ax_t.set_xlabel("t (s)"); ax_t.set_ylabel("I (A)")
    ax_t.set_title(f"cycle {window['selected_cycle']}->{window['next_cycle']}: region1 "
                    f"Vreset1->Vset1 (blue bg) | region2 Vset1->Vreset2 (orange bg)", fontsize=9)
    ax_t.legend(fontsize=8, loc="upper left")

    ax_iv.plot(fit["V"], fit["I"], ".", ms=3, color="0.5", label="actual")
    ax_iv.plot(fit["V"], fit["I_model"], "-", lw=1.2, color="crimson", label="model")
    ax_iv.set_xlabel("V (V)"); ax_iv.set_ylabel("I (A)")
    ax_iv.set_title(f"rmse={fit['rmse']:.3e}  (region1={fit['rmse_region1']:.3e}, "
                     f"region2={fit['rmse_region2']:.3e})", fontsize=9)
    ax_iv.legend(fontsize=8)

    fig.suptitle(f"fit_region_pair: cycle {window['selected_cycle']} + {window['next_cycle']}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--cycle", type=int, required=True,
                     help="the SELECTED cycle -- fits its own Vreset through the NEXT cycle's Vreset")
    ap.add_argument("--smooth", type=int, default=10,
                     help="threshold-estimator smoothing window (same convention as gui_view.py, default 10)")
    ap.add_argument("--max-nfev", type=int, default=2000)
    ap.add_argument("--fix", nargs="*", default=[], metavar="PARAM",
                     help="parameter name(s) to hold fixed at NOMINAL, e.g. --fix gam etar")
    ap.add_argument("--no-fit-derivative", dest="fit_derivative", action="store_false",
                     help="fit I(t) alone, the old behavior -- by default (2026-10-12) the smoothed "
                          "dI/dt is ALSO fit, not just I itself (see fit_region_pair's own docstring)")
    ap.add_argument("--deriv-smooth-n", type=int, default=10,
                     help="moving-average window applied to both real and model dI/dt before "
                          "fitting/scoring it (default 10)")
    ap.add_argument("--plot", action="store_true", help="show a diagnostic plot")
    ap.add_argument("--save-plot", default=None, metavar="PATH", help="also save the plot to PATH")
    ap.add_argument("--out", default=None, metavar="CSV", help="append the fitted row to this CSV")
    args = ap.parse_args()

    fixed = {name: dmm.NOMINAL[name] for name in args.fix}
    fit, window = fit_cycle_pair(args.data, args.cycle, args.smooth,
                                  fixed=fixed, max_nfev=args.max_nfev,
                                  fit_derivative=args.fit_derivative, deriv_smooth_n=args.deriv_smooth_n)

    n = window["i_vreset2"] - window["i_vreset1"] + 1
    print(f"Fitted cycle {args.cycle} + {window['next_cycle']} "
          f"(Vreset1 idx={window['i_vreset1']}, Vset1 idx={window['i_vset1']}, "
          f"Vreset2 idx={window['i_vreset2']}, n={n} samples):")
    for name, value in fit["params_dict"].items():
        tag = " (fixed)" if name in fixed else ""
        print(f"  {name:6s} = {value:.6g}{tag}")
    print(f"  rmse (whole window)       = {fit['rmse']:.6g}")
    print(f"  rmse (region1 reset->set) = {fit['rmse_region1']:.6g}")
    print(f"  rmse (region2 set->reset) = {fit['rmse_region2']:.6g}")
    print(f"  didt_rmse (smoothed, n={args.deriv_smooth_n}) = {fit['didt_rmse']:.6g}"
          f"{'' if args.fit_derivative else '  (not fit against -- --no-fit-derivative)'}")
    print(f"  scipy: {fit['result'].message} (nfev={fit['result'].nfev})")

    if args.out:
        row = dict(cycle=args.cycle, next_cycle=window["next_cycle"], **fit["params_dict"],
                   rmse=fit["rmse"], rmse_region1=fit["rmse_region1"], rmse_region2=fit["rmse_region2"])
        header = not os.path.exists(args.out)
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=header, index=False)
        print(f"Appended to {args.out}")

    if args.plot or args.save_plot:
        plot_fit(fit, window, save_path=args.save_plot)
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
