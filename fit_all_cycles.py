"""Walks the WHOLE dataset as a chain of overlapping three-threshold-point
windows and fits dmm_model.py's parameters to all of it, one window at a
time -- no GUI. Writes one row per window to a CSV for later analysis.
By request (2026-10-09):

    "start by fitting the first vset-vreset-vset (discard the previous
    data until the vset), and then go fitting a new cycle each time:
    after the first vset-vreset-vset are done, shift to the
    vreset-vset-vreset, keeping the calculated parameters for the
    vreset-vset; then, shift to the next vset-vreset-vset, keeping the
    calculated parameters for the vset-reset...."

Threshold points in time order are Vset(1), Vreset(1), Vset(2),
Vreset(2), ... -- one (set, reset) pair per cycle, same estimator
fit_region_pair.py already uses (see its own module docstring for why
one Vset always sits between two consecutive Vreset, and vice versa).
Window i spans points[i : i+3]:
    i=0: Vset(1) -> Vreset(1) -> Vset(2)      ("vset-vreset-vset")
    i=1: Vreset(1) -> Vset(2) -> Vreset(2)     ("vreset-vset-vreset")
    i=2: Vset(2) -> Vreset(2) -> Vset(3)       ("vset-vreset-vset")
    ... alternating, sliding by one point each time.

Window i's FIRST half (points[i:i+2]) is the exact same data as window
(i-1)'s own SECOND half -- already fit there -- so whichever parameter
SIDE governs that half is held FIXED at window (i-1)'s own value for it
(not re-optimized); only the SECOND half's side (not seen before) is
free, fit fresh against this window's new data:
    SET_SIDE   = vs, etas, ron, aon, ion          (active while V>=0, rising)
    RESET_SIDE = vr, etar, gam, roff, aoff, ioff  (active while V<0)
This means each window re-estimates ONE side only (alternating SET/RESET
each step) -- giving a per-switching-EVENT parameter track, not just a
per-window one. The very first window (i=0) has no prior fit to hold
anything fixed from, so all 12 parameters (including g0, the device's
otherwise-unknown absolute starting state) are fit freely there. Every
later window's starting g is NOT refit at all -- it's carried over
exactly (dmm.simulate's g0_override) from the previous window's own
simulated g at that shared point, for a single continuous device
trajectory across the whole walk.

Error recovery (by request, 2026-10-09, after a 10-window test showed
RMSE drifting ~6x when one window failed to converge and the bad result
got carried forward as a later window's FIXED/"known" side): each
window is tried at --max-nfev; if that hits the cap WITHOUT converging
(scipy's own "max function evaluations exceeded" status), it's retried
ONCE at --max-nfev * --retry-multiplier. If the retry ALSO
caps out, the window is marked `bad` in the output and its result is
NOT carried forward. Every window always starts its FIRST attempt at
the plain --max-nfev cap ("recover the initial capping") -- the elevated
cap is never carried over to later windows, only used for that one
window's own retry. By request (2026-10-11), --retry-multiplier<=1
skips that second call entirely (it would use the same cap and the
same deterministic starting point as the first attempt, so it can only
reproduce the same non-convergent result) -- the window is marked bad
straight off the first attempt instead, saving the wasted call.

Extended by request (2026-10-10, cycle 6829 -- a double-bump reset shape
the model can't represent well enough to converge there): a `bad`
window's cycle is the actual problem, so instead of just restarting at
the very next (overlapping) threshold point -- which just re-presents
the same bad cycle and was observed to cap out again -- the chain SKIPS
AHEAD to the next recorded Vset strictly after the bad cycle, discarding
every point in between (none of them get a window/row of their own),
and resumes there as a free, all-12-parameter restart fit, exactly like
the very first window. This can cascade across several bad cycles in a
row until one finally converges.

Extended again by request (2026-10-10, cycles 5588 and 7285-7287): a
window can converge perfectly cleanly by scipy's own standard and STILL
be physically wrong -- a single window's new-side parameters (5-6 of
them, e.g. SET_SIDE's vs/etas/ron/aon/ion) are underdetermined by just
one switching event's worth of data, so the optimizer is free to settle
on a compensating-but-wrong combination (verified directly: vs drifting
from a sensible ~0.6-0.7V down to ~0.2V over thousands of restart-free
cycles, while the REAL detected Vset for that whole stretch stays put
at ~0.6-0.7V the entire time) -- and the existing cap-based recovery
above can't see this at all, since nothing ever hits a cap.

A first attempt at catching this (2026-10-10/11) re-ran the threshold
estimator on the model's own curve and compared its apparent vs/vr
crossing to the real one (`apparent_threshold_gap()`, still here,
still computed and recorded on every window for diagnostic purposes --
but NO LONGER used to gate anything). By request, after that scheme was
run for real and made the WHOLE 2200-9900 cycle range worse (mean rmse
~5x higher than before, confirmed in session): a borderline-but-actually-
fine fit (gap just over the tolerance) got force-retried with vs/vr
tightly pinned, the pinned retry landed on a WORSE fit that still
happened to pass the narrow gap check, and -- because every later window
only re-fits ONE side and holds the other FIXED from whatever the
previous window produced -- that one bad "fix" got carried forward
into every subsequent window for thousands of cycles, with nothing
able to correct it since the gap check it passed doesn't look at
overall fit quality at all, only the one threshold's location.

Replaced (2026-10-11) with a direct check on `rmse` itself: a rolling
baseline of the last RMSE_WINDOW ACCEPTED windows' own rmse was kept,
and a newly converged window was rejected if its rmse was above the
MIDPOINT of that baseline's [min, max]. Replaced AGAIN (2026-10-11,
after visually inspecting the full-dataset GUI run with this scheme
in place): the relative/rolling version let rmse drift upward slowly
enough that the baseline itself crept up with it, so genuinely bad
fits downstream of a drifting-but-technically-fine window still
passed. By direct request, after that inspection made it clear there
is a visible, fairly sharp absolute split in the real data -- rmse at
or below ~0.0003 corresponds to a visibly good fit, anything
meaningfully above that to a visibly unusable one -- this is now a
plain FIXED threshold, RMSE_THRESHOLD (default 0.0004), applied to
EVERY converged window including restarts (no rolling baseline, so
the restart-exclusion/cascading-rejection problem from the relative
scheme doesn't apply here). A window that fails this is marked `bad`
exactly like a cap-twice failure, reusing the same skip-ahead recovery
described above.

Found a THIRD, distinct failure mode this still doesn't catch (by
request, 2026-10-11, after the user visually spotted cycle 7037-7040's
fitted `vr`/`etar`/`ron` looking physically nonsensical despite a fine
rmse): a window can match the real I-V curve's overall SHAPE well
enough for rmse to pass while the device's own internal state `g`
never actually switches during it -- confirmed directly, simulating
individual cycles and plotting g(t): a "degenerate" fit's g stays
essentially flat for the whole window (e.g. 0.00015-0.0013) while a
"healthy" nearby fit's g properly swings from ~0.01 (OFF) to ~0.8 (ON);
the degenerate one also visibly overshoots the reset dip by >2x, an
error rmse dilutes away over the rest of the window. Checked across a
1500-window sample of the (rmse-passing) dataset: g_range
(max(g_hist)-min(g_hist) over the window) is SHARPLY bimodal -- 25% of
windows sit below 0.002 (no real switching), the rest cluster at
0.4-1.0 (real switching), almost nothing in between -- and 99.6% of
windows where `ron` was pinned at its own box bound also had
g_range < 0.05. This makes sense: every window spans one real SET or
RESET event by construction, so g MUST move meaningfully; if it
doesn't, the optimizer found a way to trace the curve's shape using
ron/aon/ion geometry alone, without g doing its actual physical job.
Gated the same way as the rmse check: G_RANGE_THRESHOLD (default 0.1,
comfortably inside the 0.002-0.4 gap), applied to EVERY converged
window including restarts.

Added (2026-10-12, by request): the RAW data's own threshold-crossing
voltage at the window's newly-reached point (`volt[i_b]`, no model
involved -- this is exactly what build_threshold_sequence's own
estimator already detected) is now saved on every row as
`raw_threshold_v`, alongside `vs_vr_gap` = abs(the matching fitted
parameter -- vs if kind_b=='set', vr if kind_b=='reset' -- minus
raw_threshold_v). Checked across the full 2026-10-12 run before adding
any new gate on this: the gap is NOT primarily an eta-driven local-
optimum problem. On the 'set' side it's moderate (median 0.10V, only
4% of windows over 0.2V) and does correlate somewhat with etas
(r=0.35); re-fitting a handful of large-gap 'set' windows from a
deliberately low etas start mostly reproduced the same etas/gap, but
occasionally (when etas started pinned very high, e.g. 564) found a
genuinely better basin (lower etas AND lower rmse together). On the
'reset' side the gap is large and nearly universal (median 0.59V,
~100% of windows over 0.1V) and does NOT correlate with etar
(r=-0.07) -- re-fitting from a low etar start reliably reconverges to
the SAME etar and the SAME gap. Chased this further (2026-10-12): the
gap correlates strongly with `gam` (r=0.78-0.83) and with the ion/ioff
contrast (r=0.75-0.84), NOT with etar or the local sweep rate -- but
re-fitting from a low `gam` start doesn't help either (most windows
reconverge to nearly the same gam/gap; the few that moved to a lower
gam got WORSE rmse, e.g. 0.00025->0.0018, for no real gap improvement).
The actual biggest driver found: the device's own simulated g at the
moment reset nominally begins is usually nowhere near 1 (median 0.41,
25th percentile 0.34 across a 600-window sample) -- correlates with
the gap at r=-0.73. This is expected physics, not a bug: step_g's own
SET update is `g=1-(1-g)*exp(-Seff*dt)`, an exponential APPROACH to 1
that never fully saturates in finite dwell time, and because `gam`
makes the RESET rate self-quenching as g drops (see its own analysis
below), starting a reset from a partially-SET state needs
substantially more overvoltage to finish -- so most of the vs_vr_gap
on the reset side is the combination of incomplete SET saturation
plus gam's feedback, not a fitting defect at all. Given THREE
independent starting-point retries (etar, gam, and the negative
control of not retrying) all failed to show a real, safe improvement
on this side, no retry/gate is added for RESET windows -- only the
diagnostic columns above. (`gam`'s own role here is NOT "transition
steepness" the way etar is: etar sits directly on the sinh's exponent
in V, a conventional sigmoid-sharpness knob; gam sits on g^gamma
INSIDE that same exponent, so it governs how much the RESET rate
self-limits as g falls -- a LOW gam keeps g^gamma near 1 even as g
drops, so the rate barely weakens and reset completes fast once
triggered [sharp, small gap]; a HIGH gam collapses g^gamma fast,
throttling the very rate meant to keep driving the reset [self-
quenching, needs much more overvoltage, wide/smeared transition, big
gap] -- the opposite of what "higher = steeper" intuition suggests.)

On the 'set' side, by contrast, the gap is smaller (median 0.10V) and
DOES correlate with etas (r=0.35), and a handful of high-etas windows
really did find a better basin (lower etas AND lower rmse together)
when retried from a low etas start. So ETA_EXTREME_THRESHOLD/
ETA_RETRY_START (below) gate ONLY 'set' windows whose own fitted etas
came out above the threshold: retried once from a low etas start, and
the retry's result is kept only if it does not make rmse meaningfully
worse AND actually shrinks the vs/apparent gap -- otherwise the
original fit is kept untouched.

Fit against dI/dt too, not just I(t) (by request, 2026-10-12, after the
user spotted cycle 8887 fitting I(t) passably while its derivative
looked "totally different"): every fit_region_pair() call here now
also fits the smoothed dI/dt by default (see that module's own
docstring for the mechanics and a worked example on cycle 8887's own
window) -- a genuine trade-off, not a free improvement, so this is a
real change to what "a good fit" even means, not just an extra
diagnostic. --no-fit-derivative recovers the exact old I-only
behavior; --deriv-smooth-n (default 10) controls the smoothing. The
resulting didt_rmse is recorded on every row regardless.

Usage:
    python3 fit_all_cycles.py                              # the whole dataset
    python3 fit_all_cycles.py --out dades/rolling_fit.csv --max-windows 200
    python3 fit_all_cycles.py --start-cycle 2000 --max-windows 50  # a quick spot-check
"""
import argparse
import os
import time as pytime

import numpy as np
import pandas as pd

import dmm_model as dmm
import fit_region_pair as frp
from data_utils import DATA_PATH, DEFAULT_BOUNDS, load_data, cycle_bounds

SET_SIDE = ["vs", "etas", "ron", "aon", "ion"]
RESET_SIDE = ["vr", "etar", "gam", "roff", "aoff", "ioff"]
assert set(SET_SIDE) | set(RESET_SIDE) | {"g0"} == set(dmm.PARAM_NAMES)

OUT_CSV = "dades/rolling_fit.csv"
RMSE_THRESHOLD = 0.0004     # fixed bad-fit cutoff -- see run_chain() and module docstring
G_RANGE_THRESHOLD = 0.1     # min required max(g)-min(g) over a window -- see module docstring
ETA_EXTREME_THRESHOLD = 300.0  # 'set' windows with etas above this get the low-start retry below
ETA_RETRY_START = 10.0         # low etas starting point for that retry -- see module docstring
RMSE_REGRESSION_TOLERANCE = 1.05  # retry's rmse may be at most 5% worse than the original's


def build_threshold_sequence(bounds_by_cycle, cycles, time_, volt, curr, smooth_n=10,
                              start_cycle=None, threshold_fn=None):
    """Chronological [(global_index, kind, cycle), ...] -- Vset(1),
    Vreset(1), Vset(2), Vreset(2), ... -- one (set, reset) pair per
    cycle. `threshold_fn(t_seg, V_seg, I_seg, smooth_n) -> (i_set,
    i_reset)` picks which per-cycle detector locates these points;
    defaults to `fit_region_pair.estimate_cycle_thresholds` (the
    production max(d2I/dV2) estimator, same one fit_region_pair.py uses
    elsewhere) -- pass `fit_region_pair.estimate_cycle_thresholds_extracted`
    for the alternative, newer-extractor-based windowing (RULES_DOC.md
    section 13). A cycle missing either point is just skipped (not
    interpolated or bridged) -- the next window spanning across the gap
    still works fine (it's just a window with more raw samples in that
    one half than usual), no special-casing needed. `start_cycle`, if
    given, drops every point before that cycle (the "discard the
    previous data until the [first] vset" request) -- defaults to the
    very first cycle in the data."""
    if threshold_fn is None:
        threshold_fn = frp.estimate_cycle_thresholds
    points = []
    for n in cycles:
        if start_cycle is not None and n < start_cycle:
            continue
        i0, i1 = bounds_by_cycle[n]
        if i1 - i0 < 8:
            continue
        i_set, i_reset = threshold_fn(time_[i0:i1], volt[i0:i1], curr[i0:i1], smooth_n)
        if i_set is None or i_reset is None:
            continue
        points.append((i0 + i_set, "set", n))
        points.append((i0 + i_reset, "reset", n))
    return points


def _fit_with_retry(time_, volt, curr, i_a, i_b, i_c, x0_dict, fixed, g0_override,
                     base_max_nfev, retry_multiplier, fit_derivative=True, deriv_smooth_n=10,
                     derive_ron=False, turning_point_weight=0.0):
    """One window's fit, with the retry-then-give-up scheme (by request,
    2026-10-09): try at `base_max_nfev`; if that hits the cap WITHOUT
    converging (scipy's own status==0 -- "the maximum number of function
    evaluations is exceeded", the one non-convergence outcome among
    least_squares's termination codes), retry the SAME window once at
    `base_max_nfev * retry_multiplier`; if THAT also caps out, give up on
    this window (caller marks it 'bad' and restarts the chain fresh at
    the next one -- see run_chain). The elevated cap is used for this one
    window's retry ONLY -- every window, including the very next one,
    starts over at `base_max_nfev` ("recover the initial capping").
    By request (2026-10-11), retry_multiplier<=1 skips the second call
    entirely -- it would use the same cap and the same deterministic
    x0/fixed/g0_override as the first attempt, so it can only reproduce
    the identical non-convergent result; the window is marked bad off
    the first attempt instead.
    Returns (fit, attempt, capped) -- attempt is 1 or 2 (which try
    produced the returned fit), capped is True if even the retry hit its
    own cap."""
    fit = frp.fit_region_pair(time_, volt, curr, i_a, i_b, i_c,
                               x0_dict=x0_dict, fixed=fixed, max_nfev=base_max_nfev,
                               g0_override=g0_override, fit_derivative=fit_derivative,
                               deriv_smooth_n=deriv_smooth_n, derive_ron=derive_ron,
                               turning_point_weight=turning_point_weight)
    if fit["result"].status != 0:  # 0 = max_nfev reached without converging; anything else is real convergence
        if retry_multiplier <= 1:
            return fit, 1, True
        return fit, 1, False

    fit2 = frp.fit_region_pair(time_, volt, curr, i_a, i_b, i_c,
                                x0_dict=x0_dict, fixed=fixed,
                                max_nfev=base_max_nfev * retry_multiplier,
                                g0_override=g0_override, fit_derivative=fit_derivative,
                                deriv_smooth_n=deriv_smooth_n, derive_ron=derive_ron,
                                turning_point_weight=turning_point_weight)
    return fit2, 2, fit2["result"].status == 0


def _maybe_retry_extreme_etas(fit, time_, volt, curr, i_a, i_b, i_c, x0_dict, fixed,
                               g0_override, max_nfev, raw_threshold_v,
                               fit_derivative=True, deriv_smooth_n=10, derive_ron=False,
                               turning_point_weight=0.0):
    """'set'-side-only refinement (by request, 2026-10-12 -- see module
    docstring for the investigation behind this): if a converged fit's
    own etas came out above ETA_EXTREME_THRESHOLD, retry the SAME window
    once from a low ETA_RETRY_START etas start (everything else in
    x0_dict/fixed/g0_override unchanged). The retry is kept ONLY if it
    converged AND its rmse isn't meaningfully worse (within
    RMSE_REGRESSION_TOLERANCE) AND it actually shrinks the vs/raw-
    threshold gap -- otherwise the original fit is returned untouched,
    so this can only ever help, never silently regress a window.
    Returns (fit, retried) -- retried is True iff the retry's result was
    actually substituted in."""
    if fit["params_dict"]["etas"] <= ETA_EXTREME_THRESHOLD:
        return fit, False
    x0_retry = dict(x0_dict)
    x0_retry["etas"] = ETA_RETRY_START
    fit_retry = frp.fit_region_pair(time_, volt, curr, i_a, i_b, i_c,
                                     x0_dict=x0_retry, fixed=fixed, max_nfev=max_nfev,
                                     g0_override=g0_override, fit_derivative=fit_derivative,
                                     deriv_smooth_n=deriv_smooth_n, derive_ron=derive_ron,
                                     turning_point_weight=turning_point_weight)
    if fit_retry["result"].status == 0:  # capped without converging -- discard
        return fit, False
    gap_orig = abs(fit["params_dict"]["vs"] - raw_threshold_v)
    gap_retry = abs(fit_retry["params_dict"]["vs"] - raw_threshold_v)
    if fit_retry["rmse"] <= fit["rmse"] * RMSE_REGRESSION_TOLERANCE and gap_retry < gap_orig:
        return fit_retry, True
    return fit, False


def apparent_threshold_gap(fit, kind_c, smooth_n=10):
    """How far a CONVERGED fit's own MODEL curve's apparent threshold
    crossing (on `kind_c`'s polarity -- 'set' or 'reset') falls from the
    REAL data's own threshold for this exact window. The real value
    needs no separate detection: `fit['V'][-1]`, the window's own
    endpoint, IS the real detected crossing for whichever polarity
    `kind_c` is, by construction (every window ends exactly at its own
    newly-introduced threshold point -- see build_threshold_sequence).

    The model's own apparent crossing is found by re-running the SAME
    max(d2I/dV2) estimator on `fit['I_model']` instead of the real data,
    restricted to `fit['split']` onward (the window's SECOND region) --
    a window's FIRST point can share `kind_c`'s own polarity too (e.g. a
    'vset-vreset-vset' window starts AND ends on a 'set'), so searching
    the whole window could pick up that earlier, already-fit occurrence
    instead of the one this window is actually being graded on.

    Returns (gap, real_v, model_v) -- gap and model_v are None if the
    model's own curve has no qualifying kind_c-polarity sample in that
    tail at all (treated as an automatic failure by the caller, same as
    a numeric non-convergence)."""
    split = fit["split"]
    t_tail, V_tail, Im_tail = fit["t"][split:], fit["V"][split:], fit["I_model"][split:]
    i_set, i_reset = frp.estimate_cycle_thresholds(t_tail, V_tail, Im_tail, smooth_n)
    i_model = i_set if kind_c == "set" else i_reset
    real_v = float(fit["V"][-1])
    if i_model is None:
        return None, real_v, None
    model_v = float(V_tail[i_model])
    return abs(model_v - real_v), real_v, model_v


def run_chain(points, time_, volt, curr, max_nfev=2000, max_windows=None, retry_multiplier=10,
              rmse_threshold=RMSE_THRESHOLD, g_range_threshold=G_RANGE_THRESHOLD, smooth_n=10,
              fit_derivative=True, deriv_smooth_n=10, fix_gam=True, gam_fixed_value=None,
              fix_vr_vs=False, seed_vr_vs=False, fix_ron_roff=False, ron_roff_value=None, derive_ron=False,
              fast_global_values=None, turning_point_weight=0.0):
    """Walks `points` (see build_threshold_sequence) as the overlapping
    three-point window chain described in the module docstring. Yields
    one dict per window: window index, the three (global_index, kind,
    cycle) points, which side was newly fit this step ("set"/"reset",
    None for a free/full-12-param window), attempt (1 or 2 -- whether a
    10x-cap retry was needed), bad (True if even the retry hit its cap,
    OR the rolling-rmse check below rejected it -- this window's own
    fit is then unreliable, NOT carried into the next window),
    retry_reason (None, "capped", "rmse_spike", or "g_collapse" -- see below),
    apparent_gap/apparent_real_v/apparent_model_v (apparent_threshold_gap()'s
    own return for this window's fit, kept purely as a diagnostic column
    now -- see the module docstring for why it's no longer used to gate
    anything; all None if the window never converged numerically at
    all), restart (True if this
    window used a free, all-12-fixed-nothing fit -- either the very
    first window, or the one right after a `bad` window, per the
    recovery scheme described in the module docstring), skip_to_cycle
    (None normally; the cycle number this window's `bad` result caused
    the chain to jump to, if any -- see below), and the
    fit_region_pair() result dict itself.

    Cap-twice recovery, extended by request (2026-10-10, cycle 6829 --
    a double-bump reset shape the model can't represent well enough to
    converge there): a `bad` window (capped at both --max-nfev and
    --max-nfev * --retry-multiplier) means the DATA at that window's new
    cycle (points[i+1]'s own cycle) is the problem, not just this one
    fit's starting point -- sliding forward by a single threshold point,
    as every other window does, just re-presents the same bad cycle to
    the next (restart) attempt (observed in practice: it capped out
    again). So instead, a `bad` window's cycle is marked bad and SKIPPED
    entirely: the chain jumps forward to the next recorded Vset strictly
    after it, discarding every point in between (none of them get their
    own window/row at all), and resumes there as a clean, free
    vset-vreset-vset restart fit -- exactly like the very first window.
    This can cascade (if that restart is ALSO bad, its own cycle is
    skipped the same way) until a window finally converges.

    RMSE-threshold recovery (replaces the apparent-threshold check and
    then the rolling-baseline version, by request 2026-10-11 -- see
    module docstring for why): any converged window, restart or not, is
    rejected (treated exactly like a cap-twice failure, same skip-ahead
    recovery) if its own rmse exceeds the fixed `rmse_threshold`.

    g-range recovery (added 2026-10-11, see module docstring): any
    converged window, restart or not, is ALSO rejected (same skip-ahead
    recovery) if its own simulated g never moves more than
    `g_range_threshold` over the whole window -- a real SET or RESET
    event (which every window spans, by construction) must actually
    switch the device's internal state; a window where it doesn't has
    found some other, non-physical way to trace the I-V curve's shape.

    `max_windows`, if given, stops the GENERATOR itself after that many
    windows have been YIELDED (skipped points never get a window, so
    don't count against this).

    fix_gam (default True, by request 2026-10-12, see RULES_DOC.md
    section 7): `gam` is held at `gam_fixed_value` (default
    dmm.NOMINAL["gam"]=0.22) EVERY window, restart or chained, instead
    of being freely fit. Justified numerically, not just by
    convenience: profiling the FULL model's own rmse objective near a
    confirmed-switching synthetic truth found a genuine RIDGE spanning
    nearly a full order of magnitude of gam (0.3x-3x) where a
    compensating etar reproduces the data to within a few percent --
    i.e. gam is NOT reliably identifiable from one window's data no
    matter how good the search is (confirmed NOT a local-minimum
    search failure). Empirically, this dataset's own good-fit windows
    already cluster tightly around gam~=0.216 (median, std 0.027,
    n=3264 in dades/rolling_fit.csv) even with gam free -- close to
    NOMINAL's 0.22 -- so fixing it should cost little on windows that
    were already behaving, while removing one never-really-free degree
    of freedom from every search (one less dimension particularly for
    the "capped" bad-fit category, the single largest one in that same
    file: 469/808 bad windows). Pass fix_gam=False to restore the
    original free-gam behavior for comparison.

    Validated on 500 real windows (cycles 2000+, identical range to
    this file's own dades/rolling_fit.csv baseline): bad fraction
    15.4%->14.6% (77->73/500) -- a modest NET win, but a mixed one.
    `capped` (unconverged-search) failures drop sharply (46->31,
    consistent with removing one never-really-free search dimension),
    but `rmse_spike`/`g_collapse` rejections rise (24->31, 7->11) and
    median rmse among SUCCESSFUL fits is ~5% worse (0.000280 vs
    0.000267) -- forcing gam to one global constant costs a few
    windows whose own best-fit point on the ridge happens to sit far
    enough from that constant that even etar's compensation can't
    fully make up the difference. Kept as the default anyway: the net
    bad-fraction win is real, and the fits it costs weren't measuring
    a real gam signal in the first place.

    fix_vr_vs (default False -- CONFIRMED BROKEN, do not enable; kept
    only so the mechanism stays reproducible): would hold the newly
    -active side's own threshold param (vs on a "set" window, vr on a
    "reset" window, BOTH on a restart) fixed at its own EXTRACTED value
    (fit_region_pair.estimate_threshold_from_d2idt2 for vs,
    estimate_threshold_from_d3_before_d2 for vr -- the apparent/visible
    crossing, same detector shown in analyze_rolling_fit.py's Curve
    inspector tab) instead of fitting it freely.

    Tested on 500 real windows (cycles 2000+): 100% bad, EVERY window a
    restart, g never switches (g_range~1e-17) -- not a tuning problem,
    a structural one (see RULES_DOC.md section 10 for the full
    writeup). Root cause, traced directly on window 0: the apparent
    crossing (e.g. vs=0.74 there) sits BEYOND where the internal
    voltage VA can ever reach for ANY physically sensible (ron,aon,ion)
    -- the window's own FREE fit only reaches VA=0.52 at its own Vmax.
    Forcing vs=0.74 doesn't just cost some rmse; the optimizer, unable
    to satisfy an impossible constraint, runs off into a wildly
    different, internally-inconsistent corner of parameter space
    (ion jumping from a sensible 0.08 to 24, an extreme "good
    -conductor" regime where VA ends up pinned near 0 EVERYWHERE,
    an ARTIFACT of the broken fit, not a real property of the device)
    while still failing to reach the fixed threshold, wrecking the
    whole fit rather than degrading it gracefully. This is the same
    apparent-vs-true gap documented in section 9, now demonstrated to
    be fatal (not just inaccurate) when used as a FIXED constraint
    rather than a diagnostic.

    seed_vr_vs (default False, by request 2026-10-14, RULES_DOC.md
    section 9.4): the never-before-tried OTHER half of fix_vr_vs's own
    idea -- seed x0_dict['vs']/['vr'] from an extracted threshold, but
    leave BOTH in the FREE parameter set (never added to `fixed`), so
    least_squares can correct a bad guess instead of being forced to
    satisfy it exactly. vs uses the same `estimate_threshold_from_d2idt2`
    as fix_vr_vs did; vr uses the new, better `estimate_vr_seed`
    (fit_region_pair.py) instead of `estimate_threshold_from_d3_before_d2`.
    Since this only changes the FIRST guess least_squares sees, the
    weak apparent-vs-true correlations documented throughout section 9
    matter much less here than they did for fix_vr_vs -- the question
    is whether a better-than-NOMINAL starting point measurably helps
    convergence/bad-fraction, not whether it's an accurate vr on its
    own.

    fix_ron_roff (default False, by request 2026-10-12): holds ron AND
    roff equal and fixed at ron_roff_value (default
    dmm.NOMINAL["ron"]=52.0 -- the paper's own Fig.5 modelcard
    reference, which also doesn't distinguish an ON-state from an
    OFF-state series resistance, on the rationale that this dataset's
    device is the same kind the paper characterizes). Real-data test
    in progress."""
    if fix_gam:
        gam_fixed_value = dmm.NOMINAL["gam"] if gam_fixed_value is None else gam_fixed_value
    if fix_ron_roff:
        ron_roff_value = dmm.NOMINAL["ron"] if ron_roff_value is None else ron_roff_value
    prev_fit = None  # None whenever the NEXT window should be a free/restart fit
    i = 0
    n_yielded = 0
    while i + 2 < len(points):
        if max_windows is not None and n_yielded >= max_windows:
            break
        p_a, p_b, p_c = points[i], points[i + 1], points[i + 2]
        i_a, kind_a, cyc_a = p_a
        i_b, kind_b, cyc_b = p_b
        i_c, kind_c, cyc_c = p_c
        if not (kind_a != kind_b != kind_c):
            raise AssertionError(f"threshold points not strictly alternating at window {i}: "
                                  f"{kind_a}, {kind_b}, {kind_c}")

        restart = prev_fit is None
        if restart:
            x0_dict, fixed, g0_override, new_side = dict(dmm.NOMINAL), {}, None, None
        else:
            new_side_list = RESET_SIDE if kind_b == "reset" else SET_SIDE
            hold_side = SET_SIDE if new_side_list is RESET_SIDE else RESET_SIDE
            x0_dict = dict(prev_fit["params_dict"])
            fixed = {name: x0_dict[name] for name in hold_side}
            g0_override = float(prev_fit["g_hist"][prev_fit["split"]])
            new_side = "reset" if new_side_list is RESET_SIDE else "set"
        if fix_gam:
            x0_dict["gam"] = gam_fixed_value
            fixed["gam"] = gam_fixed_value
        if fast_global_values is not None:
            for name, val in fast_global_values.items():
                x0_dict[name] = fixed[name] = val
        if fix_ron_roff:
            x0_dict["ron"] = x0_dict["roff"] = ron_roff_value
            fixed["ron"] = fixed["roff"] = ron_roff_value
        if fix_vr_vs:
            t_w, V_w, I_w = time_[i_a:i_c + 1], volt[i_a:i_c + 1], curr[i_a:i_c + 1]
            if restart or kind_b == "set":
                j_vs = frp.estimate_threshold_from_d2idt2(t_w, V_w, I_w, 0.2, "set")
                if j_vs is not None:
                    x0_dict["vs"] = fixed["vs"] = float(V_w[j_vs])
            if restart or kind_b == "reset":
                j_vr = frp.estimate_threshold_from_d3_before_d2(t_w, V_w, I_w, -0.2, "reset")
                if j_vr is not None:
                    x0_dict["vr"] = fixed["vr"] = float(V_w[j_vr])
        if seed_vr_vs:
            # SEED ONLY (by request, 2026-10-14, RULES_DOC.md section 9.4):
            # unlike fix_vr_vs above, vr/vs stay in the FREE parameter set --
            # only x0_dict's own starting guess is touched, never `fixed`.
            #
            # IR-drop corrected (by request, 2026-10-14, after a first,
            # UNCORRECTED version came out CATASTROPHIC on a 15-window real
            # smoke test -- 100% bad, g_range~0, every window stuck right at
            # its own seed value): the detector returns a raw MEASURED V,
            # but the model's vs/vr gate VA = V - Rs(g)*I, not V itself (same
            # Ron/Roff IR-drop distinction as section 13.2) -- seeding with
            # raw V directly placed vr ~0.6-0.7V away from where the model
            # can actually reach it, in a region with ~zero local gradient,
            # so least_squares (a LOCAL optimizer) got stuck there despite
            # being nominally free -- functionally the same failure mode as
            # fix_vr_vs's hard constraint, just not declared as one. Fixed
            # by subtracting `estimate_rs_half`'s own Rs estimate times the
            # current AT that same detected sample before using it as x0.
            # See RULES_DOC.md section 13.3.
            t_w, V_w, I_w = time_[i_a:i_c + 1], volt[i_a:i_c + 1], curr[i_a:i_c + 1]
            v_rising_w = np.concatenate(([True], np.diff(V_w) >= 0))
            split_local = i_b - i_a
            # Rs estimates (fit_region_pair.estimate_rs_half, 0.5/slope0),
            # each measured once per window: "on" from the frozen ON segment
            # (V>=0 & falling, same convention as derive_ron) -- pairs with
            # the RESET side, since ron/the IR drop at a vr crossing reflects
            # the pre-reset ON state; "off" from the pre-switch segment
            # before this window's own split (only meaningful when kind_b
            #=="set", i.e. the split point IS the vs crossing; falls back to
            # the whole window otherwise, e.g. a restart) -- pairs with the
            # SET side the same way.
            on_mask = (V_w >= 0) & (~v_rising_w)
            rs_on = frp.estimate_rs_half(V_w[on_mask], I_w[on_mask])
            off_hi = split_local if kind_b == "set" else len(V_w)
            rs_off = frp.estimate_rs_half(V_w[:off_hi], I_w[:off_hi])

            if restart or kind_b == "set":
                j_vs = frp.estimate_threshold_from_d2idt2(t_w, V_w, I_w, 0.2, "set")
                if j_vs is not None:
                    v_raw, i_raw = float(V_w[j_vs]), float(I_w[j_vs])
                    x0_dict["vs"] = (v_raw - rs_off * i_raw) if rs_off is not None else v_raw
                if rs_off is not None:
                    x0_dict["roff"] = float(np.clip(rs_off, *DEFAULT_BOUNDS["roff"]))
            if restart or kind_b == "reset":
                j_vr = frp.estimate_vr_seed(t_w, V_w, I_w, -0.2)
                if j_vr is not None:
                    v_raw, i_raw = float(V_w[j_vr]), float(I_w[j_vr])
                    x0_dict["vr"] = (v_raw - rs_on * i_raw) if rs_on is not None else v_raw
                if rs_on is not None:
                    x0_dict["ron"] = float(np.clip(rs_on, *DEFAULT_BOUNDS["ron"]))

        derive_ron_now = derive_ron and (restart or kind_b == "set")  # only when SET_SIDE is
        # actually being freshly fit this window -- NOT on a "reset" window, where ron/SET_SIDE
        # should stay carried over unchanged, not re-derived from this window's own data.
        fit, attempt, bad = _fit_with_retry(time_, volt, curr, i_a, i_b, i_c,
                                             x0_dict, fixed, g0_override,
                                             max_nfev, retry_multiplier,
                                             fit_derivative=fit_derivative,
                                             deriv_smooth_n=deriv_smooth_n,
                                             derive_ron=derive_ron_now,
                                             turning_point_weight=turning_point_weight)
        retry_reason = "capped" if bad else None
        apparent_gap = apparent_real_v = apparent_model_v = g_range = None
        raw_threshold_v = float(volt[i_b])
        eta_retried = False

        if not bad:
            if kind_b == "set":  # 'set'-side-only refinement, see module docstring
                fit, eta_retried = _maybe_retry_extreme_etas(
                    fit, time_, volt, curr, i_a, i_b, i_c, x0_dict, fixed,
                    g0_override, max_nfev, raw_threshold_v,
                    fit_derivative=fit_derivative, deriv_smooth_n=deriv_smooth_n,
                    derive_ron=derive_ron_now, turning_point_weight=turning_point_weight)
            apparent_gap, apparent_real_v, apparent_model_v = apparent_threshold_gap(
                fit, kind_c, smooth_n)  # diagnostic only, see module docstring
            g_range = float(np.ptp(fit["g_hist"]))
            if fit["rmse"] > rmse_threshold:
                bad = True
                retry_reason = "rmse_spike"
            elif g_range < g_range_threshold:
                bad = True
                retry_reason = "g_collapse"

        skip_to_cycle = None
        if bad:
            prev_fit = None  # next window restarts clean regardless
            j = i + 1
            while j < len(points) and not (points[j][1] == "set" and points[j][2] > cyc_b):
                j += 1
            if j < len(points):
                skip_to_cycle = points[j][2]
            i = j
        else:
            prev_fit = fit
            i += 1

        check_param = "vr" if kind_b == "reset" else "vs"
        vs_vr_gap = abs(fit["params_dict"][check_param] - raw_threshold_v)

        yield dict(window=n_yielded, p_a=p_a, p_b=p_b, p_c=p_c, new_side=new_side,
                   eta_retried=eta_retried,
                   attempt=attempt, bad=bad, restart=restart, skip_to_cycle=skip_to_cycle,
                   retry_reason=retry_reason, apparent_gap=apparent_gap,
                   apparent_real_v=apparent_real_v, apparent_model_v=apparent_model_v,
                   g_range=g_range, raw_threshold_v=raw_threshold_v, vs_vr_gap=vs_vr_gap,
                   fit=fit)
        n_yielded += 1


def _row_from_window(w):
    (i_a, kind_a, cyc_a), (i_b, kind_b, cyc_b), (i_c, kind_c, cyc_c) = w["p_a"], w["p_b"], w["p_c"]
    fit = w["fit"]
    row = dict(window=w["window"],
               i_a=i_a, kind_a=kind_a, cycle_a=cyc_a,
               i_b=i_b, kind_b=kind_b, cycle_b=cyc_b,
               i_c=i_c, kind_c=kind_c, cycle_c=cyc_c,
               new_side=w["new_side"] or "both (restart)", restart=w["restart"],
               attempt=w["attempt"], bad=w["bad"], skip_to_cycle=w["skip_to_cycle"],
               retry_reason=w["retry_reason"], apparent_gap=w["apparent_gap"],
               apparent_real_v=w["apparent_real_v"], apparent_model_v=w["apparent_model_v"],
               g_range=w["g_range"], raw_threshold_v=w["raw_threshold_v"], vs_vr_gap=w["vs_vr_gap"],
               eta_retried=w["eta_retried"],
               rmse=fit["rmse"], rmse_region1=fit["rmse_region1"], rmse_region2=fit["rmse_region2"],
               didt_rmse=fit["didt_rmse"],
               nfev=fit["result"].nfev, status=fit["result"].message)
    row.update(fit["params_dict"])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--max-nfev", type=int, default=2000,
                     help="per-window BASE cap (default 2000, same as fit_region_pair.py; each "
                          "window here has far fewer free params than a from-scratch 12-param fit, "
                          "so usually converges in well under this). A window that hits this cap "
                          "without converging is retried once at --retry-multiplier times this "
                          "cap before being marked 'bad' -- see --retry-multiplier.")
    ap.add_argument("--retry-multiplier", type=int, default=10,
                     help="a window that hits --max-nfev without converging is retried ONCE at "
                          "max_nfev * this (default 10x). If the retry ALSO hits its own cap, the "
                          "window is marked bad (not carried forward) and the NEXT window restarts "
                          "the chain fresh (free 12-param fit from NOMINAL, like the very first "
                          "window) rather than building on the unreliable result.")
    ap.add_argument("--rmse-threshold", type=float, default=RMSE_THRESHOLD,
                     help=f"fixed bad-fit cutoff (default {RMSE_THRESHOLD}) -- any converged window "
                          "(restart or not) whose own rmse exceeds this is rejected (marked bad, same "
                          "skip-ahead recovery as a cap-twice failure).")
    ap.add_argument("--g-range-threshold", type=float, default=G_RANGE_THRESHOLD,
                     help=f"minimum required max(g)-min(g) over a window (default {G_RANGE_THRESHOLD}) "
                          "-- any converged window (restart or not) whose own simulated g never "
                          "switches by at least this much is rejected (marked bad, same skip-ahead "
                          "recovery) -- catches fits that match the I-V curve's shape without the "
                          "device's internal state actually switching, invisible to the rmse check.")
    ap.add_argument("--no-fit-derivative", dest="fit_derivative", action="store_false",
                     help="fit I(t) alone, the old behavior -- by default (2026-10-12) the smoothed "
                          "dI/dt is ALSO fit, not just I itself (see fit_region_pair.py's own "
                          "docstring for the cycle-8887 investigation behind this).")
    ap.add_argument("--deriv-smooth-n", type=int, default=10,
                     help="moving-average window applied to both real and model dI/dt before "
                          "fitting/scoring it (default 10)")
    ap.add_argument("--free-gam", dest="fix_gam", action="store_false",
                     help="let gam be freely fit per window, the old behavior -- by default "
                          "(2026-10-12, see run_chain()'s own docstring and RULES_DOC.md section 7) "
                          "gam is held fixed at --gam-value every window: profiling the full model's "
                          "own rmse objective near a known truth found a genuine ridge spanning "
                          "nearly a decade of gam where a compensating etar fits just as well, so a "
                          "single window's data cannot reliably pin gam down regardless of search "
                          "quality -- fixing it removes a never-really-free degree of freedom.")
    ap.add_argument("--gam-value", type=float, default=None,
                     help="the value gam is held at when --free-gam is NOT given (default: "
                          "dmm.NOMINAL['gam']=0.22, close to this dataset's own good-fit median of "
                          "~0.216 even with gam free).")
    ap.add_argument("--fix-vr-vs", action="store_true",
                     help="CONFIRMED CATASTROPHIC on a real 500-window chain test (2026-10-12, "
                          "default OFF, do not enable -- see --seed-vr-vs below for the working "
                          "version of this idea): fix (not just seed) the newly-active side's own "
                          "vs (on a 'set' window) or vr (on a 'reset' window) at its own EXTRACTED "
                          "value, held constant rather than fit freely. 100%% bad (500/500): forcing "
                          "an unreachable threshold sends the optimizer into a nonsensical corner of "
                          "parameter space. See RULES_DOC.md section 10.")
    ap.add_argument("--seed-vr-vs", action="store_true",
                     help="CONFIRMED NET REGRESSION on a 15-window real smoke test (2026-10-14, "
                          "default OFF, do not enable -- see RULES_DOC.md section 14): SEEDS (not "
                          "fixes) vs/vr/ron/roff from an IR-drop-corrected extracted threshold -- "
                          "13/15 bad vs. the no-seeding baseline's ~4/15 bad. Leaving vs/vr FREE (not "
                          "fixed, unlike --fix-vr-vs) avoided that flag's catastrophic 100%% bad "
                          "failure mode, but the Rs correction (fit_region_pair.estimate_rs_half, "
                          "0.5/slope0) still isn't accurate enough -- it assumes g~=1 at the "
                          "detection point, but incomplete SET saturation (median g~=0.41) means the "
                          "true local Rs sits well above a pure-ron estimate.")
    ap.add_argument("--fix-fast-global", action="store_true",
                     help="CONFIRMED BAD on a real 500-window chain test (2026-10-13, default OFF, "
                          "do not enable): estimates ron/roff/aon/aoff/ion/ioff ONCE globally "
                          "(pooling many windows' own frozen ON segments, fit_region_pair."
                          "estimate_global_on_fast_params -- that pooling itself converges cleanly "
                          "and lands on sensible values) and holds all six fixed for the whole "
                          "chain -- 100%% bad (500/500): the per-window drift in these params is "
                          "real device behavior, not noise, so no single shared value fits enough "
                          "windows. See RULES_DOC.md section 12.2.")
    ap.add_argument("--fix-ron-roff", action="store_true",
                     help="CONFIRMED BAD at every value tried on real chain tests (2026-10-12, "
                          "default OFF, do not enable): hold ron AND roff equal and fixed at "
                          "--ron-roff-value. Tried at 52.0 (dmm.NOMINAL['ron'], the paper's own "
                          "Fig.5 modelcard reference) and 5.2 -- 100%% bad (every window a restart) "
                          "at both. See RULES_DOC.md section 12.1.")
    ap.add_argument("--ron-roff-value", type=float, default=None,
                     help="the value ron/roff are held at when --fix-ron-roff is given (default: "
                          "dmm.NOMINAL['ron']=52.0).")
    ap.add_argument("--derive-ron", action="store_true",
                     help="CONFIRMED NET REGRESSION on a real 500-window chain test (2026-10-12, "
                          "default OFF, do not enable): derives ron each iteration from (aon,ion)'s "
                          "own trial values plus a near-V=0 slope measured directly from this "
                          "window's real data, instead of fitting ron as an independent free "
                          "parameter. A SINGLE-window test looked essentially free (rmse unchanged, "
                          "1 fewer param) -- but the real chain-level test (500 windows, same range "
                          "as the gam-fixed baseline) came out WORSE: bad fraction 14.6%%->16.8%% "
                          "(73->84/500), rmse_spike 31->39 and g_collapse 11->14, with capped "
                          "failures unchanged. Kept only so the mechanism/finding stays reproducible "
                          "-- see fit_region_pair.fit_region_pair()'s own docstring and RULES_DOC.md "
                          "section 11.1 for the full writeup.")
    ap.add_argument("--extracted-windows", action="store_true",
                     help="experimental (2026-10-13, default OFF): define window boundaries using "
                          "the newer, synthetic-validated raw-data extractors (fit_region_pair."
                          "estimate_cycle_thresholds_extracted) instead of the production "
                          "max(d2I/dV2) estimator. This flag's own chain-level test was stopped "
                          "before completing -- unvalidated either way, see RULES_DOC.md section 13.")
    ap.add_argument("--turning-point-weight", type=float, default=0.0,
                     help="experimental (RULES_DOC.md section 11.2, validated on synthetic data but "
                          "noisy, not yet chain-tested on real data before 2026-10-13): extra "
                          "weighted residuals matching model-vs-data dI/dt at the window's own "
                          "Vmin/Vmax turning points, where dV/dt=0 decouples the fast relation from "
                          "the slow (gam/etar) dynamics exactly. Default 0.0 (off); ~0.5 was the "
                          "synthetic study's own reasonable, conservative choice if enabled.")
    ap.add_argument("--start-cycle", type=int, default=None,
                     help="skip all data before this cycle (default: the very first cycle present)")
    ap.add_argument("--max-windows", type=int, default=None,
                     help="stop after this many windows (default: the whole dataset)")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--flush-every", type=int, default=1,
                     help="write/append to --out every N windows (default 1 -- each window is "
                          "flushed immediately, so a killed/interrupted run keeps everything done so far)")
    args = ap.parse_args()

    curr, cyc, time_, volt = load_data(args.data)
    bounds_by_cycle = cycle_bounds(cyc)
    cycles = sorted(bounds_by_cycle)
    threshold_fn = frp.estimate_cycle_thresholds_extracted if args.extracted_windows else None
    points = build_threshold_sequence(bounds_by_cycle, cycles, time_, volt, curr,
                                       args.smooth, args.start_cycle, threshold_fn)
    print(f"{len(cycles)} cycles loaded, {len(points)} threshold points found.")

    fast_global_values = None
    if args.fix_fast_global:
        print("Estimating global ron/roff/aon/aoff/ion/ioff (pooling frozen ON segments)...")
        fast_global_values = frp.estimate_global_on_fast_params(points, time_, volt, curr)
        print(f"  -> {fast_global_values}")

    _run_chain_and_write(points, time_, volt, curr, args, args.out, fast_global_values)


def _run_chain_and_write(points, time_, volt, curr, args, out_path, fast_global_values):
    """One full chain run: walks `points` via run_chain, writes rows to
    `out_path` as it goes (flushed every args.flush_every windows),
    prints the same per-window/summary lines as before, and returns the
    list of row dicts."""
    n_windows_total = max(0, len(points) - 2)
    if args.max_windows is not None:
        n_windows_total = min(n_windows_total, args.max_windows)
    print(f"{n_windows_total} windows to fit. Writing to {out_path} (flush every "
          f"{args.flush_every} window(s)).")
    if os.path.exists(out_path):
        print(f"NOTE: {out_path} already exists -- overwriting it (this run starts the "
              f"chain fresh, it does not resume/append to a previous run).")

    buffer = []
    rows = []
    header_written = False
    t_start = pytime.time()
    n_done = 0
    n_bad = 0
    n_restart = 0
    for w in run_chain(points, time_, volt, curr, max_nfev=args.max_nfev, max_windows=args.max_windows,
                        retry_multiplier=args.retry_multiplier, rmse_threshold=args.rmse_threshold,
                        g_range_threshold=args.g_range_threshold, smooth_n=args.smooth,
                        fit_derivative=args.fit_derivative, deriv_smooth_n=args.deriv_smooth_n,
                        fix_gam=args.fix_gam, gam_fixed_value=args.gam_value,
                        fix_vr_vs=args.fix_vr_vs, seed_vr_vs=args.seed_vr_vs, fix_ron_roff=args.fix_ron_roff,
                        ron_roff_value=args.ron_roff_value, derive_ron=args.derive_ron,
                        fast_global_values=fast_global_values,
                        turning_point_weight=args.turning_point_weight):
        row = _row_from_window(w)
        rows.append(row)
        buffer.append(row)
        n_done += 1
        n_bad += row["bad"]
        n_restart += row["restart"]

        if n_done % args.flush_every == 0 or n_done == n_windows_total:
            df = pd.DataFrame(buffer)
            df.to_csv(out_path, mode="a" if header_written else "w",
                      header=not header_written, index=False)
            header_written = True
            buffer = []

        elapsed = pytime.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0.0
        eta_s = (n_windows_total - n_done) / rate if rate > 0 else float("nan")
        flags = ("RESTART " if row["restart"] else "") + ("attempt2+ " if row["attempt"] >= 2 else "") \
            + ("BAD " if row["bad"] else "")
        gap_txt = f" gap={row['apparent_gap']:.3g}V" if row["apparent_gap"] is not None else ""
        g_range_txt = f" g_range={row['g_range']:.3g}" if row["g_range"] is not None else ""
        didt_txt = f" didt_rmse={row['didt_rmse']:.3g}" if row["didt_rmse"] is not None else ""
        print(f"[{n_done}/{n_windows_total}] window {w['window']} "
              f"({row['kind_a']}(cyc{row['cycle_a']})->{row['kind_b']}(cyc{row['cycle_b']})"
              f"->{row['kind_c']}(cyc{row['cycle_c']})) new_side={row['new_side']} {flags}"
              f"rmse={row['rmse']:.4g} nfev={row['nfev']}{gap_txt}{g_range_txt}{didt_txt} "
              f"[{elapsed:.0f}s elapsed, {rate:.3f} win/s, ETA {eta_s/60:.0f} min]", flush=True)
        if row["bad"]:
            skip_msg = (f"skipping ahead to cycle {row['skip_to_cycle']} (fresh vset-vreset-vset)"
                        if row["skip_to_cycle"] is not None else "no more data after this point")
            if row["retry_reason"] == "capped":
                if args.retry_multiplier <= 1:
                    reason_msg = f"capped at {args.max_nfev}, no retry (retry_multiplier<=1)"
                else:
                    reason_msg = (f"capped at {args.max_nfev} and again at "
                                   f"{args.max_nfev * args.retry_multiplier}")
            elif row["retry_reason"] == "rmse_spike":
                reason_msg = f"rmse above threshold ({args.rmse_threshold})"
            else:
                reason_msg = (f"g_range {row['g_range']:.3g} below threshold "
                               f"({args.g_range_threshold}) -- device never switched")
            print(f"  -> window {w['window']} marked BAD ({reason_msg}) -- not carried forward; "
                  f"{skip_msg}.", flush=True)

    if buffer:  # leftover partial batch
        df = pd.DataFrame(buffer)
        df.to_csv(out_path, mode="a" if header_written else "w", header=not header_written, index=False)

    print(f"Done: {n_done} windows fit in {pytime.time() - t_start:.0f}s, written to {out_path} "
          f"({n_bad} marked bad, {n_restart} restarts [including the first window]).")
    return rows


if __name__ == "__main__":
    main()
