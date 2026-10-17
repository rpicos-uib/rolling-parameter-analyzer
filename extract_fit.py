"""Per-parameter EXTRACTED-VALUE fitting (by request, 2026-10-14) -- a
deliberate restart of fuzzy_fit.py's own rule-based idea, NOT built on
top of it: forget its own +-1 VOTE mechanism and RULE_POLARITY
calibration entirely. Every parameter here has its OWN extraction
procedure, returning a value in that parameter's own physical units
(volts, ohms, amps, seconds) straight from a (t,V,I) curve alone --
the error driving each update is (value extracted from the DATA) -
(the SAME extraction applied to the MODEL's own current simulated
curve), a genuine Newton-style correction in the parameter's own
units, not a vote.

rmse is explicitly NOT the driving signal anywhere in this file (by
request -- "don't use the rms error as the killing factor, because
it's not useful here"): it is simulated and recorded purely as a
reported diagnostic of overall goodness of fit, never used to accept/
reject a step or decide when a parameter (or the whole procedure) has
converged.

Procedure (by request): seed from dmm.NOMINAL, overridden where a
direct estimate is available (vs/vr via the production max(d2I/dV2)
threshold detector; ron/roff via the near-V=0 frozen-segment slope,
the same relation fit_region_pair.py's own derive_ron/estimate_rs_half
use). Then refine ONE PARAMETER AT A TIME, in a fixed order -- vr,
ion, then the rest -- repeating that single parameter's own
extract/correct step a few times (or until it stops moving) before
moving to the next one. After a full pass through every free
parameter, the WHOLE pass repeats from these settled values, until a
pass makes no further meaningful change (or max_outer_passes is hit).

gam is excluded from the free set (held fixed at its seed -- the
established ridge/non-identifiability constraint, RULES_DOC.md section
7/8); g0 is pinned via g0_override, same convention as the rest of
this project's chained fits."""
import numpy as np
from scipy.optimize import brentq, least_squares

import dmm_model as dmm
import fit_region_pair as frp
from data_utils import DEFAULT_BOUNDS, moving_average
from fit_all_cycles import G_RANGE_THRESHOLD, SET_SIDE, RESET_SIDE
from fuzzy_fit import _local_peak_width_height

# Both thresholds FIRST (by request, 2026-10-14, "we're not using information
# enough to extract correctly the vs, because we're starting too late" --
# confirmed: vs used to sit 8th, so six other parameters (ion/ron/roff/aon/
# aoff/ioff) got extracted against whatever vs happened to be seeded as,
# every single pass, before vs itself was ever touched). vr, vs, THEN ion
# (by the original request, "first fix the vr, then the ion, then the next
# ones"), then the rest. gam/g0 are never in this list -- always
# fixed/pinned, see module docstring.
FREE_PARAMS_DEFAULT_ORDER = ["vr", "vs", "ion", "ron", "roff", "aon", "aoff", "ioff", "etas", "etar"]

# Which side's own threshold a parameter's own extraction depends on, for the
# voltage-margin gate below (by request, 2026-10-14, "check what happens if
# we add some 0.1V past vr or vs"): a trial vs/vr that the window's own real
# V never gets within 0.1V of is PHYSICALLY UNREACHABLE for this window --
# no amount of Rs/Ka/Ki tuning can make the device cross it, so any
# extraction attempted under that trial is meaningless from the start
# (confirmed in earlier traces: vs/vr got pinned at their OWN bounds -- 1.5,
# -1.4 -- well past this window's own V range of [-1.4, 1.1]).
SET_RELATED = {"vs", "ron", "aon", "ion", "etas"}
RESET_RELATED = {"vr", "roff", "aoff", "ioff", "etar"}
VOLTAGE_MARGIN = 0.1

LINEAR_PARAMS = {"vs", "vr"}  # additive correction (error is already in the param's own units)
# everything else in FREE_PARAMS_DEFAULT_ORDER: multiplicative correction (log-scale, naturally positive)


def _frozen_on_mask(V_w):
    """The window's own 0->Vmax->0 half-cycle span (fixed 2026-10-14,
    SECOND fix after region1/region2 was ALSO found wrong: "you're still
    using the old masks, not the 0-vmax-0 and 0-vmin-0 new ones" --
    find_half_cycle_spans() below is the SAME zero-crossing geometry
    bootstrap_half_cycles() already uses, now reused here too so the
    per-parameter correction loop's own frozen-ON extraction (ron/aon/
    ion) and the region1/region2-anchored version it replaced are no
    longer two different, inconsistent definitions of "ON". The earlier
    region1/region2 fix (V>=0&falling -> i_vset1->i_vreset2) was itself
    an improvement over the ORIGINAL V>=0&falling mask, but still tied
    to the DETECTED vs/vr crossings (late, biased -- RULES_DOC.md's own
    "starting too late" finding); 0->Vmax->0 needs no detection at all."""
    (lo_max, hi_max), _ = find_half_cycle_spans(V_w)
    mask = np.zeros(len(V_w), dtype=bool)
    mask[lo_max:hi_max + 1] = True
    return mask


def _frozen_off_mask(V_w):
    """The window's own 0->Vmin->0 half-cycle span -- mirror of
    _frozen_on_mask, same fix, same reasoning (see its own docstring)."""
    _, (lo_min, hi_min) = find_half_cycle_spans(V_w)
    mask = np.zeros(len(V_w), dtype=bool)
    mask[lo_min:hi_min + 1] = True
    return mask


def extract_vs_vr(t_w, V_w, I_w, smooth_n=10):
    """Apparent vs/vr via the better-validated per-side detectors
    (fit_region_pair.estimate_cycle_thresholds_extracted: first positive
    d2I/dt2 peak for vs, max d3I/dt3 at/before max d2I/dt2 for vr --
    improved 2026-10-17, by request "check the apparent vr": the vr
    side specifically was still on the older, weaker max(d2I/dV2)
    detector shared with vs, even though RULES_DOC.md's own synthetic
    validation found the d3-before-d2 detector nearly DOUBLES vr's
    correlation with the true model value (0.513 vs 0.445-0.447) --
    there was no reason left to keep using the weaker one for vr once a
    validated, better one existed). The SAME heuristic is applied to
    data and model alike, so its own systematic bias (the Ron/Roff
    IR-drop offset, RULES_DOC.md section 13.2; any peak-picking bias)
    cancels in the (data - model) difference instead of corrupting it.
    Returns (V_at_i_set, V_at_i_reset), either None if that side has no
    active samples in this window.

    RESET search is restricted to AFTER i_set (fixed 2026-10-14,
    confirmed on cycle 4155: "the transition is clearly well below the
    inverting point" -- a blind, whole-window RESET search ("V<0 &
    falling" anywhere) can lock onto the LEFTOVER TAIL of the window's
    own PRECEDING reset (right at the window's start, before any SET
    has happened) instead of the window's own genuine i_vreset2
    transition (after Vmax, after i_set) -- the SAME re-detection
    ambiguity fuzzy_fit.py's own extract_features() docstring already
    documented and worked around for the analogous SET-side case. No
    equivalent SET-side ambiguity exists in this window shape (only
    one "V>0 & rising" stretch), so only RESET needs the restriction."""
    i_set, _ = frp.estimate_cycle_thresholds_extracted(t_w, V_w, I_w, smooth_n)
    v_set = float(V_w[i_set]) if i_set is not None else None
    if i_set is not None and i_set + 1 < len(V_w):
        _, i_reset_rel = frp.estimate_cycle_thresholds_extracted(t_w[i_set:], V_w[i_set:], I_w[i_set:], smooth_n)
        i_reset = i_set + i_reset_rel if i_reset_rel is not None else None
    else:
        _, i_reset = frp.estimate_cycle_thresholds_extracted(t_w, V_w, I_w, smooth_n)
    v_reset = float(V_w[i_reset]) if i_reset is not None else None
    return v_set, v_reset


def find_mid_sweep_kinks(t_w, V_w, I_w, min_step=0.03, flatness_ratio=0.15, v_margin_frac=0.85):
    """Locates clean, ISOLATED steps in dV/dt away from the window's own
    turning point -- by request, 2026-10-15 ("in the middle of the V
    sweep we also have some other changes... the dI/dt has some steps
    that are related to the change in dV/dt"): the real instrument's
    sweep is not a perfectly constant-rate ramp, it holds one rate for a
    few samples then jumps to another, and those jumps show up as
    correlated steps in dI/dt.

    Why: dI/dt = A(V,g)*dV/dt + B(V,g)*dg/dt (dmm.analytic_didt()'s own
    decomposition). Across a clean two-sample dV/dt jump with g's gate
    state UNCHANGED (SET/RESET active-or-not, rising-or-falling the
    same on both sides), B*dg/dt is ~constant, so

        A_apparent = (dI/dt[k+1] - dI/dt[k]) / (dV/dt[k+1] - dV/dt[k])

    is a near-exact local read of A -- the SAME quantity the Vmax/Vmin
    turning-point trick targets (RULES_DOC.md section 11.2 /
    [[memristor_model_turning_point_asymmetry]]), just at many more
    V-locations per half-cycle than just the one turning point.

    VALIDATED against synthetic ground truth 2026-10-15 (known 12-param
    truth, dmm.simulate() over real cycles' own (t,V), A_apparent
    compared to the model's own true A = dI_dVA/(1+Rs*dI_dVA)): over
    200 real half-cycles, median error ~7%, p90 ~25%. Two systematic
    failure modes, both filtered out below:
      - MULTI-SAMPLE clusters (the instrument's rate change spread over
        2-3 samples instead of one clean jump): the two-point
        difference straddles a continuous ramp rather than a genuine
        two-plateau step. Filtered by `flatness_ratio`: a candidate at
        (k, k+1) is kept only if dV/dt is itself near-constant on BOTH
        neighboring samples (k-1, k+2).
      - Near the half-cycle's OWN Vmax/Vmin extremum: A(V,g) has strong
        curvature there and dg/dt is changing fastest, so even a clean
        step's two samples no longer see an ~constant B*dg/dt. Filtered
        by `v_margin_frac`: kinks with |V| beyond this fraction of the
        window's OWN peak |V| are excluded -- MUST be computed per
        half-cycle piece, not the whole raw cycle (a whole-cycle |V|
        extremum mixes the +Vmax and -Vmin scales, under-filtering
        whichever side is smaller in magnitude; confirmed directly,
        this was the first version's bug).
    Residual outliers (still ~15% of kinks >20% error, even after both
    filters) cluster near the SET/RESET activation knee (close to the
    window's own vs/vr) -- a third, not-yet-filtered curvature source.
    Not hidden: show these alongside the clean ones, same "data vs
    model, let them diverge visibly" convention as extract_vs/extract_vr.

    min_step: minimum |dV/dt jump| (V/s) to consider at all (rejects
    noise-level wiggle). flatness_ratio: each neighbor's own step must
    be below this fraction of the candidate step to count as "flat".
    v_margin_frac: reject kinks with |V| > this fraction of this
    window's own max(|V|).

    Returns a list of dicts, index order, one per qualifying kink:
    {i, t, v, dvdt_before, dvdt_after, A_apparent}."""
    t_w = np.asarray(t_w); V_w = np.asarray(V_w); I_w = np.asarray(I_w)
    n = len(t_w)
    if n < 5:
        return []
    dvdt = np.gradient(V_w, t_w)
    didt = np.gradient(I_w, t_w)
    v_limit = v_margin_frac * np.max(np.abs(V_w))
    kinks = []
    for k in range(1, n - 2):
        if abs(V_w[k]) > v_limit:
            continue
        step = dvdt[k + 1] - dvdt[k]
        if abs(step) < min_step:
            continue
        before_flat = abs(dvdt[k] - dvdt[k - 1])
        after_flat = abs(dvdt[k + 2] - dvdt[k + 1])
        if before_flat > flatness_ratio * abs(step) or after_flat > flatness_ratio * abs(step):
            continue
        kinks.append(dict(i=k, t=float(t_w[k]), v=float(V_w[k]),
                           dvdt_before=float(dvdt[k]), dvdt_after=float(dvdt[k + 1]),
                           A_apparent=float((didt[k + 1] - didt[k]) / step)))
    return kinks


def extract_rs_total(V_w, I_w, mask):
    """Total near-V=0 resistance 1/slope0 within `mask` -- the EXACT
    relation fit_region_pair.py's own derive_ron/estimate_rs_half use:
    1/slope0 = Rs + 1/(Ka*Ki), NOT yet split into Rs alone (see
    extract_ron/extract_roff below). Reuses estimate_rs_half's own
    (public) measurement rather than reaching into its private slope
    helper: estimate_rs_half returns 0.5/slope0, so 1/slope0 is just
    2x that. None if too few samples or a non-positive slope."""
    if mask.sum() < 7:
        return None
    half = frp.estimate_rs_half(V_w[mask], I_w[mask])
    return None if half is None else 2.0 * half


def extract_ron(V_w, I_w, mask_on, Ka, Ki):
    """Rs = 1/slope0 - 1/(Ka*Ki) on the frozen ON segment, given a
    CURRENT (aon,ion) trial -- the exact split of extract_rs_total's
    own total, same relation derive_ron uses each least_squares
    iteration."""
    total = extract_rs_total(V_w, I_w, mask_on)
    if total is None or Ka is None or Ki is None or Ka <= 0 or Ki <= 0:
        return None
    return total - 1.0 / (Ka * Ki)


def extract_rs_turning_point(t_w, V_w, I_w, Ka, Ki, rs_bounds=(1.0, 500.0)):
    """A second, independent Rs estimate (added 2026-10-17, by request
    "check... the slope near zero, etc, we have many things to check"),
    from the Vmax turning-point asymmetry (RULES_DOC.md /
    memristor_model_turning_point_asymmetry): immediately after the
    sweep's own peak, SET's hard gate switches off exactly at the turn,
    so the post-peak slope alone is dI/dV = A with NO slow-term
    contamination -- validated to ~0.13% median error against synthetic
    ground truth, far tighter than the near-V=0 slope extract_ron
    relies on (which is noisier -- a single finite difference over only
    7 samples -- and, unlike this one, has no independent validation
    number of its own).

    Unlike the near-V=0 case (where VA~=0 regardless of Rs, so
    J=Ka*Ki*cosh(Ka*VA)~=Ka*Ki exactly), VA at the peak is NOT small and
    itself depends on Rs (VA=V-Rs*I), so A = J/(1+Rs*J) with
    J=Ka*Ki*cosh(Ka*VA) is an implicit equation in Rs alone (Ka, Ki
    taken from the current trial) -- solved via brentq over `rs_bounds`.
    Returns None if the window has too few post-peak samples, Ka/Ki are
    invalid, the post-peak slope isn't measurable, or brentq finds no
    root in bounds (noise, or a trial Ka/Ki inconsistent with this
    window's own measured A)."""
    i_peak = int(np.argmax(V_w))
    if i_peak >= len(V_w) - 1 or Ka is None or Ki is None or Ka <= 0 or Ki <= 0:
        return None
    dt = t_w[i_peak + 1] - t_w[i_peak]
    if abs(dt) < 1e-12:
        return None
    dVdt = (V_w[i_peak + 1] - V_w[i_peak]) / dt
    if abs(dVdt) < 1e-9:
        return None
    A = (I_w[i_peak + 1] - I_w[i_peak]) / dt / dVdt
    if A <= 0:
        return None
    V_peak, I_peak = float(V_w[i_peak]), float(I_w[i_peak])

    def h(Rs):
        VA = V_peak - Rs * I_peak
        J = Ka * Ki * np.cosh(Ka * VA)
        return J / (1.0 + Rs * J) - A

    lo, hi = rs_bounds
    try:
        hlo, hhi = h(lo), h(hi)
        if hlo == 0.0:
            return float(lo)
        if hhi == 0.0:
            return float(hi)
        if hlo * hhi > 0:
            return None
        return float(brentq(h, lo, hi, xtol=1e-6))
    except (ValueError, FloatingPointError, OverflowError):
        return None


def extract_roff(V_w, I_w, mask_off, Ka, Ki):
    """Mirror of extract_ron, OFF segment. NOT independently validated
    as a FIXED constraint (derive_roff: CONFIRMED BROKEN as a hard
    constraint, fit_region_pair.py's own docstring) -- used here only
    as an error SIGNAL for a Newton-style correction, a materially
    different use, so its own accuracy here is re-examined on its own
    merits rather than assumed to fail the same way."""
    total = extract_rs_total(V_w, I_w, mask_off)
    if total is None or Ka is None or Ki is None or Ka <= 0 or Ki <= 0:
        return None
    return total - 1.0 / (Ka * Ki)


def extract_Ka_Ki(V_w, I_w, mask, Rs, a_bounds, i_bounds=None):
    """Ka, Ki for I=Ki*sinh(Ka*VA) within `mask` (the frozen ON or OFF
    segment), VA = V - Rs*I (Rs already known from extract_ron/
    extract_roff). Two stages:

    1. Two-probe EXACT inversion (fast, and exact when it works): a
    NEAR probe (small |VA|, same spirit as the 7-nearest-to-V=0 samples
    estimate_rs_half uses) and a FAR probe (the largest |V| still
    inside this frozen segment, for the strongest curvature signal).
    Solves I_far*sinh(Ka*VA_near) - I_near*sinh(Ka*VA_far) = 0 for Ka
    via brentq over `a_bounds`, then Ki = I_near/sinh(Ka*VA_near).

    2. Bounded nonlinear least-squares over EVERY sample in `mask`
    (added 2026-10-17, by request, after the two-probe stage was found
    silently freezing aoff/ioff for an entire 250-cycle Monte Carlo
    chain): stage 1 rests on exactly TWO samples, so a single noisy or
    off-branch probe (the frozen ON/OFF mask spans the WHOLE half-cycle,
    rising and falling branches alike, which can disagree at matched
    |V| under real hysteresis) can leave `g(lo)` and `g(hi)` the same
    sign with no root in `a_bounds` at all -- not a property of the
    device, a property of having thrown away every sample but two.
    Refitting (Ka, Ki) jointly against the FULL masked segment is far
    less sensitive to any one bad sample and has no sign-change
    precondition to fail. Seeded from stage 1 when it succeeded (its
    exact answer is normally an excellent start), else from the
    bounds' own geometric/arithmetic midpoints.

    Returns (Ka, Ki), or (None, None) only if the mask is too small,
    Rs is unknown, or every sample inside it has |VA|~0 (truly no
    signal to fit against, not a precondition this stage could relax)."""
    idx = np.flatnonzero(mask)
    if len(idx) < 7 or Rs is None:
        return None, None
    Vm, Im = V_w[idx], I_w[idx]
    VA_all = Vm - Rs * Im
    if np.max(np.abs(VA_all)) < 1e-9:
        return None, None

    order = np.argsort(np.abs(Vm))
    Vm_s, Im_s = Vm[order], Im[order]
    VA_s = Vm_s - Rs * Im_s
    j_near = min(3, len(VA_s) - 1)
    j_far = len(VA_s) - 1
    VA_near, I_near = VA_s[j_near], Im_s[j_near]
    VA_far, I_far = VA_s[j_far], Im_s[j_far]

    lo, hi = a_bounds
    Ka_probe = Ki_probe = None
    if abs(VA_far - VA_near) >= 1e-6 and abs(VA_near) >= 1e-9:
        def g(Ka):
            return I_far * np.sinh(Ka * VA_near) - I_near * np.sinh(Ka * VA_far)
        try:
            glo, ghi = g(lo), g(hi)
            if glo == 0.0:
                Ka_probe = lo
            elif ghi == 0.0:
                Ka_probe = hi
            elif glo * ghi < 0:
                Ka_probe = brentq(g, lo, hi, xtol=1e-6)
        except (ValueError, FloatingPointError, OverflowError):
            Ka_probe = None
        if Ka_probe is not None:
            sh = np.sinh(Ka_probe * VA_near)
            if abs(sh) >= 1e-15:
                Ki_cand = I_near / sh
                if Ki_cand > 0:
                    Ki_probe = Ki_cand
            if Ki_probe is None:
                Ka_probe = None

    if Ka_probe is not None and Ki_probe is not None:
        return float(Ka_probe), float(Ki_probe)

    # Stage 2: bounded least-squares over the whole masked segment.
    i_lo, i_hi = i_bounds if i_bounds is not None else (1e-8, 1e4)
    with np.errstate(over="ignore"):
        Ki_scale = float(np.clip(np.max(np.abs(Im)) / max(np.sinh(hi * np.max(np.abs(VA_all))), 1e-12),
                                  i_lo, i_hi))

    def resid(x):
        Ka, Ki = x
        with np.errstate(over="ignore"):
            return Ki * np.sinh(Ka * VA_all) - Im

    x0 = [Ka_probe if Ka_probe is not None else 0.5 * (lo + hi),
          Ki_probe if Ki_probe is not None else Ki_scale]
    x0[0] = float(np.clip(x0[0], lo, hi))
    x0[1] = float(np.clip(x0[1], i_lo, i_hi))
    try:
        res = least_squares(resid, x0, bounds=([lo, i_lo], [hi, i_hi]), max_nfev=200)
    except (ValueError, FloatingPointError, OverflowError):
        return None, None
    Ka_fit, Ki_fit = float(res.x[0]), float(res.x[1])
    if Ki_fit <= 0:
        return None, None
    return Ka_fit, Ki_fit


def extract_eta_width(t_w, V_w, I_w, i_anchor, smooth_n=10, half_window=60):
    """FWHM (seconds) of smoothed |dI/dt|'s own peak around i_anchor
    (the transition's own apparent threshold index) -- narrower means
    faster switching, i.e. a LARGER etaS/etaR. Reuses fuzzy_fit.py's
    own _local_peak_width_height (a pure peak-geometry helper, no
    rule/vote machinery). None if i_anchor is None or no finite peak
    is found.

    half_window was 12 (fuzzy_fit.py's own eta_half_window default) --
    widened to 60 (by request, 2026-10-14, "each set or reset cycle
    should be able to include a small part from before... if we don't
    use that, we lose many points in the transition"): confirmed
    directly on cycle 4536's own SET transition that dI/dt doesn't
    actually peak until ~48 samples past the detected threshold (a
    slow, incomplete-saturation transition, not a sharp one) -- a
    +-12 window was clipping off most of the real rise, truncating
    both the measured peak height AND width to a wrong fragment of the
    curve. 60 matches fuzzy_fit.py's own gam_half_window, already
    established as the right scale for this device's own slow tails."""
    if i_anchor is None:
        return None
    didt = moving_average(np.gradient(I_w, t_w), smooth_n)
    _, width = _local_peak_width_height(didt, t_w, i_anchor, half_window)
    return float(width) if np.isfinite(width) else None


def extract_eta_width_masked(t_w, V_w, I_w, mask, smooth_n=10):
    """FWHM (seconds) of smoothed |dI/dt|'s own peak WITHIN `mask` (by
    request, 2026-10-14, "it's very apparent that the GUI is not using
    the 0-vmax-0 and 0-vmin-0 schema... not only in the extraction, but
    also in the drawing" -- confirmed: etas/etar were STILL using
    extract_eta_width's own OLD anchor+-60-sample-window convention,
    centered on the SEPARATELY-detected vs/vr crossing, entirely
    independent of mask_on/mask_off's own 0->Vmax->0 / 0->Vmin->0
    spans). Searches for the peak across the WHOLE given mask (its own
    midpoint as the anchor, its own half-length as the window) instead
    of a fixed, detection-anchored +-60 -- etas/etar now characterize
    the SAME span ron/aon/ion (mask_on) and roff/aoff/ioff (mask_off)
    already do, not a separately-detected, narrower region.

    Returns None if `mask` selects fewer than 3 samples or no finite
    peak is found."""
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        return None
    lo, hi = int(idx.min()), int(idx.max())
    mid = (lo + hi) // 2
    half_window = (hi - lo) // 2 + 1
    didt = moving_average(np.gradient(I_w, t_w), smooth_n)
    _, width = _local_peak_width_height(didt, t_w, mid, half_window)
    return float(width) if np.isfinite(width) else None


def extract_fit(t, V, I, i_vreset1, i_vset1, i_vreset2, x0_dict=None, bounds_dict=None,
                 fixed=None, g0_override=None, param_order=None,
                 max_inner_iters=1, max_outer_passes=40, tol_frac=2e-3, smooth_n=10,
                 max_step_ratio=1.02, max_step_frac=0.02):
    """The driver (by request, 2026-10-14): seeds from NOMINAL plus
    direct extraction, then refines ONE PARAMETER AT A TIME (vr, ion,
    then the rest, FREE_PARAMS_DEFAULT_ORDER) via that parameter's own
    extracted-value error -- additive for vs/vr (already in volts, the
    same units as the error), multiplicative for the rest (log-scale,
    naturally positive).

    max_inner_iters=1 (by request, 2026-10-14, "make them run a single
    iteration per parameter" -- was up to 20, repeating ONE parameter's
    own correction to convergence before ever moving to the next):
    each parameter now gets exactly ONE nudge per visit, then control
    moves straight to the next parameter -- true round-robin, much
    closer to a single coordinated step than a block-converge-then-move
    scheme. Convergence instead comes from REPEATING THE WHOLE PASS (up
    to max_outer_passes times, or until a pass makes no parameter
    move) -- the persistent per-parameter damping state (ratio_cap/
    frac_cap/prev_sign, see below) already carries across passes
    unchanged by this, so a parameter that oscillates still gets
    damped over successive visits, just one visit (not many) per pass.

    rmse is NEVER used to accept/reject a step or to decide when to
    stop (by request) -- simulated and recorded in `history` purely as
    a reported diagnostic of overall goodness of fit.

    max_step_ratio/max_step_frac (damping, by request 2026-10-14, after
    a real divergence: with no damping, a single multiplicative
    correction could swing a parameter 50x in one step -- e.g. ioff
    dropping from 2e-4 towards ~0 in one iteration -- which then blew
    up 1/(Ka*Ki) and corrupted roff's own extraction entirely, while
    ron/vs/etas/etar oscillated between bounds across outer passes
    without ever settling). max_step_ratio caps how far a single
    multiplicative (log-scale) correction may move a parameter per
    inner iteration (clipped to [1/max_step_ratio, max_step_ratio]
    before being applied); max_step_frac caps a single additive (vs/vr)
    correction to that fraction of the parameter's own bounds span per
    inner iteration.

    Both tightened again 2026-10-14 ("use a much smaller update
    increase" -- same request as the GUI's own 1%-per-click nudge
    buttons, applied here to the algorithm's own steps): 1.02/0.02
    (+-2% per step), down from the original damping fix's 1.5/0.25
    (+-50%/25%). max_outer_passes correspondingly raised to 40 (was 8)
    so this many more, much smaller steps can still reach the same
    convergence rather than running out of passes early.

    Returns dict: params_dict, rmse (diagnostic only), I_model, g_hist,
    va_hist, t, V, I, history (one entry per actual attempted
    parameter update: round (a running step counter, for GUI
    plotting), outer_pass, param, extracted_data, extracted_model,
    error, old_value, new_value, moved (bool), params (full snapshot),
    rmse -- rmse here too is informational only)."""
    fixed = dict(fixed or {})
    if g0_override is not None:
        fixed["g0"] = g0_override
    bounds_dict = {**DEFAULT_BOUNDS, **(bounds_dict or {})}
    seed = {**dmm.NOMINAL, **(x0_dict or {})}
    fixed.setdefault("gam", seed["gam"])

    sl = slice(i_vreset1, i_vreset2 + 1)
    t_w, V_w, I_w = np.asarray(t[sl]), np.asarray(V[sl]), np.asarray(I[sl])

    mask_on = _frozen_on_mask(V_w)
    mask_off = _frozen_off_mask(V_w)
    v_set_data, v_reset_data = extract_vs_vr(t_w, V_w, I_w, smooth_n)

    # 1. Seed: NOMINAL, overridden by direct extraction where available
    # ("use the standard values, plus an estimation done using our
    # direct extraction procedures, plus our constraints", by request) --
    # but ONLY for a parameter the CALLER didn't already seed explicitly
    # (fixed 2026-10-14: this used to override vs/vr/ron unconditionally,
    # clobbering a good x0_dict -- e.g. a known-good reference cycle's own
    # fitted values -- with a cruder heuristic BEFORE the loop even
    # started; confirmed live: seeding from the true joint-fit params
    # still came back with the wrong vr/ron because of this).
    params = dict(seed)
    explicit = set(x0_dict or {})
    if v_set_data is not None and "vs" not in explicit:
        params["vs"] = float(np.clip(v_set_data, *bounds_dict["vs"]))
    if v_reset_data is not None and "vr" not in explicit:
        params["vr"] = float(np.clip(v_reset_data, *bounds_dict["vr"]))
    rs_on_seed = frp.estimate_rs_half(V_w[mask_on], I_w[mask_on]) if mask_on.sum() >= 7 else None
    if rs_on_seed is not None and "ron" not in explicit:
        params["ron"] = float(np.clip(rs_on_seed, *bounds_dict["ron"]))
    # roff's own OFF-segment slope seed is deliberately NOT used here (confirmed
    # 2026-10-14, traced live): a bad roff seed makes Rs=roff dominate while
    # g~=g0~=0, suppressing VA so it never reaches vs -- SET never triggers,
    # g stays frozen, and every downstream extraction breaks. This is the
    # SAME unreliable OFF-segment measurement RULES_DOC.md section 11.1
    # already found broken for derive_roff, for the same underlying reason
    # (a narrow, noisy near-zero segment) -- falls back to NOMINAL instead,
    # same as every other parameter with no validated direct seed.
    params["gam"] = fixed["gam"]

    order = list(param_order) if param_order else list(FREE_PARAMS_DEFAULT_ORDER)
    free_names = [n for n in order if n not in fixed]

    def _simulate(p):
        with np.errstate(over="ignore"):
            return dmm.simulate(t_w, V_w, dmm.pack(p), g0_override=g0_override)

    def _extracted(name, p, I_model):
        """(extracted_data, extracted_model) for parameter `name`,
        given the model's current simulated curve `I_model` -- vs/vr
        reuse the data-side value precomputed above (doesn't depend on
        `p`); ron/roff/aon/ion/aoff/ioff genuinely need the CURRENT
        trial context (Ka,Ki or Rs) for their own VA correction, so
        those are recomputed here every call."""
        if name == "vr":
            _, m = extract_vs_vr(t_w, V_w, I_model, smooth_n)
            return v_reset_data, m
        if name == "vs":
            m, _ = extract_vs_vr(t_w, V_w, I_model, smooth_n)
            return v_set_data, m
        if name == "ron":
            Ka, Ki = p["aon"], p["ion"]
            # NOT combined with extract_rs_turning_point() (tried
            # 2026-10-17, reverted): that relation assumes g is at (or
            # near) 1 at Vmax, which Section 4.1 of the paper already
            # flags as the atypical case -- checked directly here too,
            # 248/250 real SET windows have g(Vmax)<0.8, so the
            # turning-point read is almost always of the g-blend
            # K(ron,roff,g_peak), not the ron endpoint itself. Using it
            # unconditionally would trade a noisier but unbiased signal
            # for a cleaner but systematically wrong one.
            return extract_ron(V_w, I_w, mask_on, Ka, Ki), extract_ron(V_w, I_model, mask_on, Ka, Ki)
        if name == "roff":
            Ka, Ki = p["aoff"], p["ioff"]
            return extract_roff(V_w, I_w, mask_off, Ka, Ki), extract_roff(V_w, I_model, mask_off, Ka, Ki)
        if name in ("aon", "ion"):
            Rs = p["ron"]
            Ka_d, Ki_d = extract_Ka_Ki(V_w, I_w, mask_on, Rs, bounds_dict["aon"], bounds_dict["ion"])
            Ka_m, Ki_m = extract_Ka_Ki(V_w, I_model, mask_on, Rs, bounds_dict["aon"], bounds_dict["ion"])
            return (Ka_d, Ka_m) if name == "aon" else (Ki_d, Ki_m)
        if name in ("aoff", "ioff"):
            Rs = p["roff"]
            Ka_d, Ki_d = extract_Ka_Ki(V_w, I_w, mask_off, Rs, bounds_dict["aoff"], bounds_dict["ioff"])
            Ka_m, Ki_m = extract_Ka_Ki(V_w, I_model, mask_off, Rs, bounds_dict["aoff"], bounds_dict["ioff"])
            return (Ka_d, Ka_m) if name == "aoff" else (Ki_d, Ki_m)
        if name == "etas":
            return (extract_eta_width_masked(t_w, V_w, I_w, mask_on, smooth_n),
                    extract_eta_width_masked(t_w, V_w, I_model, mask_on, smooth_n))
        if name == "etar":
            return (extract_eta_width_masked(t_w, V_w, I_w, mask_off, smooth_n),
                    extract_eta_width_masked(t_w, V_w, I_model, mask_off, smooth_n))
        return None, None

    # Per-parameter damping state, PERSISTENT across outer passes (by
    # request, 2026-10-14, after a per-pass-reset version let etas/ron
    # creep unboundedly: resetting the cap back to full strength every
    # pass meant each pass "forgot" how much damping the previous one
    # needed and crept a bit further in the same direction every time --
    # e.g. etas climbing 149->305->433->554->...->737 across 8 passes,
    # never leveling off. Carrying ratio_cap/frac_cap/prev_sign across
    # passes means a parameter that oscillated once stays damped for the
    # rest of the run, instead of re-earning the same instability every
    # pass.) Initialized once, indexed by parameter name.
    ratio_cap = {n: max_step_ratio for n in free_names}
    frac_cap = {n: max_step_frac for n in free_names}
    prev_sign = {n: 0 for n in free_names}

    history = []
    step_counter = 0
    for outer in range(max_outer_passes):
        pass_moved = False
        for name in free_names:
            lo, hi = bounds_dict[name]
            local_ratio_cap = ratio_cap[name]
            local_frac_cap = frac_cap[name]
            local_prev_sign = prev_sign[name]
            for inner in range(max_inner_iters):
                old = params[name]
                # Voltage-margin gate (by request, 2026-10-14, "check what happens if
                # we add some 0.1V past vr or vs") -- cheap, BEFORE simulating: if the
                # CURRENT trial vs/vr leaves no real headroom within this window's own
                # V range, no extraction attempted under it can mean anything.
                if name in SET_RELATED and params["vs"] + VOLTAGE_MARGIN > V_w.max():
                    break
                if name in RESET_RELATED and params["vr"] - VOLTAGE_MARGIN < V_w.min():
                    break
                I_model, g_hist_trial, _ = _simulate(params)
                # Switching-state gate (by request, 2026-10-14, "always check that
                # the curve is changing state, even for vs" -- this is the SAME
                # root cause behind every divergence traced today: vr's detector
                # going flat, ron/etas/etar creeping, all happened because the
                # CURRENT trial params left the device barely switching (g~=g0
                # throughout), so whatever "apparent"/extracted value came out
                # was measuring noise or a geometric artifact, not a real
                # transition. Same g_range convention/threshold fit_all_cycles.py
                # already uses (G_RANGE_THRESHOLD=0.1) -- if THIS trial's own
                # simulated curve doesn't switch, extracted_model can't be
                # trusted for ANY parameter, so skip the update entirely rather
                # than act on it.
                if float(np.ptp(g_hist_trial)) < G_RANGE_THRESHOLD:
                    break
                extracted_data, extracted_model = _extracted(name, params, I_model)
                if extracted_data is None or extracted_model is None:
                    break  # can't measure this round -- leave this parameter, move on
                if name in LINEAR_PARAMS:
                    error = extracted_data - extracted_model
                    step = float(np.clip(error, -local_frac_cap * (hi - lo), local_frac_cap * (hi - lo)))
                    new = old + step
                    cur_sign = 0 if step == 0 else (1 if step > 0 else -1)
                else:
                    if extracted_model == 0.0 or (extracted_data > 0) != (extracted_model > 0):
                        break  # sign flip/zero: a ratio correction isn't meaningful here
                    error = extracted_data - extracted_model
                    ratio = float(np.clip(extracted_data / extracted_model, 1.0 / local_ratio_cap, local_ratio_cap))
                    new = old * ratio
                    cur_sign = 0 if ratio == 1.0 else (1 if ratio > 1.0 else -1)
                if local_prev_sign != 0 and cur_sign != 0 and cur_sign != local_prev_sign:
                    local_ratio_cap = 1.0 + (local_ratio_cap - 1.0) * 0.5
                    local_frac_cap *= 0.5
                local_prev_sign = cur_sign
                new = float(np.clip(new, lo, hi))
                span = (hi - lo) if name in LINEAR_PARAMS else abs(old) + 1e-12
                moved = abs(new - old) > tol_frac * span
                params[name] = new
                I_model_after, g_hist, va_hist = _simulate(params)
                rmse_after = float(np.sqrt(np.mean((I_model_after - I_w) ** 2)))
                history.append(dict(round=step_counter, outer_pass=outer, param=name, inner=inner,
                                     extracted_data=extracted_data, extracted_model=extracted_model,
                                     error=error, old_value=old, new_value=new, moved=moved,
                                     params=dict(params), rmse=rmse_after))
                step_counter += 1
                if moved:
                    pass_moved = True
                else:
                    break  # this parameter has converged -- move to the next one
            ratio_cap[name], frac_cap[name], prev_sign[name] = local_ratio_cap, local_frac_cap, local_prev_sign
        if not pass_moved:
            break

    I_model, g_hist, va_hist = _simulate(params)
    rmse = float(np.sqrt(np.mean((I_model - I_w) ** 2)))
    return dict(params_dict=params, rmse=rmse, I_model=I_model, g_hist=g_hist, va_hist=va_hist,
                t=t_w, V=V_w, I=I_w, history=history)


# ------------------------------------------------------- half-cycle bootstrap
def find_half_cycle_spans(V_w):
    """Returns ((lo_max, hi_max), (lo_min, hi_min)): LOCAL index spans for
    the 0->Vmax->0 and 0->Vmin->0 half-cycle excursions within V_w, found
    from its own zero crossings alone -- no detection/extraction
    dependency at all (by request, 2026-10-14: "going from 0-Vmax-0 to
    extract the LRS params, and 0-Vmin-0 to extract the HRS params").
    Falls back to V_w's own start/end when a natural zero crossing isn't
    available on one side within V_w itself (e.g. V_w's own first/last
    sample doesn't sit exactly at V=0 -- true for extract_fit's own
    vreset1->vreset2 window, which starts/ends mid-excursion, not at a
    zero crossing)."""
    sign = np.sign(V_w)
    crossings = np.flatnonzero(np.diff(sign) != 0)
    i_max = int(np.argmax(V_w))
    i_min = int(np.argmin(V_w))
    before_max = [c for c in crossings if c < i_max]
    after_max = [c for c in crossings if c > i_max]
    lo_max = before_max[-1] + 1 if before_max else 0
    hi_max = after_max[0] if after_max else len(V_w) - 1
    before_min = [c for c in crossings if c < i_min]
    after_min = [c for c in crossings if c > i_min]
    lo_min = before_min[-1] + 1 if before_min else 0
    hi_min = after_min[0] if after_min else len(V_w) - 1
    return (lo_max, hi_max), (lo_min, hi_min)


def bootstrap_half_cycles(t, V, I, i_vreset1, i_vset1, i_vreset2, x0_dict=None,
                           bounds_dict=None, max_nfev=2000):
    """Two-stage least-squares bootstrap (by request, 2026-10-14, "for the
    first window 0-Vmax-0, just make a least squares fit of all the
    parameters... because we need some starting point"): a genuine joint
    fit_region_pair.fit_region_pair() call for each side -- NOT
    extract_fit's own per-parameter correction -- used to get a sane
    starting point for it (or used standalone).

    Stage 1 (0->Vmax->0, found via find_half_cycle_spans -- pure zero-
    crossing geometry): fits SET_SIDE + g0 freely; RESET_SIDE (by
    request, "for the first part, the HRS parameters are frozen") is
    held fixed at x0_dict/NOMINAL, since a window that never goes
    negative has no RESET information to fit them from anyway.

    Stage 2 (0->Vmin->0): fits RESET_SIDE freely; SET_SIDE (by request,
    "for the second part, the LRS are frozen") is held fixed at
    stage 1's own result. g0 is independently free here too (NOT
    chained from stage 1 -- the two spans aren't necessarily in
    chronological order within an arbitrary (i_vreset1,i_vset1,
    i_vreset2) window, e.g. the 0->Vmin->0 span often comes FIRST in
    time for a vreset-vset-vreset window -- this is a bootstrap
    STARTING POINT generator, not a continuous-time chain link).

    Returns dict: params_dict (the merged, final 12 -- literally stage
    2's own params_dict, which already reports the stage-1-fixed
    SET_SIDE values too), fit_max, fit_min (the two raw
    fit_region_pair() results, for inspection)."""
    sl = slice(i_vreset1, i_vreset2 + 1)
    t_w, V_w, I_w = np.asarray(t[sl]), np.asarray(V[sl]), np.asarray(I[sl])
    (lo_max, hi_max), (lo_min, hi_min) = find_half_cycle_spans(V_w)
    i_peak_max = lo_max + int(np.argmax(V_w[lo_max:hi_max + 1]))
    i_peak_min = lo_min + int(np.argmin(V_w[lo_min:hi_min + 1]))

    x0 = {**dmm.NOMINAL, **(x0_dict or {})}
    fixed_max = {n: x0[n] for n in RESET_SIDE}
    fit_max = frp.fit_region_pair(t_w, V_w, I_w, lo_max, i_peak_max, hi_max,
                                   x0_dict=x0, bounds_dict=bounds_dict, fixed=fixed_max,
                                   max_nfev=max_nfev)
    p_max = fit_max["params_dict"]

    x0_2 = {**x0, **p_max}
    # gam held fixed too (by request, 2026-10-14, after tracing etar's own
    # long-tail MC failures: this used to only fix SET_SIDE, leaving gam
    # free -- gam is part of RESET_SIDE but reintroduces the EXACT gam/etar
    # ridge non-identifiability RULES_DOC.md section 7/8 already diagnosed
    # and fixed for the production joint fitter; the bootstrap had simply
    # forgotten to apply that same constraint).
    fixed_min = {n: p_max[n] for n in SET_SIDE}
    fixed_min["gam"] = x0["gam"]
    fit_min = frp.fit_region_pair(t_w, V_w, I_w, lo_min, i_peak_min, hi_min,
                                   x0_dict=x0_2, bounds_dict=bounds_dict, fixed=fixed_min,
                                   max_nfev=max_nfev)
    return dict(params_dict=fit_min["params_dict"], fit_max=fit_max, fit_min=fit_min)


# ------------------------------------------------------- real half-cycle chain
def _geo_reset_value(lo_b, hi_b):
    """Sign-preserving geometric mean of a parameter's own two bounds --
    a scale-appropriate 'central' value regardless of whether it ranges
    linearly (ron: 1-500) or across many orders of magnitude (ioff:
    1e-7-1), used as the universal pin-escape reset target in
    chain_half_cycles() below. Reproduces the previously hand-picked
    ron/roff reset value of 22 almost exactly (sqrt(1*500)=22.36) --
    independent confirmation this is the right general rule, not a
    per-parameter magic constant. This is exactly the LOG-SPACE midpoint
    of [lo_b, hi_b] -- see _near_bound_log()'s own docstring for why
    that matters."""
    sign = 1.0 if hi_b >= 0 else -1.0
    return sign * (abs(lo_b) * abs(hi_b)) ** 0.5


LINEAR_MARGIN_PARAMS = {"ron", "roff", "vs", "vr"}  # by request, 2026-10-14/15: these
# 4 keep a LINEAR-space margin check (same as the original vs/vr fix) --
# everyone else (aon/aoff/ion/ioff/etas/etar/gam/g0) uses LOG space, see
# _near_bound_log()'s own docstring for why that matters for the ones
# with very lopsided, many-orders-of-magnitude bounds.


def _near_bound_linear(v, lo_b, hi_b, margin_frac):
    """True if v is within margin_frac of either edge of [lo_b, hi_b] as a
    LINEAR fraction of the span -- the original vs/vr check. Kept (by
    request) for ron/roff/vs/vr specifically, whose bounds are a more
    moderate ratio (500x/500x/30x/30x) than ion/ioff/etas/etar/aon/aoff/
    gam/g0 (up to 1e7x), where linear space stays meaningful."""
    span = hi_b - lo_b
    return v < lo_b + margin_frac * span or v > hi_b - margin_frac * span


def _near_bound_log(v, lo_b, hi_b, margin_frac):
    """True if v is within margin_frac of either edge of [lo_b, hi_b] in
    LOG space (of magnitude, sign-aware), not linear space (by request,
    2026-10-14/15, after generalizing the pin-escape reset to all 12
    params immediately self-trapped roff/ion/ioff: each one sat frozen
    at EXACTLY its own _geo_reset_value() for 2000 straight half-cycles,
    cycle 2000-2999, right from the start of a fresh full-dataset run.
    Cause: a LINEAR fraction-of-span margin is meaningless once the
    bounds span many orders of magnitude -- e.g. roff's margin zone
    under the old check was v<25.95 (5% of its 1-500 span above the
    lower edge), but _geo_reset_value(1,500)=22.36 ITSELF falls inside
    that zone, so resetting roff just re-qualifies it for another reset
    next half-cycle, forever (confirmed arithmetically: 22.36-1=21.36 <
    0.05*499=24.95). ion (0.0001-100, a 1e6 ratio) and ioff (1e-7-1, a
    1e7 ratio) have the exact same problem, just more extreme -- almost
    ANY physically plausible value sits "within 5% of the lower bound"
    in LINEAR terms when the upper bound is a million times larger.
    Fix: check proximity to each edge's own ORDER OF MAGNITUDE instead
    of its raw distance. _geo_reset_value() is BY CONSTRUCTION the exact
    log-space midpoint between the two edges (50% of the log-span from
    either one), so for any margin_frac < 0.5 it can never itself fall
    inside the zone this function flags -- no self-trap is possible,
    regardless of how wide or lopsided a parameter's bounds are."""
    lo_a, hi_a = sorted((abs(lo_b), abs(hi_b)))
    log_lo, log_hi = np.log(max(lo_a, 1e-300)), np.log(max(hi_a, 1e-300))
    span = log_hi - log_lo
    log_v = np.log(max(abs(v), 1e-300))
    return log_v < log_lo + margin_frac * span or log_v > log_hi - margin_frac * span


def chain_half_cycles(t, V, I, bounds_by_cycle, cycles, start_cycle, n_cycles,
                       x0_dict=None, bounds_dict=None, max_nfev=2000,
                       ron_roff_max_change_frac=0.25, ron_roff_reset_value=22.0,
                       margin_frac=0.05,
                       rmse_spike_factor=3.0, rmse_spike_min_history=20, rmse_baseline_window=50,
                       fix_gam=None,
                       method="lsq", ga_kwargs=None, dga_kwargs=None, ga_lsq_kwargs=None,
                       progress_every=None, progress_cb=None):
    """Walks N consecutive RAW logical cycles as an alternating chain of
    0->Vmax->0 (LRS/SET_SIDE free) / 0->Vmin->0 (HRS/RESET_SIDE free)
    half-cycle fits (by request, 2026-10-14, confirming the intended
    design: "the parameters for the hrs between 0-vmax-0 correspond to
    the previous half cycle, and the lrs params between 0-vmin-0
    correspond also to the previous half cycle?" -- yes): each half-
    cycle's own FREE side is a genuine fit_region_pair() joint
    least_squares fit; the OTHER (fixed) side is held at the
    IMMEDIATELY PRECEDING half-cycle's own freshly-fit value -- the
    SAME "hold the other side fixed at the prior window's own value"
    convention fit_all_cycles.py's own production chain already uses,
    now applied to the 0-Vmax-0/0-Vmin-0 window definition instead of
    the old detection-anchored vreset-vset-vreset one. g0 is carried
    continuously via g0_override from each half-cycle's own final g to
    the next's starting point. The VERY FIRST half-cycle (no prior to
    warm-start from) fits all 12 parameters (and g0) freely -- same
    "no prior fit" convention as fit_all_cycles.py's own very first
    window.

    ron_roff_max_change_frac (by request, 2026-10-14, after the first
    real-data chain run showed excellent overall rmse but ron bouncing
    between ITS OWN bounds every single half-cycle -- 114->12->1.00
    (pinned)->...->500 (pinned)->2.5->500 (pinned) -- the classic
    compensating-degeneracy pattern this project has repeatedly found:
    a window/chain can fit beautifully while individual parameters
    wander unphysically): a DYNAMIC bound on ron and roff,
    +-ron_roff_max_change_frac (default 0.25, was 0.5 -- tightened by
    request after the FULL-DATASET run below exposed a second bug the
    50% version let through) of the IMMEDIATELY PRECEDING half-cycle's
    own value for that parameter, clipped to the global DEFAULT_BOUNDS
    range -- not just the fixed, wide [1,500] range every half-cycle
    could freely roam. Applied from the SECOND half-cycle onward (the
    very first has no prior value to base it on, so uses the plain
    global bounds, same as always). Harmless to compute for whichever
    of ron/roff isn't this half-cycle's own free side -- a bound on an
    already-fixed parameter has no effect.

    Confirmed on the FULL 16603-cycle real dataset (by request, "can
    you launch and fit all the experimental data?"): a PURELY relative
    bound can ratchet a parameter into a corner it can never escape --
    ron fell in a perfect geometric halving sequence (46.9->23.5->
    11.7->...->1.0, the global floor) within the first 10 cycles, then
    stayed PINNED AT EXACTLY 1.0 for the remaining 16593 cycles (99.95%
    of the whole dataset), because every subsequent step's own +-50%
    window around an already-tiny value is itself tiny in absolute
    terms ([0.5, 1.5] around ron=1) -- no matter how strongly later
    data wanted a larger ron, the window could never admit it. rmse
    stayed misleadingly reasonable throughout (other parameters, likely
    aon/ion, compensated) -- the SAME "fits beautifully while wandering
    unphysically" pattern, just hidden one level deeper.

    Fix, first scoped to ron/roff only (by request, "if it gets pinned,
    either to up or lower bounds, set it to 22 and go on"), then to
    vs/vr too (by request, after cycle 3887 showed THE SAME entrapment
    pattern in a different parameter pair: vs and etas both sitting
    exactly at the same value, pinned near their own upper bounds,
    unmoving across 7 straight logical cycles) -- and FINALLY made
    general for all 12 params (by request, 2026-10-14, after the
    FULL-DATASET run -- the one that motivated the vs/vr fix above --
    turned out to still collapse catastrophically at cycle 9169: ioff
    landed on its own global lower bound 1e-7 and stayed PINNED there
    for the remaining 14869 half-cycles, 51% of the entire run, dragging
    every other parameter into a persistently worse-fitting compensating
    branch (median rmse jumped ~5x and never recovered) -- because ioff,
    unlike ron/roff/vs/vr, had NO safety net at all. The same bug was
    always going to recur one parameter at a time; ion/aon/aoff/etas/
    etar/gam/g0 are just as capable of it):

    after each half-cycle's own fit, EVERY one of the 12 params is
    checked against its own GLOBAL bounds ONLY -- if it's within
    margin_frac (default 0.05, i.e. 5%) of EITHER edge, IN LOG SPACE
    (_near_bound_log() -- a LINEAR-space margin was tried first and
    immediately self-trapped roff/ion/ioff, each one frozen at exactly
    its own reset value for 2000 straight half-cycles right from the
    start of a fresh run: their bounds are lopsided enough (roff 1-500,
    ion 1e-4-100, ioff 1e-7-1) that the reset value itself kept landing
    back inside the margin zone -- see _near_bound_log()'s own
    docstring) -- it's reset to _geo_reset_value() of its own global
    bounds: the sign-preserving geometric mean of the two edges, a
    scale-appropriate 'central' value whether the parameter ranges
    linearly (ron: 1-500) or across many orders of magnitude (ioff:
    1e-7-1).

    An EARLIER version of this check ALSO reset ron/roff whenever they
    landed exactly at the edge of THEIR OWN dynamic +-ron_roff_max_
    change_frac window (not just the global bound) -- removed (by
    request, 2026-10-14/15) after the log-space fix above was itself
    validated on cycles 2000-2999 and showed roff/aoff frozen at exactly
    their reset values for the ENTIRE 1000-cycle run: the real data in
    that window wants roff far higher (~90-110) than its reset pivot
    (22.36), but resetting to the pivot EVERY time it reached its own
    narrow window's edge meant it could only ever take one small step
    before being yanked straight back to the start -- never given the
    chance to walk incrementally toward where the data actually pulls
    it. Touching your OWN current window's edge is normal, expected
    behavior for a value migrating toward a distant target; only being
    near the GLOBAL bound is genuine entrapment (the original ron-
    ratchets-to-1.0 bug this project started from) -- and the log-space
    global check above already catches that case on its own, just
    slightly later in the ratchet (once ron is within 5% of the global
    floor in log space) rather than at the exact moment it first touches
    some arbitrary per-step window edge. For every OTHER parameter (no
    dynamic window, so "this step's own bound" was always just the
    global bound anyway) this removal changes nothing.
    This reproduces the originally hand-picked ron/roff reset value of
    22 almost exactly (sqrt(1*500)=22.36) -- independent confirmation
    it's the right general rule, not a per-parameter magic constant.
    Harmless to compute for whichever parameters aren't this half-
    cycle's own free side -- a reset on an already-fixed parameter has
    no effect on this step, only on what gets carried into the next one.

    method/ga_kwargs/dga_kwargs/ga_lsq_kwargs (by request, 2026-10-16,
    "return to the global fit... add it as an alternative way... try to
    use a genetic algorithm"; "directed_ga" and "ga_lsq" both added
    2026-10-17, see fit_region_pair_directed_ga()'s and
    fit_region_pair_ga_seeded_lsq()'s own docstrings for why):
    method="lsq" (default, unchanged behavior) calls fit_region_pair()'s
    gradient-based joint least_squares for each half-cycle's free side.
    method="ga" calls fit_region_pair_ga() instead (differential_evolution,
    isotropic random mutation). method="directed_ga" calls
    fit_region_pair_directed_ga() instead (a custom GA whose mutations are
    drawn from a PCA of the local Jacobian, biased toward the stiffest
    directions, instead of isotropically). method="ga_lsq" calls
    fit_region_pair_ga_seeded_lsq() instead (a "desperate test" hybrid:
    GA generates a population, EVERY member of it -- or `top_frac` of it
    -- gets its own local least_squares polish, and the best-fitting
    polished result wins). Same chaining/pin-escape-margin machinery
    regardless of method, only the per-half-cycle fit call differs. For
    "ga"/"directed_ga"/"ga_lsq", this function also computes THIS half-
    cycle's own direct vs/vr estimate (extract_vs_vr(), the production
    max(d2I/dV2) threshold detector) and passes it through as that
    method's own lock_values -- vs for a 'max' half-cycle, vr for a 'min'
    one (whichever is this call's free side) -- so the search stays
    anchored near the one directly-measured signal available for this
    project's most failure-sensitive parameters, rather than exploring
    vs/vr as freely as everything else. ga_kwargs/dga_kwargs/ga_lsq_kwargs
    is passed straight through to fit_region_pair_ga()/
    fit_region_pair_directed_ga()/fit_region_pair_ga_seeded_lsq()
    respectively (only the one matching `method` is actually used).

    rmse_spike_factor (method="lsq" only; by request 2026-10-16, "detect
    rmse spikes directly ... as an independent reset/retry trigger" --
    found the SAME day a full-16602-cycle real-data run derailed
    catastrophically: a single half-cycle's fit landed vs/vr just past
    the window's own applied voltage extremes, a literal zero-gradient
    dead zone for the gradient-based optimizer -- confirmed directly by
    perturbing vr there and finding bit-for-bit IDENTICAL rmse across a
    wide range, i.e. no local signal to climb out on -- which froze
    vs/vr bit-for-bit for the remaining 11445 half-cycles (69% of the
    whole dataset), because the existing margin_frac pin-escape check
    only guards GLOBAL-bound proximity and this failure had nothing to
    do with that (vr missed the 5% rescue zone by ~0.03V; the real issue
    was window-voltage-reachability, a different axis entirely). The
    failing half-cycle's own rmse WAS elevated (~10x the surrounding
    half-cycles') even though nowhere near a dramatic blowup -- this
    catches that directly instead of requiring bound proximity: each
    half-cycle's fit["rmse"] is compared against the median of the last
    rmse_baseline_window (default 50) half-cycles' own recorded rmse
    (skipped until rmse_spike_min_history, default 20, half-cycles of
    history exist, and skipped entirely for the very first half-cycle).
    If it exceeds rmse_spike_factor (default 3.0) times that local
    baseline, the SAME half-cycle is retried ONCE with its own free
    side's params reset to their _geo_reset_value() (ron/roff to
    ron_roff_reset_value, as usual) as x0 instead of the carried-forward
    value, and whichever of the two attempts has the LOWER rmse is kept
    -- always a genuine, self-consistent refit (never a post-hoc
    override of params without resimulating, which would reintroduce
    the exact rmse/params recording mismatch bug fixed 2026-10-16
    earlier the same day). Pass rmse_spike_factor=None to disable.

    fix_gam (by request 2026-10-17, "etas/etar are usually pinned, it
    looks wrong" -- confirmed: etas pinned 48.8%/etar 10.8% of half-
    cycles at a 300-cycle spot check, --etas-etar-upper-bound 150):
    root-caused to memory memristor_model_dmm_relations_restrictions's
    ALREADY-DOCUMENTED gam/etar ridge ("a wide RIDGE, not a sharp
    minimum... this is why the production chain fixes gam=0.22 by
    default rather than fitting it freely") -- this chain's own
    RESET_SIDE frees gam ALONGSIDE etar, reproducing exactly that
    ridge. Passing fix_gam=0.22 (not None) holds gam at that value for
    the WHOLE chain (added to `fixed` every half-cycle, and to
    `params` directly, since fit_region_pair()'s own `fixed` dict only
    supplies NAMES to exclude from optimization -- the held VALUE
    always comes from x0_dict/params, not from `fixed`'s own value, a
    real gotcha found while testing this). Confirmed on cycles
    2000-2249: etar pinning 39.2%->10.8% with fix_gam=0.22, no peak-
    current accuracy cost. Does NOT help etas (gam is RESET_SIDE only,
    never touches the SET-side fit) -- etas's own high pin rate needs
    `--etas-etar-upper-bound` raised instead (300, not 150: cuts etas
    pin 48.8%->17.6%/etar 10.8%->3.2% with no accuracy cost; 500 is
    catastrophic, 47.89%/30.32% median peak error -- reproduces the
    original unbounded-wander failure, just at a higher threshold).

    Returns a list of per-half-cycle dicts (in chain order): cycle,
    kind ('max' or 'min'), i_lo/i_peak/i_hi (global indices),
    params_dict, rmse, I_model, g_hist, t_w, V_w, I_w."""
    results = []
    rmse_history = []
    params = {**dmm.NOMINAL, **(x0_dict or {})}
    if fix_gam is not None:
        params["gam"] = fix_gam
    g0_override = None
    first = True
    base_bounds = {**DEFAULT_BOUNDS, **(bounds_dict or {})}
    start_pos = cycles.index(start_cycle)
    window_cycles = cycles[start_pos:start_pos + n_cycles]
    for ci, cyc in enumerate(window_cycles):
        if progress_cb is not None:
            progress_cb(ci, len(window_cycles), cyc, results)
        elif progress_every and ci % progress_every == 0:
            print(f"chain_half_cycles: {ci}/{len(window_cycles)} logical cycles (cycle {cyc})", flush=True)
        i0, i1 = bounds_by_cycle[cyc]
        V_cyc = V[i0:i1]
        (lo_max, hi_max), (lo_min, hi_min) = find_half_cycle_spans(V_cyc)
        for kind, (lo, hi) in [("max", (lo_max, hi_max)), ("min", (lo_min, hi_min))]:
            i_lo, i_hi = i0 + lo, i0 + hi
            if i_hi <= i_lo:
                continue
            i_peak = i_lo + (int(np.argmax(V[i_lo:i_hi + 1])) if kind == "max"
                              else int(np.argmin(V[i_lo:i_hi + 1])))
            other_side = RESET_SIDE if kind == "max" else SET_SIDE
            fixed = {} if first else {n: params[n] for n in other_side}
            if fix_gam is not None:
                # fit_region_pair()'s own `fixed` only supplies NAMES to exclude
                # from optimization -- the actual held value always comes from
                # x0_dict (params, here), not from `fixed`'s own value (confirmed
                # directly 2026-10-17 after a gam-value sweep gave IDENTICAL
                # results for gam=0.1/0.22/0.5 -- a test-script bug, not a real-
                # code one: `params["gam"]` was never actually being set to the
                # swept value, only added to `fixed`'s key set). Setting it once
                # above (before the loop) is enough since gam, once fixed, is
                # never touched by anything else -- repeated here for robustness
                # against future refactors, and because `params["gam"]` on the
                # FIRST half-cycle only gets set above, not through this branch.
                params["gam"] = fix_gam
                fixed["gam"] = fix_gam
            this_bounds = dict(base_bounds)
            if not first:
                for name in ("ron", "roff"):
                    prev = params[name]
                    lo_b = max(base_bounds[name][0], prev * (1.0 - ron_roff_max_change_frac))
                    hi_b = min(base_bounds[name][1], prev * (1.0 + ron_roff_max_change_frac))
                    this_bounds[name] = (lo_b, hi_b)
            retried = False
            if method in ("ga", "directed_ga", "ga_lsq"):
                # ga_lock_names restricts which of vs/vr actually get the direct-
                # extraction lock (by default just vs -- NOT vr) -- found 2026-10-16,
                # same day as adding this: extract_vs_vr()'s own apparent-threshold
                # vr estimate landed around -1.0 to -1.09 V on real cycles 2000-2009,
                # while the SAME half-cycles' own genuinely-fitted (lsq) vr sits at
                # -0.13 to -0.22 V -- off by ~0.9V, a ~600% relative error. This is
                # NOT a new finding: memory memristor_model_dmm_relations_restrictions
                # already documented, from rigorous synthetic ground truth, that the
                # apparent/curvature-extracted vr correlates only weakly with the true
                # model vr ("especially bad, ~210% median relative error") and that
                # FIXING vr to it "breaks the fit completely" -- this GA lock was
                # repeating that exact mistake with a tighter leash. vs's own apparent
                # estimate, by contrast, matched lsq's fitted vs within ~0.05V on the
                # same cycles (consistent with vs being the much more reliable one of
                # the two per that same memory) -- confirmed directly by the GA's own
                # SET-side quality being fine while RESET-side was consistently bad
                # (median RESET peak err 15.6%, 75% of cycles >10%, vs. lsq's 0% on
                # the identical window) BEFORE this fix. ga, directed_ga (added
                # 2026-10-17), and ga_lsq (added 2026-10-17, see
                # fit_region_pair_ga_seeded_lsq()'s own docstring) all share this
                # same lock mechanism.
                this_method_kwargs = {"ga": ga_kwargs, "directed_ga": dga_kwargs,
                                       "ga_lsq": ga_lsq_kwargs}[method]
                lock_names = (this_method_kwargs or {}).get("lock_names", ("vs",))
                v_set_w, v_reset_w = extract_vs_vr(t[i_lo:i_hi + 1], V[i_lo:i_hi + 1], I[i_lo:i_hi + 1])
                lock_values = {}
                if kind == "max" and v_set_w is not None and "vs" in lock_names:
                    lock_values["vs"] = float(v_set_w)
                if kind == "min" and v_reset_w is not None and "vr" in lock_names:
                    lock_values["vr"] = float(v_reset_w)
                method_kwargs_clean = {k: v for k, v in (this_method_kwargs or {}).items() if k != "lock_names"}
                fit_fn = {"ga": frp.fit_region_pair_ga, "directed_ga": frp.fit_region_pair_directed_ga,
                          "ga_lsq": frp.fit_region_pair_ga_seeded_lsq}[method]
                fit = fit_fn(t, V, I, i_lo, i_peak, i_hi, x0_dict=params,
                             bounds_dict=this_bounds, fixed=fixed,
                             g0_override=g0_override, lock_values=lock_values,
                             **method_kwargs_clean)
            else:
                fit = frp.fit_region_pair(t, V, I, i_lo, i_peak, i_hi, x0_dict=params,
                                           bounds_dict=this_bounds, fixed=fixed,
                                           g0_override=g0_override, max_nfev=max_nfev)
                # rmse-spike retry (by request 2026-10-16, "detect rmse spikes
                # directly ... as an independent reset/retry trigger" -- see the
                # function docstring for the full root-cause: a half-cycle's lsq
                # fit can land vs/vr just past the window's OWN applied voltage
                # range, a literal zero-gradient dead zone the optimizer can
                # never locally climb out of once it's the carried-forward x0 --
                # confirmed on real data to freeze vs/vr bit-for-bit for 11445
                # straight half-cycles (69% of a full 16602-cycle run) because
                # the existing margin_frac check only guards GLOBAL-bound
                # proximity, not window-voltage-reachability. The failing
                # half-cycle's own rmse WAS elevated (~10x neighbors) even
                # though unremarkable in isolation -- catch THAT directly: if
                # this fit's rmse is rmse_spike_factor x worse than the recent
                # local median, retry ONCE from this half-cycle's own free-side
                # params reset to their _geo_reset_value() (well inside the
                # window's own reachable range by construction) instead of the
                # possibly-already-bad carried-forward x0, and keep whichever
                # attempt actually fits better -- always a genuine resimulated
                # fit, never a post-hoc override (that already burned this
                # project once, see the "fitted_params" comment just below).
                if (rmse_spike_factor is not None and not first
                        and len(rmse_history) >= rmse_spike_min_history):
                    baseline = float(np.median(rmse_history[-rmse_baseline_window:]))
                    if baseline > 0 and fit["rmse"] > rmse_spike_factor * baseline:
                        free_side = SET_SIDE if kind == "max" else RESET_SIDE
                        retry_x0 = dict(params)
                        for name in free_side:
                            lo_g, hi_g = this_bounds[name]
                            retry_x0[name] = (ron_roff_reset_value if name in ("ron", "roff")
                                               else _geo_reset_value(lo_g, hi_g))
                        retry_fit = frp.fit_region_pair(t, V, I, i_lo, i_peak, i_hi, x0_dict=retry_x0,
                                                         bounds_dict=this_bounds, fixed=fixed,
                                                         g0_override=g0_override, max_nfev=max_nfev)
                        if retry_fit["rmse"] < fit["rmse"]:
                            fit = retry_fit
                            retried = True
            # fitted_params is exactly what fit_region_pair[_ga] simulated for THIS
            # half-cycle (what fit["rmse"]/fit["I_model"] actually correspond to) --
            # recorded as-is below. params (mutated below by the pin-escape reset)
            # becomes the CARRIED-FORWARD state for the next half-cycle's x0/fixed
            # only; bug fixed 2026-10-16 (found while diagnosing cycles 1500-1999
            # looking catastrophic on resimulation despite a fine recorded rmse):
            # this used to reset `params` in place and record THAT (post-reset)
            # dict as this row's own params_dict, while still recording fit["rmse"]
            # (the PRE-reset rmse) alongside it -- a genuine mismatch between the
            # saved rmse and the saved params for every half-cycle where any of the
            # 12 params (not just this side's free ones) happened to be near its own
            # global edge, silently corrupting that row's own recorded fit (and
            # everything resimulated from it) even though the chain's own future
            # steps were correctly protected from entrapment either way.
            fitted_params = dict(fit["params_dict"])
            rmse_history.append(fit["rmse"])
            params = dict(fitted_params)
            for name in dmm.PARAM_NAMES:
                lo_g, hi_g = base_bounds[name]  # the plain GLOBAL bound -- the ONLY check now
                v = params[name]
                near_edge = (_near_bound_linear(v, lo_g, hi_g, margin_frac) if name in LINEAR_MARGIN_PARAMS
                             else _near_bound_log(v, lo_g, hi_g, margin_frac))
                if near_edge:
                    # ron/roff (by request, 2026-10-14/15) keep a FIXED reset
                    # value (22, not a bounds-derived one) -- _geo_reset_value
                    # of their own global bounds would depend on whatever
                    # ron_roff_upper_bound happens to be passed in (e.g. 500
                    # vs the 75 being tried below), and 22 is the value
                    # already validated against real data earlier this
                    # session. Everyone else still uses _geo_reset_value().
                    params[name] = (ron_roff_reset_value if name in ("ron", "roff")
                                     else _geo_reset_value(lo_g, hi_g))
            g0_override = float(fit["g_hist"][-1])
            results.append(dict(cycle=cyc, kind=kind, i_lo=i_lo, i_peak=i_peak, i_hi=i_hi,
                                 params_dict=fitted_params, rmse=fit["rmse"], I_model=fit["I_model"],
                                 g_hist=fit["g_hist"], t_w=t[i_lo:i_hi + 1], V_w=V[i_lo:i_hi + 1],
                                 I_w=I[i_lo:i_hi + 1], retried=retried))
            first = False
    return results


def chain_results_to_dataframe(results):
    """Converts chain_half_cycles()'s own return value (a list of dicts,
    each carrying the heavy arrays I_model/g_hist/t_w/V_w/I_w alongside
    the scalar fields) into a plain pandas DataFrame of JUST the scalar
    per-half-cycle summary (cycle, kind, i_lo/i_peak/i_hi, rmse, all 12
    params) -- NOT the arrays, which are far too large to keep for a
    long chain (by request, 2026-10-14, "can't you share the results
    and the functions between the cli and the gui? It seems
    inefficient to have them duplicated" -- this is the ONE place that
    defines what a saved/displayed chain result row looks like; both
    fit_chain_cli.py and analyze_rolling_fit.py's own GUI "Save chain
    to CSV" button call this instead of each reimplementing their own
    row-building loop)."""
    import pandas as pd
    rows = []
    for r in results:
        row = dict(cycle=r["cycle"], kind=r["kind"], i_lo=r["i_lo"], i_peak=r["i_peak"], i_hi=r["i_hi"],
                    rmse=r["rmse"], retried=r.get("retried", False))
        row.update(r["params_dict"])
        rows.append(row)
    return pd.DataFrame(rows)
