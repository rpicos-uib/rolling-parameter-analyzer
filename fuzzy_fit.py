"""Rule-based ("fuzzy logic") parameter extractor -- an alternative to
fit_region_pair.py's blind least_squares optimizer (by request,
2026-10-12). See RULES_DOC.md for the full rule table with physical
rationale, written at paper-section detail.

MOTIVATION: a full session's worth of findings showed a blind gradient
optimizer struggling with this model in several distinct ways:
  - a single switching event underdetermines 5-6 free params (SET_SIDE
    or RESET_SIDE) when fit alone -- fit_all_cycles.py's own module
    docstring, and the reason it freezes one side per window;
  - freezing one side per window bakes in a geometric (curvature-
    detector) window boundary that doesn't match where vs/vr actually
    "activate" in the model's own dynamics (the vr/apparent-crossing
    gap, self-quenching, incomplete-SET-saturation findings);
  - fitting several cycles jointly with ALL 12 params free per event
    (fit_cycle_block(), same day) is tractable only with a good warm
    start, and even then plateaus ~2x worse than the per-event chain --
    a cold, 22+ dimensional joint least_squares call is a hard landscape
    for a finite-difference Jacobian to navigate;
  - a cold free fit can wander into unphysical compensating
    combinations (ron/etar bound-hugging while g never truly switches)
    that a pure rmse objective can't see at all (the g_range gate was
    added specifically because rmse stayed low throughout).

Instead of ONE joint numerical optimization, this module extracts a
handful of INTERPRETABLE FEATURES from both the REAL data and the
model's OWN simulated curve over one window, and for EACH of the 12
parameters applies at least two independent RULES -- simple, physically
-motivated comparisons ("is the model's transition narrower than the
data's?", "is the model's current at Vmax higher than the data's?"),
each voting increase/decrease/no-change. Votes are combined and each
parameter is nudged by a SMALL step in the voted direction (log-space
for dmm.LOG_PARAMS, linear otherwise -- same convention fit_region_pair
.py uses), for several ROUNDS with a decaying step size, re-simulating
and re-extracting the model's own features each round.

This is deliberately NOT gradient-based -- no Jacobian, no objective
gradient anywhere. Every rule is a one-sentence, physically-grounded
heuristic; a domain expert could read (and audit) the whole rule table
without any optimization theory background.
"""
import numba
import numpy as np

import dmm_model as dmm
import fit_region_pair as frp
from data_utils import DEFAULT_BOUNDS, moving_average

SET_SIDE = ["vs", "etas", "ron", "aon", "ion"]
RESET_SIDE = ["vr", "etar", "gam", "roff", "aoff", "ioff"]


# ---------------------------------------------------------------- features
@numba.njit(cache=True)
def _peak_width_height_core(y, t, lo, hi):
    """Numba-jitted core of _local_peak_width_height (by request,
    2026-10-12, same convention as dmm_model.py's own _simulate_loop):
    this runs 2-3x per extract_features() call, and extract_features()
    runs ~2x per free parameter per round inside fuzzy_fit()'s own
    iteration -- thousands of calls per window, almost all spent in
    this tight left/right scan. Pure array indexing/comparisons, no
    Optional/None handling (that stays in the thin Python wrapper
    below, same split dmm_model.simulate()/_simulate_loop() uses)."""
    seg = y[lo:hi]
    t_seg = t[lo:hi]
    n = len(seg)
    if n == 0:
        return np.nan, np.nan
    peak_local = 0
    best = -1.0
    any_finite = False
    for i in range(n):
        v = seg[i]
        if np.isfinite(v):
            any_finite = True
            av = abs(v)
            if av > best:
                best = av
                peak_local = i
    if not any_finite:
        return np.nan, np.nan
    height = best
    half = height / 2.0
    L = peak_local
    while L > 0 and abs(seg[L]) >= half:
        L -= 1
    R = peak_local
    while R < n - 1 and abs(seg[R]) >= half:
        R += 1
    width = t_seg[R] - t_seg[L] if R > L else np.nan
    return height, width


def _local_peak_width_height(y, t, idx_guess, half_window):
    """Height and FWHM (in seconds) of |y|'s own highest peak within
    +-half_window samples of idx_guess -- used on smoothed dI/dt to
    characterize how SHARP a transition is (width) and how STRONG
    (height), for both etaS/etaR (narrow half_window, the sharp onset)
    and gam (wide half_window, the whole tail -- see RULES_DOC.md).
    Thin wrapper (handles the Optional idx_guess) around the Numba-
    jitted _peak_width_height_core()."""
    if idx_guess is None or not np.isfinite(idx_guess):
        return float("nan"), float("nan")
    idx_guess = int(idx_guess)
    lo = max(0, idx_guess - half_window)
    hi = min(len(y), idx_guess + half_window + 1)
    height, width = _peak_width_height_core(y, t, lo, hi)
    return float(height), float(width)


def extract_features(t, V, I, smooth_n=10, probe_offset=15, gam_half_window=60, eta_half_window=12,
                      anchor_set=None, anchor_reset=None, i_set_known=None, i_reset_known=None):
    """Features from ONE window's own curve -- call this identically on
    the REAL (t,V,I) and on the MODEL's own simulated (t,V,I_model);
    every RULE_SPECS rule compares the two dicts this returns.

    i_set_known/i_reset_known: the window's OWN exact Vset1/Vreset1
    local indices, when known (e.g. fit_region_pair.find_fit_window()'s
    own i_vset1-i_vreset1 and 0) -- used DIRECTLY instead of re-running
    estimate_cycle_thresholds. Fixed (2026-10-12) after calibration
    caught a real bug: a fit_region_pair-style window spans Vreset1 ->
    Vset1 -> Vreset2, which has TWO separate "V>=0 & rising" stretches
    (the actual Vset1 crossing, AND the later full rise to Vmax within
    region2) -- estimate_cycle_thresholds's own global max(d2I/dV2)
    search picked the WRONG one (Vmax, not Vset1) on the very first
    real test window, silently poisoning most downstream features.
    This is the SAME ambiguity fit_region_pair.apparent_threshold_gap()
    already had to guard against (restricting its own search to the
    window's tail) -- see RULES_DOC.md's "what didn't work" section.
    When the known indices aren't available (e.g. exploring a NEW
    curve with no prior window structure), falls back to
    estimate_cycle_thresholds as before -- still usable, just carries
    the same re-detection risk documented above.

    i_set/i_reset: the indices actually used (i_set_known/i_reset_known
    if given, else estimate_cycle_thresholds's own best guess).
    v_set/v_reset: V at those indices (nan if not found) -- diagnostic
    only, not used by any rule.
    i_vmax/i_vmin: index of this window's own V extrema (unambiguous,
    plain argmax/argmin); I_vmax/I_vmin: I there.
    didt: smoothed dI/dt (moving_average(np.gradient(I,t), smooth_n)) --
    well-conditioned everywhere (dt>0 always, unlike dI/dV at a V
    turning point).
    set_peak_height/set_peak_width: _local_peak_width_height of didt
    around i_set, narrow window -- characterizes etaS.
    reset_peak_height/reset_peak_width: ditto around i_reset -- etaR.
    reset_wide_width: _local_peak_width_height of didt around i_reset,
    WIDE window -- characterizes gam's self-quenching tail (etaR's own
    sharp onset is a small fraction of this wider span; the EXTRA width
    beyond what etaR alone would produce is gam's signature).

    anchor_set/anchor_reset: GLOBAL ANCHOR indices for the vs/vr probe
    features below -- pass the DATA curve's own i_set here when
    extracting the MODEL's features, so both curves are probed at the
    EXACT SAME physical location (sample index). Defaults to this
    curve's own i_set/i_reset (correct when extracting the DATA's own
    features, which IS the anchor).
    I_probe_set_near/_far/_far2: I at anchor_set + a SMALL/MEDIUM/LARGE
    offset -- index-anchored (not re-detected) checks on how far along
    the SET transition this curve is, past the data's own threshold.
    _near/_far alone are used as absolute LEVEL comparisons (vs, ion,
    etc); _far2 exists so aon/aoff (see RULE_SPECS) can compare a RATIO
    of two probes instead -- near V=0, I~=ion*Ka*VA to leading order, so
    a ratio of two probe levels cancels most of ion/ron's own scale
    factor, isolating aon's curvature (how much faster I grows than a
    straight extrapolation from the near probe). I_probe_reset_*:
    mirror, RESET side."""
    t, V, I = np.asarray(t), np.asarray(V), np.asarray(I)
    didt = moving_average(np.gradient(I, t), smooth_n)
    if i_set_known is not None and i_reset_known is not None:
        i_set, i_reset = int(i_set_known), int(i_reset_known)
    else:
        i_set, i_reset = frp.estimate_cycle_thresholds(t, V, I, smooth_n)
    i_vmax, i_vmin = int(np.argmax(V)), int(np.argmin(V))

    set_h, set_w = _local_peak_width_height(didt, t, i_set, eta_half_window)
    reset_h, reset_w = _local_peak_width_height(didt, t, i_reset, eta_half_window)
    _, reset_wide_w = _local_peak_width_height(didt, t, i_reset, gam_half_window)

    a_set = anchor_set if anchor_set is not None else i_set
    a_reset = anchor_reset if anchor_reset is not None else i_reset

    def _probe(arr, idx, offset):
        if idx is None:
            return float("nan")
        j = idx + offset
        return float(arr[j]) if 0 <= j < len(arr) else float("nan")

    return dict(
        i_set=i_set, i_reset=i_reset,
        v_set=float(V[i_set]) if i_set is not None else float("nan"),
        v_reset=float(V[i_reset]) if i_reset is not None else float("nan"),
        i_vmax=i_vmax, i_vmin=i_vmin, I_vmax=float(I[i_vmax]), I_vmin=float(I[i_vmin]),
        set_peak_height=set_h, set_peak_width=set_w,
        reset_peak_height=reset_h, reset_peak_width=reset_w, reset_wide_width=reset_wide_w,
        I_probe_set_near=_probe(I, a_set, probe_offset), I_probe_set_far=_probe(I, a_set, 3 * probe_offset),
        I_probe_set_far2=_probe(I, a_set, 6 * probe_offset),
        I_probe_reset_near=_probe(I, a_reset, probe_offset), I_probe_reset_far=_probe(I, a_reset, 3 * probe_offset),
        I_probe_reset_far2=_probe(I, a_reset, 6 * probe_offset),
        # dI/dt probes (by request, 2026-10-14): SAME anchor+offset convention
        # as the I probes above, just reading the already-computed smoothed
        # `didt` array instead of `I` -- so every parameter's own rule set can
        # compare the DERIVATIVE's shape between data/model, not just the
        # function's own level, the same "fit I AND dI/dt" discipline
        # fit_region_pair.py's own joint least_squares residual already uses
        # (fit_derivative=True there) but fuzzy_fit's rules didn't before.
        didt_probe_set_near=_probe(didt, a_set, probe_offset),
        didt_probe_set_far=_probe(didt, a_set, 3 * probe_offset),
        didt_probe_reset_near=_probe(didt, a_reset, probe_offset),
        didt_probe_reset_far=_probe(didt, a_reset, 3 * probe_offset),
    )


# ------------------------------------------------------------------ rules
# Each rule maps (data_features, model_features) -> (data_val, model_val)
# for ONE scalar comparison. _sign(data_val, model_val) is the RAW vote
# (+1 if model_val should rise to match data_val). Whether "model_val
# should rise" means "the PARAMETER should rise" (polarity +1) or "fall"
# (polarity -1) is NOT hand-derived -- a first pass tried that by hand
# and got it wrong in ways that weren't obvious from reading the code
# (e.g. two rules for the SAME parameter voting opposite directions, or
# a rule voting the SAME way regardless of whether the parameter was
# perturbed too high or too low -- see RULES_DOC.md's "what didn't
# work" section for the full story). Polarity is instead determined
# EMPIRICALLY by calibrate_polarities() against a synthetic ground
# truth and cached in RULE_POLARITY; a rule that isn't CONSISTENT
# across both a too-high and a too-low synthetic perturbation is
# dropped (not silently kept with a guessed sign).

def _sign(data_v, model_v, tol_frac=0.02):
    """+1 if model_v should INCREASE to match data_v (model < data),
    -1 if it should decrease, 0 if within tol_frac of |data_v| (or
    either is nan)."""
    if not (np.isfinite(data_v) and np.isfinite(model_v)):
        return 0
    tol = tol_frac * max(abs(data_v), 1e-12)
    diff = data_v - model_v
    if abs(diff) <= tol:
        return 0
    return 1 if diff > 0 else -1


def _feat_pair(key):
    """Simple rule: compare the SAME named feature on both curves."""
    return lambda fd, fm: (fd[key], fm[key])


def _gam_vmin_rule(fd, fm):
    """How far I has moved from I_probe_reset_near toward I_vmin, on
    each curve -- a "how much of the reset has completed by Vmin"
    proxy; gam's self-quenching should slow this down, so a model
    that's moved FURTHER than the data by Vmin needs MORE gam."""
    d = abs(fd["I_probe_reset_near"] - fd["I_vmin"])
    m = abs(fm["I_probe_reset_near"] - fm["I_vmin"])
    return d, m


def _ratio(near_key, far_key):
    """Ratio of two probe levels on the SAME curve -- near V=0,
    I~=ion*Ka*VA to leading order, so ion/ion's own scale factor is
    roughly the SAME in both probes and mostly cancels in the ratio,
    leaving mostly aon/aoff's own curvature (by request, 2026-10-12,
    after aon's level-based rules abstained on every real test: aon
    only shows up in a single window's current LEVEL blended with
    ion/ron, never alone -- see fuzzy_fit's own module docstring)."""
    def rule(fd, fm):
        d = fd[far_key] / (fd[near_key] + 1e-15)
        m = fm[far_key] / (fm[near_key] + 1e-15)
        return d, m
    return rule


# Each parameter: >=2 independent feature-comparison rules (by request).
# ion/ioff and ron/roff are only partially separable from a single
# window on physical grounds alone (both affect the SAME peak current
# jointly) -- see RULES_DOC.md's identifiability section; the fuzzy
# iteration refines them one small step at a time rather than solving a
# possibly-degenerate system exactly, same honest limitation a blind
# optimizer has. aon/aoff use RATIO rules instead of level rules (see
# _ratio()'s own docstring) specifically to escape this same
# degeneracy with ion/ron, rather than live with it.
RULE_SPECS = {
    "vs": [_feat_pair("I_probe_set_near"), _feat_pair("I_probe_set_far")],
    # vr's 2nd rule (by request, 2026-10-14): was I_probe_reset_far (a
    # 2nd ARBITRARY sample offset, 3x the near one) -- replaced with
    # I_vmin, the window's own current AT its V-extremum (already
    # computed, no tunable offset at all, "don't go so far away").
    # Validated against the SAME synthetic Monte Carlo harness as
    # section 16 before being made the default -- see RULES_DOC.md
    # section 17.
    "vr": [_feat_pair("I_probe_reset_near"), _feat_pair("I_vmin")],
    "etas": [_feat_pair("set_peak_width"), _feat_pair("set_peak_height")],
    "etar": [_feat_pair("reset_peak_width"), _feat_pair("reset_peak_height")],
    "gam": [_feat_pair("reset_wide_width"), _gam_vmin_rule],
    "ron": [_feat_pair("I_vmax"), _feat_pair("set_peak_height")],
    "roff": [_feat_pair("I_vmin"), _feat_pair("reset_peak_height")],
    "aon": [_ratio("I_probe_set_near", "I_probe_set_far"), _ratio("I_probe_set_far", "I_probe_set_far2")],
    # aoff kept on the ORIGINAL level-based rules (2026-10-12): a ratio
    # version mirroring aon's was tried and REGRESSED -- both rules
    # went from kept (polarity -1,-1) to dropped (same-sign both
    # directions). Reverted rather than keep a worse design just for
    # symmetry with aon.
    "aoff": [_feat_pair("I_vmin"), _feat_pair("I_probe_reset_near")],
    "ion": [_feat_pair("I_vmax"), _feat_pair("I_probe_set_far")],
    "ioff": [_feat_pair("I_vmin"), _feat_pair("I_probe_reset_far")],
}

# Derivative-augmented variant (by request, 2026-10-14): adds ONE new
# dI/dt-probe rule (mirroring the existing I-probe rules' own anchor+
# offset convention, see extract_features()'s own didt_probe_* fields)
# to every parameter that currently has only LEVEL (I-based) rules --
# vs, vr, ron, roff, aoff, ion, ioff -- so the rule set as a whole
# votes on whether the model's own RATE OF CHANGE matches the data's,
# not just its level, the same "fit I AND dI/dt" discipline
# fit_region_pair.py's own joint least_squares residual already uses.
# etas/etar/gam already have a derivative-based rule (peak width/
# height, reset_wide_width are all `didt`-derived); aon's ratio rules
# are left as is (see aoff's own comment above for why a ratio
# redesign there previously regressed). Validated against synthetic
# Monte Carlo ground truth (RULES_DOC.md section 16) BEFORE being
# promoted to the module's own default RULE_SPECS -- see that section
# for the actual comparison and which one ended up in use.
RULE_SPECS_WITH_DIDT = {
    "vs": RULE_SPECS["vs"] + [_feat_pair("didt_probe_set_near")],
    "vr": RULE_SPECS["vr"] + [_feat_pair("didt_probe_reset_near")],
    "etas": RULE_SPECS["etas"],
    "etar": RULE_SPECS["etar"],
    "gam": RULE_SPECS["gam"],
    "ron": RULE_SPECS["ron"] + [_feat_pair("didt_probe_set_near")],
    "roff": RULE_SPECS["roff"] + [_feat_pair("didt_probe_reset_near")],
    "aon": RULE_SPECS["aon"],
    "aoff": RULE_SPECS["aoff"] + [_feat_pair("didt_probe_reset_near")],
    "ion": RULE_SPECS["ion"] + [_feat_pair("didt_probe_set_far")],
    "ioff": RULE_SPECS["ioff"] + [_feat_pair("didt_probe_reset_far")],
}

# Selective variant (by request, 2026-10-14, after the full add-everywhere
# version above came out MIXED against synthetic Monte Carlo truth -- see
# RULES_DOC.md section 16): the didt-probe rule measurably helped `vs`
# (median relative error 180.7%->82.1%) but measurably HURT `roff`
# (88.6%->268.6%); everything else was unchanged (those rules got dropped
# as inconsistent during calibration, zero net effect either way). This
# keeps ONLY the `vs` addition that actually won, reverting everything
# else to the original RULE_SPECS.
RULE_SPECS_VS_DIDT = dict(RULE_SPECS)
RULE_SPECS_VS_DIDT["vs"] = RULE_SPECS["vs"] + [_feat_pair("didt_probe_set_near")]
# g0 has no own rule here -- within a chained window it's pinned via
# g0_override exactly like fit_region_pair.py; a free/restart fit's own
# g0 would need its own rule (not yet added -- see RULES_DOC.md's open
# questions) and is left FIXED for now (see fuzzy_fit()'s docstring).

RULE_POLARITY = {}  # filled by calibrate_polarities(): {(param, rule_idx): +-1}

# Per-parameter step-size multiplier (by request, 2026-10-12): vs/vr are
# the physically-meaningful threshold parameters the user cares about
# most getting right (the whole cycle-2173/8887/vr-is-bad investigation
# this fuzzy approach grew out of), so they get a bigger step each
# round than the rest -- NOT a vote-combination weight (each parameter
# still only sees its OWN rules; there's no cross-parameter voting to
# weight), just how aggressively that one parameter chases its own
# target relative to the others. Override via fuzzy_fit()'s own
# `param_weight` argument; unlisted parameters default to 1.0.
DEFAULT_PARAM_WEIGHT = {"vs": 2.0, "vr": 2.0}


def calibrate_polarities(t, V, truth=None, perturb_frac=0.35, verbose=True,
                          i_set_known=None, i_reset_known=None, rule_specs=None):
    """Empirically determines each RULE_SPECS rule's POLARITY against a
    synthetic ground truth (see the comment above RULE_SPECS for why
    this isn't hand-derived). For each parameter, perturbs it both
    above and below `truth` (log-space for dmm.LOG_PARAMS, linear
    otherwise, by `perturb_frac` of its own bounds span), holding every
    OTHER parameter at truth, and checks whether each rule votes
    CONSISTENTLY: -1 (decrease) for the too-high perturbation, +1
    (increase) for too-low. Consistent rules get their polarity cached
    in RULE_POLARITY; inconsistent ones (abstain, or vote the SAME way
    both directions) are left OUT entirely -- fuzzy_fit() then skips
    them rather than apply an unverified sign.

    i_set_known/i_reset_known: passed straight through to
    extract_features() -- strongly recommended (see that function's own
    docstring for the real bug this fixes: blind re-detection on a
    multi-region window can lock onto the WRONG "V>=0 & rising" stretch
    entirely).

    Model features are extracted with anchor_set/anchor_reset PINNED to
    the DATA's own i_set/i_reset -- so a perturbed parameter's effect is
    read off at a FIXED physical location, not confounded by the
    model's own apparent threshold ALSO moving.

    rule_specs: which RULE_SPECS-shaped dict to calibrate against
    (default: the module's own RULE_SPECS) -- pass RULE_SPECS_WITH_DIDT
    to calibrate the derivative-augmented variant instead (see RULES_DOC
    .md section 16).

    Returns {param: [bool per rule, True iff kept]} for inspection;
    mutates module-level RULE_POLARITY as a side effect."""
    rule_specs = RULE_SPECS if rule_specs is None else rule_specs
    truth = {**dmm.NOMINAL, **(truth or {})}
    with np.errstate(over="ignore"):
        I_data, _, _ = dmm.simulate(t, V, dmm.pack(truth))
    data_feat = extract_features(t, V, I_data, i_set_known=i_set_known, i_reset_known=i_reset_known)
    kept = {}
    for name, rules in rule_specs.items():
        kept[name] = []
        for ridx, rule in enumerate(rules):
            votes = {}
            for direction, factor in [("high", 1), ("low", -1)]:
                params = dict(truth)
                lo, hi = DEFAULT_BOUNDS[name]
                if name in dmm.LOG_PARAMS:
                    span = np.log10(hi) - np.log10(lo)
                    x = np.log10(truth[name]) + factor * perturb_frac * span
                    params[name] = float(10.0 ** np.clip(x, np.log10(lo), np.log10(hi)))
                else:
                    span = hi - lo
                    params[name] = float(np.clip(truth[name] + factor * perturb_frac * span, lo, hi))
                with np.errstate(over="ignore"):
                    I_model, _, _ = dmm.simulate(t, V, dmm.pack(params))
                model_feat = extract_features(t, V, I_model,
                                               anchor_set=data_feat["i_set"],
                                               anchor_reset=data_feat["i_reset"],
                                               i_set_known=i_set_known, i_reset_known=i_reset_known)
                dval, mval = rule(data_feat, model_feat)
                votes[direction] = _sign(dval, mval)
            consistent = votes["high"] != 0 and votes["low"] != 0 and votes["high"] == -votes["low"]
            kept[name].append(consistent)
            if consistent:
                RULE_POLARITY[(name, ridx)] = -votes["high"]
            elif (name, ridx) in RULE_POLARITY:
                del RULE_POLARITY[(name, ridx)]
            if verbose:
                tag = "kept   " if consistent else "DROPPED"
                pol = RULE_POLARITY.get((name, ridx))
                print(f"  {tag} {name}[{ridx}]: high={votes['high']:+d} low={votes['low']:+d}"
                      f"{f' -> polarity={pol:+d}' if consistent else ''}")
    return kept


# --------------------------------------------------------------- driver
def fuzzy_fit(t, V, I, x0_dict=None, bounds_dict=None, fixed=None, g0_override=None,
              n_rounds=150, initial_step=0.08, step_decay=0.95, smooth_n=10,
              i_set_known=None, i_reset_known=None, param_weight=None, rule_specs=None):
    """Iteratively refines all 12 (minus `fixed`) dmm_model parameters
    over one window via RULE_SPECS/RULE_POLARITY instead of
    least_squares -- see module docstring for the motivation and
    RULES_DOC.md for the full rationale table. calibrate_polarities()
    MUST be called at least once (on a representative window) before
    this is used for real -- RULE_POLARITY starts empty, in which case
    every parameter just abstains (stays at its x0 value) every round.

    Each round: simulate with the CURRENT params, extract model
    features (anchored to the DATA's own i_set/i_reset, same as
    calibration), evaluate every free parameter's own CALIBRATED rules
    (those with an entry in RULE_POLARITY; others are skipped) against
    the data features, combine votes by simple sum (ties/zero -> no
    change), and step that parameter by `step` in the voted direction
    -- log-space for dmm.LOG_PARAMS, linear otherwise (same convention
    fit_region_pair.py uses). `step` starts at `initial_step` (a
    FRACTION of the parameter's own bounds span in fit-space) and
    shrinks by `step_decay` each round (a cooling schedule, so early
    rounds can move fast and later rounds settle rather than oscillate
    forever). Updates are applied ONE PARAMETER AT A TIME within a
    round (Gauss-Seidel, not Jacobi) -- by request, "apply them
    sequentially for each parameter with a small increase" -- so a
    later parameter's own rules in the SAME round already see the
    effect of earlier ones.

    param_weight: {name: multiplier} applied to THIS parameter's own
    step size each round (default DEFAULT_PARAM_WEIGHT -- vs/vr get 2x,
    everything else 1x). This is NOT a vote-combination weight -- each
    parameter only ever sees its own rules; it's purely how far that
    one parameter moves per round relative to the others.

    rule_specs: which RULE_SPECS-shaped dict to use (default: the
    module's own RULE_SPECS) -- pass RULE_SPECS_WITH_DIDT to use the
    derivative-augmented variant instead (must match whichever one
    RULE_POLARITY was last calibrated against, see RULES_DOC.md
    section 16).

    Returns a dict: params_dict, rmse, didt_rmse, I_model, g_hist,
    va_hist, t, V, I, history (list of per-round dicts: round, params
    snapshot, rmse -- for convergence inspection/plotting)."""
    rule_specs = RULE_SPECS if rule_specs is None else rule_specs
    fixed = dict(fixed or {})
    if g0_override is not None:
        fixed["g0"] = g0_override
    x0_dict = {**dmm.NOMINAL, **(x0_dict or {})}
    bounds_dict = {**DEFAULT_BOUNDS, **(bounds_dict or {})}
    param_weight = {**DEFAULT_PARAM_WEIGHT, **(param_weight or {})}
    t, V, I = np.asarray(t), np.asarray(V), np.asarray(I)

    params = dict(x0_dict)
    free_names = [n for n in dmm.PARAM_NAMES if n not in fixed and n in rule_specs]
    data_feat = extract_features(t, V, I, smooth_n=smooth_n,
                                  i_set_known=i_set_known, i_reset_known=i_reset_known)
    anchor_set, anchor_reset = data_feat["i_set"], data_feat["i_reset"]

    def _simulate(p):
        arr = dmm.pack(p)
        with np.errstate(over="ignore"):
            return dmm.simulate(t, V, arr, g0_override=g0_override)

    history = []
    step = initial_step
    for rnd in range(n_rounds):
        for name in free_names:
            I_model, g_hist, va_hist = _simulate(params)
            model_feat = extract_features(t, V, I_model, smooth_n=smooth_n,
                                           anchor_set=anchor_set, anchor_reset=anchor_reset,
                                           i_set_known=anchor_set, i_reset_known=anchor_reset)
            net = 0
            for ridx, rule in enumerate(rule_specs[name]):
                polarity = RULE_POLARITY.get((name, ridx))
                if polarity is None:
                    continue  # uncalibrated/inconsistent rule -- skip, don't guess
                dval, mval = rule(data_feat, model_feat)
                net += polarity * _sign(dval, mval)
            direction = 0 if net == 0 else (1 if net > 0 else -1)
            if direction == 0:
                continue
            w_step = step * param_weight.get(name, 1.0)
            lo, hi = bounds_dict[name]
            if name in dmm.LOG_PARAMS:
                span = np.log10(hi) - np.log10(lo)
                x = np.log10(params[name])
                x = np.clip(x + direction * w_step * span, np.log10(lo) + 1e-12, np.log10(hi) - 1e-12)
                params[name] = float(10.0 ** x)
            else:
                span = hi - lo
                params[name] = float(np.clip(params[name] + direction * w_step * span, lo, hi))

        I_model, g_hist, va_hist = _simulate(params)
        rmse = float(np.sqrt(np.mean((I_model - I) ** 2)))
        history.append(dict(round=rnd, params=dict(params), rmse=rmse))
        step *= step_decay

    # Return the BEST-rmse round seen, not necessarily the last one (by
    # design, 2026-10-12): a Gauss-Seidel sweep of independently-voting
    # rules has no guarantee of monotonic improvement -- confirmed
    # directly on cycle 2000's own window, rmse dipped to a real low
    # around round 30 then a runaway parameter (etar hit its own upper
    # bound) pushed it back up over the remaining rounds. This is a
    # standard early-stopping/best-checkpoint safeguard, not a fix to
    # the rule polarities themselves -- see RULES_DOC.md's "iteration
    # dynamics" section for the still-open question of why some
    # trajectories diverge after an initially-good round.
    best = min(history, key=lambda h: h["rmse"]) if history else None
    params = best["params"] if best is not None else params

    I_model, g_hist, va_hist = _simulate(params)
    rmse = float(np.sqrt(np.mean((I_model - I) ** 2)))
    didt_model = moving_average(np.gradient(I_model, t), smooth_n)
    didt_real = moving_average(np.gradient(I, t), smooth_n)
    didt_rmse = float(np.sqrt(np.mean((didt_model - didt_real) ** 2)))

    return dict(params_dict=params, rmse=rmse, didt_rmse=didt_rmse,
                I_model=I_model, g_hist=g_hist, va_hist=va_hist,
                t=t, V=V, I=I, history=history, best_round=best["round"] if best else None)
