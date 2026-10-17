"""Deterministic DMM (Dynamic Memdiode Model) for HfO2-based memristive
devices, from "Modeling and Simulation of Hysteron Dynamics in HfO2-Based
Memristive Devices" (see the PDF in this directory).

This file implements ONLY the model itself: the fast I(V,g) relation, the
slow g(t) state dynamics, and the per-sample forward simulator. No data
loading, no fitting, no variability -- those are separate concerns, built
on top of this by calling it.

Fast (instantaneous) dynamics -- solved for VA each sample:
    K(a, b, g)  = (a - b)*g + b                        endpoint interpolation
    I(VA, g)    = K(ion, ioff, g) * sinh(K(aon, aoff, g) * VA)
    V           = VA + K(ron, roff, g) * I(VA, g)       series resistance drop

Slow (state) dynamics -- integrated over time:
    dg/dt = (1-g) * S(VA) * H(VA) * H(dV/dt)  -  g * R(VA) * H(-VA)
    S(V)  = 2 * exp(-etaS*vS)  * sinh( etaS * V)                     SET rate
    R(V)  = 2 * exp( etaR*vR)  * sinh(-etaR * g**gamma * V)          RESET rate

H(.) is the Heaviside step function: SET kinetics act only while VA>=0
AND V is rising; RESET kinetics act only while VA<0. g in [0,1] is the
device's internal state (0: fully OFF/HRS, 1: fully ON/LRS).

Twelve parameters, in this fixed order throughout:
    ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam, g0

Numba-jitted (2026-10-11, by request): simulate()'s own per-sample loop
(ktrans/stable_2exp_sinh/solve_va_newton/current/step_g, and the loop
itself via the private _simulate_loop) is inherently sequential (g[k]
depends on g[k-1], so it can never vectorize away in numpy) and gets
called once per least_squares residual evaluation during fitting --
i.e. up to thousands of times per window. Compiling it to machine code
measured ~64x faster per call (0.071ms vs 4.5ms for a 500-sample call,
warm) and bit-for-bit equivalent results (confirmed directly against
the pre-Numba implementation: differences at float64 machine-epsilon
level only). simulate() itself stays plain Python (handles the
Optional g0_override and arbitrary `params` sequence type, both
awkward to type for Numba) and just delegates to the jitted core --
every other file's calls to dmm.simulate/.solve_va_newton/etc. are
unaffected, same signatures, same behavior."""
import math

import numba
import numpy as np

PARAM_NAMES = ["ron", "roff", "aon", "aoff", "ion", "ioff",
               "vs", "vr", "etas", "etar", "gam", "g0"]

# Positive, multi-decade parameters are naturally handled in log10 space by
# any fitting/variability code built on top of this model; vs, vr, g0 are
# signed or already O(1) and stay linear. Not used by this file itself --
# kept here as the one place that should define it, for everything else to
# import rather than redefine.
LOG_PARAMS = ["ron", "roff", "aon", "aoff", "ion", "ioff", "etas", "etar", "gam"]
LINEAR_PARAMS = ["vs", "vr", "g0"]

# A reference parameter combination, self-consistent by construction (the
# paper's own Fig. 5 modelcard) -- a reasonable default/starting point,
# NOT a fitted value for any particular real device.
NOMINAL = dict(ron=52.0, roff=52.0, aon=4.1, aoff=4.1, ion=12.0, ioff=2.1e-4,
               vs=0.57, vr=-0.067, etas=50.0, etar=150.0, gam=0.220, g0=1e-3)


@numba.njit(cache=True)
def ktrans(a, b, g):
    """K(a,b,g) = (a-b)*g + b: linear interpolation between the g=0 (b)
    and g=1 (a) endpoints -- the model's one shared mixing rule for
    ron/roff, aon/aoff, and ion/ioff alike."""
    return (a - b) * g + b


@numba.njit(cache=True)
def stable_2exp_sinh(a, b):
    """2*exp(a)*sinh(b), evaluated as exp(a+b) - exp(a-b) (the combined
    exponents directly) rather than exp(a) and sinh(b) separately.

    Needed for S(V)/R(V): a parameter far in a heavy-tailed distribution's
    excursion can make exp(a) underflow to exactly 0.0 while sinh(b)
    overflows to exactly inf in the same call, and 0.0*inf = NaN in
    IEEE754. Clipping the combined exponents to +-700 (safely inside
    float64's ~+-745 overflow range) avoids that without changing the
    result for any well-behaved (non-overflowing) input. (Plain
    if/elif clipping, not np.clip, for Numba scalar compatibility --
    numerically identical.)"""
    p = a + b
    if p > 700.0:
        p = 700.0
    elif p < -700.0:
        p = -700.0
    m = a - b
    if m > 700.0:
        m = 700.0
    elif m < -700.0:
        m = -700.0
    return math.exp(p) - math.exp(m)


@numba.njit(cache=True)
def solve_va_newton(V, g, ron, roff, aon, aoff, ion, ioff, va_guess,
                     max_iter=30, tol=1e-12):
    """Solves V = VA + Rs(g)*I(VA,g) for VA via Newton's method.

    F(VA) = VA + Rs*I(VA,g) - V is strictly increasing (F' = 1 +
    Rs*Ki*Ka*cosh(Ka*VA) > 0 always, since Rs, Ki, Ka > 0 by
    construction for any physically sensible parameter set), so Newton's
    method from any reasonable warm start converges in a handful of
    iterations -- a fresh cold start (va_guess=0) also works, just needs
    a few more iterations than warm-starting from the previous sample."""
    Ka = ktrans(aon, aoff, g)
    Ki = ktrans(ion, ioff, g)
    Rs = ktrans(ron, roff, g)
    va = va_guess
    for _ in range(max_iter):
        sh = math.sinh(Ka * va)
        ch = math.cosh(Ka * va)
        F = va + Rs * Ki * sh - V
        dF = 1.0 + Rs * Ki * Ka * ch
        step = F / dF
        va -= step
        if abs(step) < tol:
            break
    return va


@numba.njit(cache=True)
def current(va, g, aon, aoff, ion, ioff):
    """I(VA, g) = K(ion,ioff,g) * sinh(K(aon,aoff,g) * VA), given VA
    already solved (solve_va_newton) for this (V, g)."""
    Ka = ktrans(aon, aoff, g)
    Ki = ktrans(ion, ioff, g)
    return Ki * math.sinh(Ka * va)


@numba.njit(cache=True)
def step_g(g, va, dt, v_rising, ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam):
    """Advances g by one timestep dt, given the state at the START of the
    step (va, v_rising: whether V is increasing this step) via the exact
    exponential update of the (locally constant-rate) ODE:
        SET  (va>=0, v_rising): g_new = 1 - (1-g)*exp(-Seff*dt)
        RESET (va<0):           g_new = g*exp(-Reff*dt)
        otherwise (va>=0, V falling): g unchanged -- H(dV/dt) gates SET off.
    (The original's `with np.errstate(over="ignore")` is dropped here --
    not supported in Numba nopython mode, and unneeded: compiled float
    overflow just silently produces inf, no Python warning machinery
    involved either way.)"""
    if va >= 0.0:
        if v_rising:
            Seff = stable_2exp_sinh(-etas * vs, etas * va)
            g = 1.0 - (1.0 - g) * math.exp(-max(Seff, 0.0) * dt)
    else:
        Reff = stable_2exp_sinh(etar * vr, -etar * (max(g, 0.0) ** gam) * va)
        g = g * math.exp(-max(Reff, 0.0) * dt)
    return min(max(g, 0.0), 1.0)


@numba.njit(cache=True)
def dg_dt_rate(g, va, v_rising, ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam):
    """The RAW instantaneous dg/dt (the model's own ODE rate: dg/dt =
    (1-g)*S(VA)*H(VA)*H(dV/dt) - g*R(VA)*H(-VA)) -- NOT the exact-
    exponential time-integrated update step_g() performs. Needed by
    analytic_didt() below for its own chain-rule derivative; ron/roff/
    aon/aoff/ion/ioff are accepted only for a uniform call signature
    across this file's functions and are unused here."""
    if va >= 0.0:
        if v_rising:
            Seff = stable_2exp_sinh(-etas * vs, etas * va)
            return (1.0 - g) * max(Seff, 0.0)
        return 0.0
    Reff = stable_2exp_sinh(etar * vr, -etar * (max(g, 0.0) ** gam) * va)
    return -g * max(Reff, 0.0)


def analytic_didt(t, V, g_hist, va_hist, params, dvdt=None):
    """Exact, CLOSED-FORM dI/dt via the chain rule through I(VA,g) and
    V = VA + Rs(g)*I(VA,g), GIVEN dV/dt as an input -- dV/dt is treated as
    independently calculable/measurable (from real, however-smoothed
    measured data, or any other source) rather than derived internally
    from V,t the same way every time. Defaults to np.gradient(V, t) if
    `dvdt` isn't supplied.

    Requires g_hist/va_hist from a prior simulate() call over the SAME
    (t, V, params) -- this function does not itself integrate g; it only
    differentiates I=I(VA,g) through the model's own algebraic and ODE
    relations, given the state trajectory already computed.

    Derivation, at each sample (Ki=K(ion,ioff,g), Ka=K(aon,aoff,g),
    Rs=K(ron,roff,g)):
        I = Ki*sinh(Ka*VA)
        dI/dVA|_g = Ki*Ka*cosh(Ka*VA)
        dI/dg|_VA = (ion-ioff)*sinh(Ka*VA) + Ki*VA*cosh(Ka*VA)*(aon-aoff)
        V = VA + Rs*I(VA,g)
          => dV/dt = dVA/dt*(1 + Rs*dI/dVA) + dg/dt*((ron-roff)*I + Rs*dI/dg)
          => dVA/dt = (dV/dt - dg/dt*((ron-roff)*I + Rs*dI/dg)) / (1 + Rs*dI/dVA)
        dI/dt = dI/dVA * dVA/dt + dI/dg * dg/dt
    dg/dt itself is the model's own instantaneous rate (dg_dt_rate), not a
    finite difference of g_hist.

    Returns dI/dt, same length as t."""
    if dvdt is None:
        dvdt = np.gradient(V, t)
    ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam, g0 = params
    n = len(t)
    didt = np.empty(n)
    for k in range(n):
        g, va = g_hist[k], va_hist[k]
        Ki = ktrans(ion, ioff, g)
        Ka = ktrans(aon, aoff, g)
        Rs = ktrans(ron, roff, g)
        sh, ch = np.sinh(Ka * va), np.cosh(Ka * va)
        dI_dVA = Ki * Ka * ch
        dI_dg = (ion - ioff) * sh + Ki * va * ch * (aon - aoff)
        I_here = Ki * sh

        v_rising = (V[k + 1] >= V[k]) if k < n - 1 else (V[k] >= V[k - 1])
        dg_dt = dg_dt_rate(g, va, v_rising, ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam)

        denom = 1.0 + Rs * dI_dVA
        dva_dt = (dvdt[k] - dg_dt * ((ron - roff) * I_here + Rs * dI_dg)) / denom
        didt[k] = dI_dVA * dva_dt + dI_dg * dg_dt
    return didt


def simplified_conductance(g, ron, roff, aon, aoff, ion, ioff):
    """Small-signal dI/dV near V=0 (VA~=0), for CONSTANT g -- the two
    simplifying assumptions analytic_didt's general formula collapses
    under when (a) deep in a SET/RESET plateau, far from any transition,
    so dg/dt=0, and (b) V~=0, so sinh(Ka*VA)~=Ka*VA (I~=Ki*Ka*VA) and
    cosh(Ka*VA)~=1 (dI/dVA~=Ki*Ka, constant). With dg/dt=0, analytic_didt's
    own dVA/dt reduces to dV/dt/(1+Rs*dI/dVA), so:
        dI/dt ~= G * dV/dt,   G = Ki*Ka / (1 + Rs*Ki*Ka)
    the same combined series/device conductance derived earlier for the
    ion/ioff/ron/roff extraction (1/G = Rs + 1/(Ki*Ka)). Which g to pass in
    depends on WHERE near V=0 you are, not which region you're heading
    into: right at the V~=0 crossing, g still reflects the PREVIOUS
    phase's ending state (the real SET/RESET transition, gated by vs/vr,
    hasn't happened yet at such small V) -- so a "set" region (V>0, just
    after a reset left g~=0) uses g=0 (roff,aoff,ioff), and a "reset"
    region (V<0, just after a set left g~=1) uses g=1 (ron,aon,ion)."""
    Ki = ktrans(ion, ioff, g)
    Ka = ktrans(aon, aoff, g)
    Rs = ktrans(ron, roff, g)
    return Ki * Ka / (1.0 + Rs * Ki * Ka)


@numba.njit(cache=True)
def _simulate_loop(t, V, ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam, g_start, va0):
    """The actual per-sample loop simulate() below just unpacks `params`
    and delegates to -- split out so every argument is a plain scalar
    float (no Optional-typed g0_override, no runtime-length `params`
    sequence to unpack), which is what Numba's nopython mode wants.
    Inherently sequential (g[k] depends on g[k-1]), so this never
    vectorizes away in numpy either way -- compiling the scalar loop
    itself (rather than paying full Python/numpy-dispatch overhead per
    sample, EVERY residual evaluation during fitting) is where the speed
    comes from. See simulate()'s own docstring for the public contract."""
    n = len(t)
    I = np.empty(n)
    g_hist = np.empty(n)
    va_hist = np.empty(n)
    g = g_start
    va = va0
    for k in range(n):
        va = solve_va_newton(V[k], g, ron, roff, aon, aoff, ion, ioff, va)
        I[k] = current(va, g, aon, aoff, ion, ioff)
        g_hist[k] = g
        va_hist[k] = va

        if k == n - 1:
            break
        dt = t[k + 1] - t[k]
        if dt <= 0:
            continue
        v_rising = V[k + 1] >= V[k]
        g = step_g(g, va, dt, v_rising, ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam)

    return I, g_hist, va_hist


def simulate(t, V, params, g0_override=None, va0=0.0):
    """Forward-simulates the full DMM over a real (possibly non-uniformly
    spaced) excitation (t, V). Returns (I, g_hist, va_hist), all length len(t).

    params: length-12 sequence, PARAM_NAMES order (or use `unpack`/`pack`
    below to go from/to a {name: value} dict).
    g0_override: starting g, overriding params' own g0 -- lets g be
    carried continuously across multiple concatenated calls (e.g. across
    real measurement cycles) instead of reset to g0 every time.
    va0: starting guess for VA's own Newton solve (warm start).

    Thin wrapper (resolves g0_override/dtype here, in plain Python, since
    both are awkward to type for Numba) around the Numba-compiled
    _simulate_loop() -- same signature/behavior as before this was
    jitted, a drop-in replacement for every existing caller."""
    ron, roff, aon, aoff, ion, ioff, vs, vr, etas, etar, gam, g0 = params
    t = np.ascontiguousarray(t, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)
    g_start = float(g0_override) if g0_override is not None else float(g0)
    return _simulate_loop(t, V, float(ron), float(roff), float(aon), float(aoff),
                           float(ion), float(ioff), float(vs), float(vr),
                           float(etas), float(etar), float(gam), g_start, float(va0))


def pack(params_dict):
    """{name: value} -> length-12 array, PARAM_NAMES order."""
    return np.array([params_dict[name] for name in PARAM_NAMES])


def unpack(x):
    """length-12 array -> {name: value}, PARAM_NAMES order."""
    return dict(zip(PARAM_NAMES, x))
