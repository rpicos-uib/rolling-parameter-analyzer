"""Shared, Tk-independent data-loading/smoothing utilities and the default
parameter-bounds table for memristor_model -- used by gui_view.py (the
interactive viewer) AND by headless fitting/analysis scripts (e.g.
fit_region_pair.py), so everything reads the real data and bounds the same
twelve dmm_model parameters the same way. Split out of gui_view.py (which
used to define all of this itself) specifically so a fitting script can
import these without pulling in gui_view's tkinter/matplotlib dependencies.
"""
import h5py
import numpy as np

DATA_PATH = "dades/dades.mat"

# Physically-plausible range for each of dmm_model's 12 parameters. Used
# both as gui_view.py's slider range (interactive exploration) and as
# fit_region_pair.py's optimizer bounds (hard least_squares constraint) --
# one table, two uses.
DEFAULT_BOUNDS = dict(
    ron=(1.0, 500.0), roff=(1.0, 500.0),
    aon=(0.5, 10.0), aoff=(0.5, 10.0),
    ion=(1e-4, 100.0), ioff=(1e-7, 1.0),
    vs=(0.05, 1.5), vr=(-1.5, -0.05),
    etas=(0.5, 3000.0), etar=(0.5, 3000.0),
    gam=(0.01, 3.0), g0=(1e-4, 1.0),
)


def load_data(path=DATA_PATH):
    with h5py.File(path, "r") as f:
        curr = f["curr"][0][:]
        cycle = f["cycle"][0][:]
        time_ = f["time"][0][:]
        volt = f["volt"][0][:]
    return curr, np.abs(cycle).astype(int), time_, volt


def cycle_bounds(logical_cycle):
    """{cycle_number: (i0, i1)} -- i0/i1 GLOBAL sample indices (i1
    exclusive), one entry per whole logical cycle."""
    order = np.argsort(logical_cycle, kind="stable")
    sorted_cycle = logical_cycle[order]
    unique, first_pos = np.unique(sorted_cycle, return_index=True)
    last_pos = np.concatenate([first_pos[1:], [len(sorted_cycle)]])
    bounds = {}
    for N, fp, lp in zip(unique, first_pos, last_pos):
        idx = order[fp:lp]
        bounds[int(N)] = (int(idx.min()), int(idx.max()) + 1)
    return bounds


def moving_average(x, n):
    """Centered moving average, window length n, edge-padded (replicates
    the boundary value rather than zero-padding, so it doesn't pull the
    ends toward zero) -- output is the same length as x. n<=1 returns x
    unchanged."""
    n = int(n)
    if n <= 1:
        return np.asarray(x, dtype=float).copy()
    pad_lo = n // 2
    pad_hi = n - 1 - pad_lo
    xp = np.pad(np.asarray(x, dtype=float), (pad_lo, pad_hi), mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="valid")


def _fill_nonfinite(x):
    """Linearly interpolates over any non-finite (nan/inf) entries -- so a
    single bad point (e.g. from a V-repeat divide-by-zero in dI/dV) doesn't
    spread nan across a whole moving_average window around it."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.any() and not bad.all():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    elif bad.all():
        x[:] = 0.0
    return x


def smooth_on_uniform_grid(x, y, smooth_n):
    """moving_average(y, smooth_n), but computed on a UNIFORM grid in x
    (resampled via interpolation) instead of over raw sample COUNT --
    by request (2026-10-12, cycle 3039 investigation): real acquisition
    data's own sample TIMESTEP (or, in principle, voltage step) can vary
    several-fold within one window (confirmed: concentrated right at the
    SET->RESET transition onset here, almost certainly instrument dwell/
    settling time right after the device switches), so a fixed-N-sample
    average spans a very different real x-window depending on local
    sample density -- smoothing more heavily exactly where the
    acquisition slowed down, often right where a transition lives.

    x is NOT assumed globally monotonic (V oscillates within one cycle;
    t always increases) -- resampling across a non-monotonic run would
    interpolate between UNRELATED samples that happen to share a similar
    x, so this is done per maximal MONOTONIC run of x instead."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    out = y.copy()
    if n < 3:
        return out
    sign = np.sign(np.diff(x))
    breaks = np.flatnonzero((sign[1:] != sign[:-1]) & (sign[1:] != 0) & (sign[:-1] != 0)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    for s, e in zip(starts, ends):
        xs, ys = x[s:e], y[s:e]
        if e - s < 3:
            continue
        order = np.argsort(xs)
        xs_o, ys_o = xs[order], ys[order]
        xs_u, idx_u = np.unique(xs_o, return_index=True)
        if len(xs_u) < 3:
            out[s:e] = moving_average(ys, smooth_n)
            continue
        ys_u = ys_o[idx_u]
        x_grid = np.linspace(xs_u[0], xs_u[-1], max(len(xs_u), 50))
        y_grid_sm = moving_average(np.interp(x_grid, xs_u, ys_u), smooth_n)
        out[s:e] = np.interp(xs, x_grid, y_grid_sm)
    return out
