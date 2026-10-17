"""Generates a small SYNTHETIC demo dataset in the same HDF5 ("MATLAB v7.3
.mat") layout data_utils.load_data() expects (curr/cycle/time/volt
datasets, each shape (1, N)), purely so the fitting CLI, the Monte Carlo
generator, and the comparison script in this repository can be run
end-to-end by anyone who clones it -- with NO dependency on the project's
own real, unpublished measurement data.

Each cycle sweeps 0 -> Vmax -> 0 -> Vmin -> 0 with the same SET/RESET
sample counts (219 / 280) used throughout the paper. Parameters drift
cycle to cycle via a small random walk (seeded, reproducible) around the
DMM's own NOMINAL card, so the chain fit has something non-trivial to
recover -- this is a DEMO of the pipeline's mechanics, not a stand-in for
the paper's own real-device results."""
import os

import h5py
import numpy as np

import dmm_model as dmm

OUT_PATH = "demo_data/dades_demo.mat"
N_CYCLES = 80
SET_N, RESET_N = 219, 280
SET_DURATION, RESET_DURATION = 5.7, 7.7
V_MAX, V_MIN = 1.1, -1.4
SEED = 0


def half_cycle_time_voltage(n, duration, v_peak):
    t = np.linspace(0.0, duration, n)
    half = n // 2
    V = np.empty(n)
    V[:half] = np.linspace(0.0, v_peak, half, endpoint=False)
    V[half:] = np.linspace(v_peak, 0.0, n - half)
    return t, V


def main():
    rng = np.random.default_rng(SEED)
    t_set, V_set = half_cycle_time_voltage(SET_N, SET_DURATION, V_MAX)
    t_reset, V_reset = half_cycle_time_voltage(RESET_N, RESET_DURATION, V_MIN)

    params = dict(dmm.NOMINAL)
    g = params["g0"]
    curr_all, time_all, volt_all, cycle_all = [], [], [], []
    t_cursor = 0.0

    for cyc in range(1, N_CYCLES + 1):
        # small reproducible per-cycle drift on the parameters the paper's
        # own chain fit actually frees per half-cycle (Table 1)
        for name, sigma in [("vs", 0.02), ("ron", 0.08), ("etas", 0.05)]:
            params[name] = float(params[name] * np.exp(rng.normal(0, sigma))
                                  if name != "vs" else params[name] + rng.normal(0, sigma))
        for name, sigma in [("vr", 0.02), ("roff", 0.08), ("etar", 0.05)]:
            params[name] = float(params[name] * np.exp(rng.normal(0, sigma))
                                  if name != "vr" else params[name] + rng.normal(0, sigma))
        params["vs"] = float(np.clip(params["vs"], 0.1, 1.2))
        params["vr"] = float(np.clip(params["vr"], -1.2, -0.1))

        t_abs_set = t_cursor + t_set
        I_set, g_hist_s, _ = dmm.simulate(t_abs_set, V_set, dmm.pack(params), g0_override=g)
        g = float(g_hist_s[-1])
        t_cursor = t_abs_set[-1]

        t_abs_reset = t_cursor + t_reset
        I_reset, g_hist_r, _ = dmm.simulate(t_abs_reset, V_reset, dmm.pack(params), g0_override=g)
        g = float(g_hist_r[-1])
        t_cursor = t_abs_reset[-1]

        curr_all.append(I_set); curr_all.append(I_reset)
        volt_all.append(V_set); volt_all.append(V_reset)
        time_all.append(t_abs_set); time_all.append(t_abs_reset)
        cycle_all.append(np.full(SET_N, cyc)); cycle_all.append(np.full(RESET_N, cyc))

    curr = np.concatenate(curr_all)
    volt = np.concatenate(volt_all)
    time_ = np.concatenate(time_all)
    cycle = np.concatenate(cycle_all).astype(np.float64)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with h5py.File(OUT_PATH, "w") as f:
        f.create_dataset("curr", data=curr[None, :])
        f.create_dataset("cycle", data=cycle[None, :])
        f.create_dataset("time", data=time_[None, :])
        f.create_dataset("volt", data=volt[None, :])

    print(f"wrote {OUT_PATH}: {N_CYCLES} cycles, {len(curr)} samples total")


if __name__ == "__main__":
    main()
