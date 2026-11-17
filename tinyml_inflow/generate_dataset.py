"""Synthetic stormwater hydrograph generator (pipeline step 1).

Physical model, 5-minute resolution; one "episode" is one continuous record:

1. Rainfall        1-3 Gaussian storm cells, each with a few narrower embedded bursts.
2. Soil/infiltration  Horton-style: infiltration capacity decays as the soil saturates,
                   so the same rain produces far more runoff on wet soil.
3. Runoff          impervious fraction (direct runoff) + pervious excess rainfall.
4. Routing         gamma-shaped unit hydrograph turns runoff into basin inflow I(t) [m3/s].
5. Basin           storage V(t) integrates inflow minus a valve-controlled orifice outflow Q(t).

Model inputs per step: [precip, basin volume, discharge, soil saturation] with sensor
noise. Target: the noise-free inflow of the 24 steps *after* the input window.
Splits are made per episode, so no storm leaks between train/val/test.

Outputs (data/): X_/y_{train,val,test}.npy, scaler.npz/.json, dataset_meta.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import (
    DATASET_META_JSON,
    DATA_DIR,
    DEFAULT_SEED,
    FEATURE_NAMES,
    HORIZON,
    N_FEATURES,
    SEQ_LEN,
    STEP_SECONDS,
    ensure_dirs,
    get_logger,
)
from utils import Scaler, save_scaler

log = get_logger("generate")


@dataclass(frozen=True)
class CatchmentParams:
    """Site constants of the synthetic urban catchment and detention basin."""

    area_m2: float = 1.2e5             # 12 ha catchment
    imperv_frac: float = 0.35          # share of directly connected impervious area
    imperv_runoff_coef: float = 0.90
    f0_mm_hr: float = 40.0             # Horton infiltration capacity, dry soil
    fc_mm_hr: float = 4.0              # Horton infiltration capacity, saturated soil
    horton_k: float = 4.0              # decay of capacity with saturation
    k_wet: float = 2.5                 # % saturation gained per mm infiltrated (at S=0)
    k_dry_per_hr: float = 0.03         # drainage / evapotranspiration, fraction of S per hour
    uh_shape: float = 3.0              # gamma unit-hydrograph shape
    uh_tp_steps: float = 6.0           # time to peak: 6 x 5 min = 30 min
    uh_len: int = 60                   # unit hydrograph memory, steps
    base_q_m3_s: float = 0.015         # dry-weather / baseflow
    v_cap_m3: float = 4000.0           # detention basin capacity
    q_max_m3_s: float = 0.6            # outlet capacity with fully open valve


# --------------------------------------------------------------------------- #
# Physical components
# --------------------------------------------------------------------------- #
def make_rainfall(rng: np.random.Generator, n_steps: int) -> np.ndarray:
    """Rain rate in mm/hr: Gaussian storm cells with embedded convective bursts."""
    t = np.arange(n_steps, dtype=np.float64)
    rain = np.zeros(n_steps)

    n_storms = rng.choice([0, 1, 2, 3], p=[0.05, 0.40, 0.40, 0.15])
    for _ in range(n_storms):
        mu = rng.uniform(0.10, 0.85) * n_steps        # storm centre (step index)
        sigma = rng.uniform(3.0, 14.0)                 # 15-70 min std-dev
        amp = min(rng.lognormal(np.log(25.0), 0.6), 120.0)   # peak mm/hr
        rain += amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        for _ in range(rng.integers(0, 3)):            # narrower embedded cells
            mu_c = mu + rng.normal(0.0, sigma)
            sigma_c = rng.uniform(1.5, 4.0)
            amp_c = amp * rng.uniform(0.2, 0.6)
            rain += amp_c * np.exp(-0.5 * ((t - mu_c) / sigma_c) ** 2)

    if rng.random() < 0.15:                            # light background drizzle
        rain += rng.uniform(0.3, 1.5)

    rain *= rng.lognormal(0.0, 0.10, n_steps)          # intensity turbulence
    rain[rain < 0.1] = 0.0                             # gauge resolution cut-off
    return np.minimum(rain, 150.0)


def gamma_unit_hydrograph(n_lag: int, shape: float, tp_steps: float) -> np.ndarray:
    """Discrete gamma unit hydrograph normalised to unit volume."""
    scale = tp_steps / (shape - 1.0)                   # gamma mode = (k-1) * theta
    tau = np.arange(n_lag) + 0.5
    uh = tau ** (shape - 1.0) * np.exp(-tau / scale)
    return uh / uh.sum()


def make_valve_profile(rng: np.random.Generator, n_steps: int) -> np.ndarray:
    """Piecewise-constant outlet valve opening (0.35-1.0) with 0-2 control changes."""
    cuts = np.sort(rng.integers(1, n_steps, size=rng.integers(0, 3)))
    levels = rng.uniform(0.35, 1.0, size=len(cuts) + 1)
    bounds = np.concatenate(([0], cuts, [n_steps]))
    valve = np.empty(n_steps)
    for k, level in enumerate(levels):
        valve[bounds[k]:bounds[k + 1]] = level
    return valve


def simulate_episode(rng: np.random.Generator, n_steps: int, p: CatchmentParams) -> dict[str, np.ndarray]:
    """Simulate one continuous storm record. Returns noisy features and clean inflow."""
    dt_hr = STEP_SECONDS / 3600.0
    rain = make_rainfall(rng, n_steps)

    # --- soil moisture + runoff generation (Horton infiltration decay) ------------
    fc = p.fc_mm_hr * rng.uniform(0.8, 1.2)
    s = rng.uniform(15.0, 70.0)                        # antecedent saturation, %
    soil = np.empty(n_steps)
    r_eff = np.empty(n_steps)                          # effective runoff intensity, mm/hr
    for t in range(n_steps):
        f_cap = fc + (p.f0_mm_hr - fc) * np.exp(-p.horton_k * s / 100.0)
        infiltration = min(rain[t], f_cap)
        excess = rain[t] - infiltration
        r_eff[t] = p.imperv_frac * p.imperv_runoff_coef * rain[t] + (1.0 - p.imperv_frac) * excess
        s += p.k_wet * infiltration * dt_hr * (1.0 - s / 100.0) - p.k_dry_per_hr * s * dt_hr
        s = float(np.clip(s, 0.0, 100.0))
        soil[t] = s                                    # saturation at the END of step t

    # --- routing: runoff -> basin inflow via unit hydrograph ---------------------
    uh = gamma_unit_hydrograph(p.uh_len, p.uh_shape, p.uh_tp_steps * rng.uniform(0.85, 1.15))
    q_eff = r_eff / 1000.0 / 3600.0 * p.area_m2        # m3/s
    baseflow = p.base_q_m3_s * (0.5 + soil / 100.0)
    inflow = np.convolve(q_eff, uh)[:n_steps] + baseflow

    # --- detention basin mass balance --------------------------------------------
    valve = make_valve_profile(rng, n_steps)
    volume = np.empty(n_steps)
    discharge = np.empty(n_steps)
    v = rng.uniform(0.02, 0.25) * p.v_cap_m3
    for t in range(n_steps):
        discharge[t] = valve[t] * p.q_max_m3_s * np.sqrt(v / p.v_cap_m3)   # orifice-like
        v = float(np.clip(v + (inflow[t] - discharge[t]) * STEP_SECONDS, 0.0, p.v_cap_m3))
        volume[t] = v                                  # volume at the END of step t

    # --- sensor noise (applied to inputs only, never to the target) ----------------
    n = n_steps
    rain_obs = np.clip(rain * rng.normal(1.0, 0.03, n) + rng.normal(0.0, 0.05, n), 0.0, None)
    vol_obs = np.clip(volume + rng.normal(0.0, 2.0, n), 0.0, p.v_cap_m3)
    q_obs = np.clip(discharge * rng.normal(1.0, 0.02, n) + rng.normal(0.0, 0.002, n), 0.0, None)
    soil_obs = np.clip(soil + rng.normal(0.0, 0.5, n), 0.0, 100.0)

    features = np.stack([rain_obs, vol_obs, q_obs, soil_obs], axis=1)   # (n, 4)
    return {"features": features, "inflow": inflow}


# --------------------------------------------------------------------------- #
# Windowing, splitting, scaling
# --------------------------------------------------------------------------- #
def make_windows(features: np.ndarray, inflow: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice one episode into (X: [N, 24, 4], y: [N, 24]) samples.

    X[i] covers steps s..s+23; y[i] is the inflow of steps s+24..s+47, i.e. the
    2 hours that follow the input window.
    """
    n = len(inflow) - (SEQ_LEN + HORIZON) + 1
    if n <= 0:
        raise ValueError("Episode too short for one input window plus forecast horizon.")
    fw = sliding_window_view(features, SEQ_LEN, axis=0)      # (T-23, 4, 24)
    iw = sliding_window_view(inflow, HORIZON, axis=0)        # (T-23, 24)
    x = fw[:n].transpose(0, 2, 1)[::stride]
    y = iw[SEQ_LEN:SEQ_LEN + n][::stride]
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)


def run(n_episodes: int = 600, n_steps: int = 192, stride: int = 2, seed: int = DEFAULT_SEED) -> dict:
    """Generate, split, normalise and save the dataset. Returns a summary dict."""
    ensure_dirs()
    params = CatchmentParams()
    log.info("Simulating %d episodes x %d steps (%.1f h each), window stride %d, seed %d",
             n_episodes, n_steps, n_steps * STEP_SECONDS / 3600, stride, seed)

    # One independent RNG stream per episode -> reproducible regardless of count.
    streams = np.random.SeedSequence(seed).spawn(n_episodes)
    n_train = int(round(0.70 * n_episodes))
    n_val = int(round(0.15 * n_episodes))
    split_of = ["train"] * n_train + ["val"] * n_val + ["test"] * (n_episodes - n_train - n_val)

    raw: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"train": [], "val": [], "test": []}
    peaks = []
    for i, (ss, split) in enumerate(zip(streams, split_of)):
        ep = simulate_episode(np.random.default_rng(ss), n_steps, params)
        raw[split].append(make_windows(ep["features"], ep["inflow"], stride))
        peaks.append(float(ep["inflow"].max()))
        if (i + 1) % 100 == 0 or i + 1 == n_episodes:
            log.info("  simulated %d/%d episodes", i + 1, n_episodes)

    data = {s: (np.concatenate([w[0] for w in wins]), np.concatenate([w[1] for w in wins]))
            for s, wins in raw.items()}

    # Fit scaling on the training split only, then apply to all splits.
    x_train, y_train = data["train"]
    scaler = Scaler(
        feature_min=x_train.reshape(-1, N_FEATURES).min(axis=0),
        feature_max=x_train.reshape(-1, N_FEATURES).max(axis=0),
        target_min=float(y_train.min()),
        target_max=float(y_train.max()),
    )
    save_scaler(scaler)

    meta: dict = {"seed": seed, "episodes": n_episodes, "steps_per_episode": n_steps, "stride": stride,
                  "catchment": params.__dict__, "splits": {}}
    for split, (x, y) in data.items():
        xn, yn = scaler.normalize_features(x), scaler.normalize_target(y)
        np.save(DATA_DIR / f"X_{split}.npy", xn)
        np.save(DATA_DIR / f"y_{split}.npy", yn)
        storm_share = float((y.max(axis=1) > 0.1).mean())
        meta["splits"][split] = {"samples": int(len(x)), "x_shape": list(xn.shape),
                                 "y_shape": list(yn.shape), "share_with_storm_inflow": storm_share}
        log.info("  %-5s X%s y%s | %.0f%% of windows forecast inflow > 0.1 m3/s",
                 split, xn.shape, yn.shape, 100 * storm_share)

    meta["scaler"] = {"feature_names": list(FEATURE_NAMES),
                      "feature_min": [float(v) for v in scaler.feature_min],
                      "feature_max": [float(v) for v in scaler.feature_max],
                      "target_min": scaler.target_min, "target_max": scaler.target_max}
    DATASET_META_JSON.write_text(json.dumps(meta, indent=2))

    q = np.percentile(peaks, [50, 90, 99])
    log.info("Episode peak inflow m3/s: median %.2f | p90 %.2f | p99 %.2f | max %.2f", *q, max(peaks))
    log.info("Feature ranges (train): %s", {n: (round(float(a), 2), round(float(b), 2))
             for n, a, b in zip(FEATURE_NAMES, scaler.feature_min, scaler.feature_max)})
    log.info("Dataset + scaler saved to %s", DATA_DIR)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--steps", type=int, default=192, help="5-minute steps per episode")
    ap.add_argument("--stride", type=int, default=2, help="window stride in steps")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    run(a.episodes, a.steps, a.stride, a.seed)
