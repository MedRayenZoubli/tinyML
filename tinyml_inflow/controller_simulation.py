"""Offline closed-loop simulation of the INT8 forecast and valve controller.

The basin physics run at one-minute resolution. The INT8 model is queried every
five minutes using 24 five-minute samples, matching its training geometry.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from config import (
    ARTIFACT_DIR,
    DEFAULT_SEED,
    SEQ_LEN,
    STEP_MINUTES,
    STEP_SECONDS,
    TFLITE_MODEL_PATH,
    ensure_dirs,
    get_logger,
)
from generate_dataset import CatchmentParams, simulate_episode
from utils import load_scaler, make_interpreter

log = get_logger("controller")
CONTROLLER_METRICS_PATH = ARTIFACT_DIR / "controller_simulation.json"
PHYSICS_STEP_SECONDS = 60.0
SOURCE_STEP_MINUTES = STEP_MINUTES
MINUTES_PER_SOURCE_STEP = SOURCE_STEP_MINUTES
FORECAST_SCENARIOS = (0.80, 1.00, 1.25, 1.50)


class AnticipatoryValveController:
    """Choose the least release that protects against the worst forecast scenario."""

    def __init__(
        self,
        params: CatchmentParams,
        emergency_volume_percent: float = 70.0,
        preferred_volume_percent: float = 50.0,
        deadband_percent: float = 5.0,
    ) -> None:
        if not 0 < preferred_volume_percent < emergency_volume_percent <= 100:
            raise ValueError("Volume targets must satisfy 0 < preferred < emergency <= 100.")
        if not 0 <= deadband_percent <= 100:
            raise ValueError("deadband_percent must be between 0 and 100.")
        self.params = params
        self.emergency_volume = params.v_cap_m3 * emergency_volume_percent / 100.0
        self.preferred_volume = params.v_cap_m3 * preferred_volume_percent / 100.0
        self.deadband = deadband_percent / 100.0

    def _project_max_volume(
        self,
        current_volume: float,
        predicted_inflow: np.ndarray,
        valve_opening: float,
    ) -> float:
        volume = current_volume
        max_volume = volume
        for inflow in predicted_inflow:
            discharge = valve_opening * self.params.q_max_m3_s * np.sqrt(
                max(volume, 0.0) / self.params.v_cap_m3
            )
            volume = float(np.clip(
                volume + (float(inflow) - discharge) * PHYSICS_STEP_SECONDS,
                0.0,
                self.params.v_cap_m3,
            ))
            max_volume = max(max_volume, volume)
        return max_volume

    def choose_valve(
        self,
        current_volume: float,
        forecast_scenarios: list[np.ndarray],
        previous_valve: float,
    ) -> tuple[float, str]:
        """Select a safe command, applying a 5-point deadband unless urgent."""
        if current_volume >= self.emergency_volume:
            return 1.0, "emergency_override"

        candidates = np.linspace(0.0, 1.0, 21)
        for candidate in candidates:
            worst_volume = max(
                self._project_max_volume(current_volume, scenario, float(candidate))
                for scenario in forecast_scenarios
            )
            if worst_volume <= self.emergency_volume:
                reason = "forecast_safe" if current_volume < self.preferred_volume else "anticipatory_release"
                if abs(float(candidate) - previous_valve) < self.deadband and current_volume < self.emergency_volume * 0.95:
                    return previous_valve, "deadband_hold"
                return float(candidate), reason
        return 1.0, "maximum_release_required"


def _run_int8_forecast(interpreter, x_window: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_scale, input_zero_point = inp["quantization"]
    output_scale, output_zero_point = out["quantization"]
    quantized = np.clip(
        np.round(x_window / input_scale) + input_zero_point,
        -128,
        127,
    ).astype(np.int8)
    interpreter.set_tensor(inp["index"], quantized)
    interpreter.invoke()
    quantized_output = interpreter.get_tensor(out["index"])[0]
    return (quantized_output.astype(np.float32) - output_zero_point) * output_scale


def _scenario_forecasts(predicted_inflow: np.ndarray) -> list[np.ndarray]:
    """Expand five-minute predictions into one-minute physical scenarios."""
    base = np.repeat(predicted_inflow, MINUTES_PER_SOURCE_STEP)
    return [base * multiplier for multiplier in FORECAST_SCENARIOS]


def _warmup_scenarios(estimated_inflow: float, horizon_minutes: int = 120) -> list[np.ndarray]:
    base = np.full(horizon_minutes, max(estimated_inflow, 0.015), dtype=np.float32)
    return [base * multiplier for multiplier in FORECAST_SCENARIOS]


def _is_physically_infeasible(inflow: np.ndarray, params: CatchmentParams, start_volume: float) -> bool:
    """Check whether full opening still reaches 70% under the actual storm trace."""
    volume = start_volume
    for value in inflow:
        discharge = params.q_max_m3_s * np.sqrt(max(volume, 0.0) / params.v_cap_m3)
        volume = float(np.clip(volume + (float(value) - discharge) * PHYSICS_STEP_SECONDS,
                               0.0, params.v_cap_m3))
        if volume >= params.v_cap_m3 * 0.70:
            return True
    return False


def simulate_controlled_episode(
    rng: np.random.Generator,
    source_steps: int,
    params: CatchmentParams,
    scaler,
    interpreter,
    controller: AnticipatoryValveController,
) -> dict:
    """Run one minute-resolution controlled episode."""
    source = simulate_episode(rng, source_steps, params)
    inflow = np.repeat(source["inflow"], MINUTES_PER_SOURCE_STEP)
    source_features = np.repeat(source["features"], MINUTES_PER_SOURCE_STEP, axis=0)
    n_steps = len(inflow)
    volume = float(rng.uniform(0.05, 0.30) * params.v_cap_m3)
    start_volume = volume
    valve = 1.0
    control_reasons: dict[str, int] = {}
    valve_history = np.empty(n_steps, dtype=np.float32)
    volume_history = np.empty(n_steps, dtype=np.float32)
    discharge_history = np.empty(n_steps, dtype=np.float32)
    sampled_features: list[np.ndarray] = []
    forecast_errors: list[float] = []
    first_danger_minute: int | None = None
    first_response_minute: int | None = None
    recovery_minute: int | None = None
    previous_volume = volume
    previous_discharge = 0.0

    for minute in range(n_steps):
        discharge = valve * params.q_max_m3_s * np.sqrt(max(volume, 0.0) / params.v_cap_m3)
        volume = float(np.clip(
            volume + (float(inflow[minute]) - discharge) * PHYSICS_STEP_SECONDS,
            0.0,
            params.v_cap_m3,
        ))
        feature = np.array([source_features[minute, 0], volume, discharge, source_features[minute, 3]],
                           dtype=np.float32)
        valve_history[minute] = valve
        volume_history[minute] = volume
        discharge_history[minute] = discharge

        if first_danger_minute is None and volume >= controller.emergency_volume:
            first_danger_minute = minute
        if minute % MINUTES_PER_SOURCE_STEP == MINUTES_PER_SOURCE_STEP - 1:
            sampled_features.append(feature)
            if len(sampled_features) >= SEQ_LEN:
                window = np.asarray(sampled_features[-SEQ_LEN:], dtype=np.float32)
                normalized = scaler.normalize_features(window)[None, ...]
                predicted = scaler.denormalize_target(_run_int8_forecast(interpreter, normalized))
                actual = inflow[minute + 1:minute + 1 + len(predicted) * MINUTES_PER_SOURCE_STEP:MINUTES_PER_SOURCE_STEP]
                if len(actual) == len(predicted):
                    forecast_errors.append(float(np.mean(np.abs(predicted - actual))))
                scenarios = _scenario_forecasts(predicted)
            else:
                estimated = (volume - previous_volume) / PHYSICS_STEP_SECONDS + previous_discharge
                scenarios = _warmup_scenarios(float(estimated))
                control_reasons["warmup_scenario_control"] = control_reasons.get("warmup_scenario_control", 0) + 1

            previous_command = valve
            next_valve, reason = controller.choose_valve(volume, scenarios, valve)
            valve = next_valve
            if (
                first_response_minute is None
                and first_danger_minute is not None
                and next_valve > previous_command + controller.deadband
            ):
                first_response_minute = minute
            control_reasons[reason] = control_reasons.get(reason, 0) + 1
            previous_volume = volume
            previous_discharge = discharge

        if first_danger_minute is not None and recovery_minute is None:
            if minute > first_danger_minute and volume <= controller.preferred_volume:
                recovery_minute = minute - first_danger_minute

    infeasible = _is_physically_infeasible(inflow, params, start_volume)
    feasible_overflow = int(np.count_nonzero(volume_history >= params.v_cap_m3)) if not infeasible else 0
    return {
        "start_volume_percent": float(100 * start_volume / params.v_cap_m3),
        "max_volume_m3": float(volume_history.max()),
        "max_volume_percent": float(100 * volume_history.max() / params.v_cap_m3),
        "overflow_events": feasible_overflow,
        "infeasible_with_fixed_outlet": infeasible,
        "emergency_steps": int(np.count_nonzero(volume_history >= controller.emergency_volume)),
        "mean_discharge_m3_s": float(discharge_history.mean()),
        "max_discharge_m3_s": float(discharge_history.max()),
        "mean_valve_opening": float(valve_history.mean()),
        "mean_forecast_mae_m3_s": float(np.mean(forecast_errors)) if forecast_errors else None,
        "early_warning_minutes": (
            first_danger_minute - first_response_minute
            if first_danger_minute is not None and first_response_minute is not None
            else None
        ),
        "recovery_minutes": recovery_minute,
        "control_reasons": control_reasons,
    }


def run(episodes: int = 10, steps: int = 192, seed: int = DEFAULT_SEED) -> dict:
    """Run randomized offline regression scenarios and save controller metrics."""
    if episodes <= 0 or steps < SEQ_LEN:
        raise ValueError(f"episodes must be positive and steps must be at least {SEQ_LEN}.")
    ensure_dirs()
    params = CatchmentParams()
    scaler = load_scaler()
    interpreter = make_interpreter(model_path=TFLITE_MODEL_PATH)
    controller = AnticipatoryValveController(params)
    streams = np.random.SeedSequence(seed).spawn(episodes)
    details = [
        simulate_controlled_episode(np.random.default_rng(stream), steps, params, scaler, interpreter, controller)
        for stream in streams
    ]
    feasible = [item for item in details if not item["infeasible_with_fixed_outlet"]]
    summary = {
        "episodes": episodes,
        "source_steps_per_episode": steps,
        "physics_steps_per_episode": steps * MINUTES_PER_SOURCE_STEP,
        "seed": seed,
        "controller": {
            "objective": "zero overflow in feasible storms, then minimize discharge",
            "preferred_volume_percent": 50.0,
            "maximum_volume_percent": 70.0,
            "max_discharge_m3_s": params.q_max_m3_s,
            "physics_interval_minutes": 1,
            "model_interval_minutes": 5,
            "forecast_scenarios": list(FORECAST_SCENARIOS),
            "valve_deadband_percent": 5.0,
            "model": "full INT8 inflow forecast",
        },
        "totals": {
            "overflow_events_feasible": int(sum(item["overflow_events"] for item in feasible)),
            "infeasible_storms": int(sum(item["infeasible_with_fixed_outlet"] for item in details)),
            "feasible_storms": len(feasible),
            "max_volume_percent": float(max(item["max_volume_percent"] for item in details)),
            "mean_discharge_m3_s": float(np.mean([item["mean_discharge_m3_s"] for item in details])),
            "max_discharge_m3_s": float(max(item["max_discharge_m3_s"] for item in details)),
            "mean_forecast_mae_m3_s": float(np.mean([
                item["mean_forecast_mae_m3_s"] for item in details
                if item["mean_forecast_mae_m3_s"] is not None
            ])),
            "mean_recovery_minutes": float(np.mean([
                item["recovery_minutes"] for item in details
                if item["recovery_minutes"] is not None
            ])) if any(item["recovery_minutes"] is not None for item in details) else None,
        },
        "episodes_detail": details,
    }
    CONTROLLER_METRICS_PATH.write_text(json.dumps(summary, indent=2))
    log.info(
        "Completed %d episodes | feasible %d | infeasible %d | feasible overflow steps %d | max volume %.1f%%",
        episodes,
        summary["totals"]["feasible_storms"],
        summary["totals"]["infeasible_storms"],
        summary["totals"]["overflow_events_feasible"],
        summary["totals"]["max_volume_percent"],
    )
    log.info("Saved %s", CONTROLLER_METRICS_PATH)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=192, help="source steps; each expands to five physics minutes")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run(args.episodes, args.steps, args.seed)
