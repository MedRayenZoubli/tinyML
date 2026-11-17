"""Master entry point: dataset generation -> training -> INT8 export -> local test.

    python main.py                       # full pipeline, writes artifacts/model.h
    python main.py --epochs 40           # quicker run
    python main.py --stages test         # re-run only selected stages
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time

from config import ARTIFACT_DIR, DEFAULT_SEED, METRICS_JSON, MODEL_HEADER_PATH, ensure_dirs, get_logger

STAGES = ("generate", "train", "quantize", "test", "control")
log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TinyML inflow-forecast pipeline for ESP32 (TFLite Micro, INT8).")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES[:-1]),
                    help="stages to run (always executed in pipeline order)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g = ap.add_argument_group("dataset")
    g.add_argument("--episodes", type=int, default=600, help="simulated storm records")
    g.add_argument("--steps", type=int, default=192, help="5-minute steps per record")
    g.add_argument("--stride", type=int, default=2, help="window stride in steps")
    t = ap.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=150, help="maximum epochs (early stopping applies)")
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=12, help="early-stopping patience")
    q = ap.add_argument_group("export / test")
    q.add_argument("--calibration-samples", type=int, default=100)
    q.add_argument("--max-test-samples", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dirs()
    selected = [s for s in STAGES if s in args.stages]   # keep canonical order
    t_start = time.perf_counter()

    # Modules are imported lazily so e.g. `--stages generate` never loads TensorFlow.
    def stage_generate():
        importlib.import_module("generate_dataset").run(args.episodes, args.steps, args.stride, args.seed)

    def stage_train():
        importlib.import_module("train_model").run(args.epochs, args.batch_size, args.lr, args.patience, args.seed)

    def stage_quantize():
        importlib.import_module("quantize_and_export").run(args.calibration_samples, args.seed)

    def stage_test():
        importlib.import_module("test_model_local").run(args.max_test_samples)

    def stage_control():
        importlib.import_module("controller_simulation").run(args.episodes, args.steps, args.seed)

    runners = {
        "generate": stage_generate,
        "train": stage_train,
        "quantize": stage_quantize,
        "test": stage_test,
        "control": stage_control,
    }

    for number, stage in enumerate(selected, start=1):
        log.info("=" * 70)
        log.info("STAGE %d/%d: %s", number, len(selected), stage.upper())
        log.info("=" * 70)
        t0 = time.perf_counter()
        runners[stage]()
        log.info("Stage '%s' finished in %.1fs", stage, time.perf_counter() - t0)

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE in %.1fs", time.perf_counter() - t_start)
    if MODEL_HEADER_PATH.exists():
        log.info("C++ model header : %s (%d bytes)", MODEL_HEADER_PATH, MODEL_HEADER_PATH.stat().st_size)
    if METRICS_JSON.exists():
        m = json.loads(METRICS_JSON.read_text())
        log.info("Test MAE (m3/s)  : float32 %.5f | INT8 %.5f", m["float32_vs_actual"]["mae"], m["int8_vs_actual"]["mae"])
    log.info("All artifacts    : %s", ARTIFACT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
