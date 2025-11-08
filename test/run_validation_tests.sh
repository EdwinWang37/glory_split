#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT="$ROOT_DIR"

echo "[1/2] Running unit shape checks (CPU)…"
PYTHONPATH="$ROOT_DIR/src" pytest -q "$ROOT_DIR/test/test_glorysplit_shapes_unit.py"

echo "[2/2] Running E2E validation smoke (CPU, 2 samples, workers 0/2)…"
RUN_E2E=1 PYTHONPATH="$ROOT_DIR/src" pytest -q "$ROOT_DIR/test/test_val_e2e_smoke.py" -k test_val_smoke_workers_0_and_2

echo "All validation tests passed."

