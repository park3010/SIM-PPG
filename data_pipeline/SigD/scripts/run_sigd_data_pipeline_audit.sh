#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
if [[ ! -d "${ROOT_DIR}/data_pipeline/SigD" || ! -d "${ROOT_DIR}/preprocessing/SigD" || ! -d "${ROOT_DIR}/protocol/SigD" ]]; then
  echo "Run this script from the SIM_PPG root." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[SigD data pipeline] 1/3 syntax check"
"${PYTHON_BIN}" -m compileall data_pipeline/SigD/src data_pipeline/SigD/scripts data_pipeline/SigD/tests

echo "[SigD data pipeline] 2/3 unit tests"
pytest data_pipeline/SigD/tests -q

echo "[SigD data pipeline] 3/3 data pipeline audit"
"${PYTHON_BIN}" data_pipeline/SigD/scripts/audit_sigd_data_pipeline.py

echo "[SigD data pipeline] done"
