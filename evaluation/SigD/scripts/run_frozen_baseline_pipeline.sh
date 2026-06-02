#!/usr/bin/env bash
set -euo pipefail

CONFIG="evaluation/SigD/config/papagei_s_frozen_cosine_eval.yaml"
AUDIT_ONLY=false
RUN_OFFICIAL=false
DOWNLOAD_OFFICIAL=false
VERIFY_OFFICIAL=false
OVERWRITE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit-only)
      AUDIT_ONLY=true
      shift
      ;;
    --run-official-baseline)
      RUN_OFFICIAL=true
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --download-official-assets)
      DOWNLOAD_OFFICIAL=true
      shift
      ;;
    --verify-official-assets)
      VERIFY_OFFICIAL=true
      shift
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_OFFICIAL" == true ]]; then
  VERIFY_OFFICIAL=true
fi

if [[ ! -d "evaluation/SigD" || ! -d "data_pipeline/SigD" ]]; then
  echo "Run this script from the SIM_PPG project root." >&2
  exit 1
fi

echo "[SigD frozen baseline] 1/4 syntax check"
python -m compileall evaluation/SigD/src evaluation/SigD/scripts evaluation/SigD/tests

echo "[SigD frozen baseline] 2/4 unit tests"
pytest evaluation/SigD/tests -q

echo "[SigD frozen baseline] 3/4 inspect PaPaGei-S local reference"
SETUP_ARGS=(--config "$CONFIG" --verbose)
if [[ "$DOWNLOAD_OFFICIAL" == true ]]; then
  SETUP_ARGS+=(--download-official-assets)
fi
if [[ "$VERIFY_OFFICIAL" == true ]]; then
  SETUP_ARGS+=(--verify)
fi
python evaluation/SigD/scripts/setup_papagei_model_reference.py "${SETUP_ARGS[@]}"

echo "[SigD frozen baseline] 4/4 mock evaluation-engine audit"
python evaluation/SigD/scripts/audit_evaluation_engine.py --config "$CONFIG"

if [[ "$AUDIT_ONLY" == true ]]; then
  echo "[SigD frozen baseline] audit-only mode complete; official baseline not executed"
  exit 0
fi

if [[ "$RUN_OFFICIAL" == true ]]; then
  OFFICIAL_ARGS=(--config "$CONFIG" --allow-skip)
  if [[ "$OVERWRITE" == true ]]; then
    OFFICIAL_ARGS+=(--overwrite)
  fi
  python evaluation/SigD/scripts/run_papagei_s_frozen_baseline.py "${OFFICIAL_ARGS[@]}"
else
  echo "[SigD frozen baseline] official baseline skipped; pass --run-official-baseline after verified checkpoint setup"
fi
