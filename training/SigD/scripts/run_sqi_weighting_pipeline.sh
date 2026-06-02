#!/usr/bin/env bash
set -euo pipefail

E8_CONFIG="training/SigD/config/papagei_s_e8_sqi_weighted_morph_e7a_seed42.yaml"
AUDIT_ONLY=false
SMOKE_TEST=false
TRAIN_E8=false
SELECT_E8=false
EVALUATE_SELECTED_E8=false
OVERWRITE=false
CANDIDATE_NAME=""
SQI_WEIGHTING_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit-only)
      AUDIT_ONLY=true
      shift
      ;;
    --smoke-test)
      SMOKE_TEST=true
      shift
      ;;
    --train-e8)
      TRAIN_E8=true
      shift
      ;;
    --candidate-name)
      CANDIDATE_NAME="$2"
      shift 2
      ;;
    --sqi-weighting-mode)
      SQI_WEIGHTING_MODE="$2"
      shift 2
      ;;
    --select-e8)
      SELECT_E8=true
      shift
      ;;
    --evaluate-selected-e8)
      EVALUATE_SELECTED_E8=true
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

if [[ ! -d "training/SigD" || ! -d "data_pipeline/SigD" || ! -d "evaluation/SigD" ]]; then
  echo "Run this script from the SIM_PPG project root." >&2
  exit 1
fi

echo "[SigD E8 SQI] syntax check"
python -m compileall training/SigD/src training/SigD/scripts training/SigD/tests

echo "[SigD E8 SQI] unit tests"
pytest training/SigD/tests -q

if [[ "$AUDIT_ONLY" == true ]]; then
  echo "[SigD E8 SQI] SQI weighting engine audit"
  python training/SigD/scripts/audit_sqi_weighting_engine.py
  exit 0
fi

train_args=()
if [[ "$SMOKE_TEST" == true ]]; then
  train_args+=(--smoke-test --max-epochs 1 --num-batches-per-epoch 2)
fi
if [[ "$OVERWRITE" == true ]]; then
  train_args+=(--overwrite)
fi
if [[ -n "$CANDIDATE_NAME" ]]; then
  train_args+=(--candidate-name "$CANDIDATE_NAME")
fi
if [[ -n "$SQI_WEIGHTING_MODE" ]]; then
  train_args+=(--sqi-weighting-mode "$SQI_WEIGHTING_MODE")
fi

if [[ "$TRAIN_E8" == true ]]; then
  python training/SigD/scripts/train_adaptation.py --config "$E8_CONFIG" "${train_args[@]}"
fi

if [[ "$SELECT_E8" == true || "$EVALUATE_SELECTED_E8" == true ]]; then
  echo "E8 selection/final evaluation are explicit later steps and are not run in this implementation smoke phase." >&2
  exit 2
fi

echo "[SigD E8 SQI] done"

