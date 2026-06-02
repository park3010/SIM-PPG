#!/usr/bin/env bash
set -euo pipefail

GENERIC_CONFIG="training/SigD/config/papagei_s_generic_supcon_head_only_seed42.yaml"
CS_CONFIG="training/SigD/config/papagei_s_cs_supcon_head_only_seed42.yaml"
CUSTOM_CONFIG=""
AUDIT_ONLY=false
SMOKE_TEST=false
TRAIN_GENERIC=false
TRAIN_CS=false
EVALUATE_GENERIC=false
EVALUATE_CS=false
OVERWRITE=false

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
    --train-generic)
      TRAIN_GENERIC=true
      shift
      ;;
    --train-cs)
      TRAIN_CS=true
      shift
      ;;
    --evaluate-generic)
      EVALUATE_GENERIC=true
      shift
      ;;
    --evaluate-cs)
      EVALUATE_CS=true
      shift
      ;;
    --config)
      CUSTOM_CONFIG="$2"
      shift 2
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

echo "[SigD adaptation] syntax check"
python -m compileall training/SigD/src training/SigD/scripts training/SigD/tests

echo "[SigD adaptation] unit tests"
pytest training/SigD/tests -q

if [[ "$AUDIT_ONLY" == true ]]; then
  echo "[SigD adaptation] training engine audit"
  python training/SigD/scripts/audit_training_engine.py
  exit 0
fi

train_args=()
if [[ "$SMOKE_TEST" == true ]]; then
  train_args+=(--smoke-test --max-epochs 1 --num-batches-per-epoch 2)
fi
if [[ "$OVERWRITE" == true ]]; then
  train_args+=(--overwrite)
fi

if [[ -n "$CUSTOM_CONFIG" && "$TRAIN_GENERIC" == false && "$TRAIN_CS" == false && "$EVALUATE_GENERIC" == false && "$EVALUATE_CS" == false ]]; then
  python training/SigD/scripts/train_adaptation.py --config "$CUSTOM_CONFIG" "${train_args[@]}"
fi

if [[ "$TRAIN_GENERIC" == true ]]; then
  python training/SigD/scripts/train_adaptation.py --config "$GENERIC_CONFIG" "${train_args[@]}"
fi

if [[ "$TRAIN_CS" == true ]]; then
  python training/SigD/scripts/train_adaptation.py --config "$CS_CONFIG" "${train_args[@]}"
fi

if [[ "$EVALUATE_GENERIC" == true ]]; then
  python training/SigD/scripts/evaluate_adapted_model.py --config "$GENERIC_CONFIG"
fi

if [[ "$EVALUATE_CS" == true ]]; then
  python training/SigD/scripts/evaluate_adapted_model.py --config "$CS_CONFIG"
fi

echo "[SigD adaptation] done"
