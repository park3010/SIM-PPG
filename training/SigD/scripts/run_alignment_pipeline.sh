#!/usr/bin/env bash
set -euo pipefail

E6_BASE_CONFIG="training/SigD/config/papagei_s_e6_base_generic_cs_batch_noalign_seed42.yaml"
E6_A_CONFIG="training/SigD/config/papagei_s_e6_a_cs_supcon_alignment_seed42.yaml"
E6_B_CONFIG="training/SigD/config/papagei_s_e6_b_generic_supcon_alignment_cs_batch_seed42.yaml"
AUDIT_ONLY=false
SMOKE_TEST=false
TRAIN_E6_BASE=false
TRAIN_E6_A_GRID=false
TRAIN_E6_B_GRID=false
SELECT_E6_A=false
SELECT_E6_B=false
EVALUATE_SELECTED_E6_A=false
EVALUATE_SELECTED_E6_B=false
OVERWRITE=false
LAMBDA_ALIGN=""

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
    --train-e6-base)
      TRAIN_E6_BASE=true
      shift
      ;;
    --train-e6-a-grid)
      TRAIN_E6_A_GRID=true
      shift
      ;;
    --train-e6-b-grid)
      TRAIN_E6_B_GRID=true
      shift
      ;;
    --select-e6-a)
      SELECT_E6_A=true
      shift
      ;;
    --select-e6-b)
      SELECT_E6_B=true
      shift
      ;;
    --evaluate-selected-e6-a)
      EVALUATE_SELECTED_E6_A=true
      shift
      ;;
    --evaluate-selected-e6-b)
      EVALUATE_SELECTED_E6_B=true
      shift
      ;;
    --lambda-align)
      LAMBDA_ALIGN="$2"
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

echo "[SigD E6 alignment] syntax check"
python -m compileall training/SigD/src training/SigD/scripts training/SigD/tests

echo "[SigD E6 alignment] unit tests"
pytest training/SigD/tests -q

if [[ "$AUDIT_ONLY" == true ]]; then
  echo "[SigD E6 alignment] alignment engine audit"
  python training/SigD/scripts/audit_alignment_engine.py
  exit 0
fi

train_args=()
if [[ "$SMOKE_TEST" == true ]]; then
  train_args+=(--smoke-test --max-epochs 1 --num-batches-per-epoch 2)
fi
if [[ "$OVERWRITE" == true ]]; then
  train_args+=(--overwrite)
fi

if [[ "$TRAIN_E6_BASE" == true ]]; then
  python training/SigD/scripts/train_adaptation.py --config "$E6_BASE_CONFIG" "${train_args[@]}"
fi

if [[ "$TRAIN_E6_A_GRID" == true ]]; then
  if [[ -z "$LAMBDA_ALIGN" ]]; then
    echo "--train-e6-a-grid requires --lambda-align during this implementation phase." >&2
    exit 2
  fi
  python training/SigD/scripts/train_adaptation.py --config "$E6_A_CONFIG" --lambda-align "$LAMBDA_ALIGN" "${train_args[@]}"
fi

if [[ "$TRAIN_E6_B_GRID" == true ]]; then
  if [[ -z "$LAMBDA_ALIGN" ]]; then
    echo "--train-e6-b-grid requires --lambda-align during this implementation phase." >&2
    exit 2
  fi
  python training/SigD/scripts/train_adaptation.py --config "$E6_B_CONFIG" --lambda-align "$LAMBDA_ALIGN" "${train_args[@]}"
fi

if [[ "$SELECT_E6_A" == true || "$SELECT_E6_B" == true || "$EVALUATE_SELECTED_E6_A" == true || "$EVALUATE_SELECTED_E6_B" == true ]]; then
  echo "Selection/final evaluation commands are implemented as explicit scripts but are not run in this smoke phase." >&2
  exit 2
fi

echo "[SigD E6 alignment] done"

