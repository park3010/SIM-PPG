#!/usr/bin/env bash
set -euo pipefail

E7_A_CONFIG="training/SigD/config/papagei_s_e7_a_generic_supcon_morph_e4_branch_seed42.yaml"
E7_B_CONFIG="training/SigD/config/papagei_s_e7_b_generic_supcon_morph_cs_batch_branch_seed42.yaml"
AUDIT_ONLY=false
SMOKE_TEST=false
TRAIN_E7_A=false
TRAIN_E7_B=false
SELECT_E7_A=false
SELECT_E7_B=false
EVALUATE_SELECTED_E7_A=false
EVALUATE_SELECTED_E7_B=false
OVERWRITE=false
CANDIDATE_NAME=""
LAMBDA_SVRI=""
LAMBDA_SQI=""

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
    --train-e7-a)
      TRAIN_E7_A=true
      shift
      ;;
    --train-e7-b)
      TRAIN_E7_B=true
      shift
      ;;
    --candidate-name)
      CANDIDATE_NAME="$2"
      shift 2
      ;;
    --lambda-svri)
      LAMBDA_SVRI="$2"
      shift 2
      ;;
    --lambda-sqi)
      LAMBDA_SQI="$2"
      shift 2
      ;;
    --select-e7-a)
      SELECT_E7_A=true
      shift
      ;;
    --select-e7-b)
      SELECT_E7_B=true
      shift
      ;;
    --evaluate-selected-e7-a)
      EVALUATE_SELECTED_E7_A=true
      shift
      ;;
    --evaluate-selected-e7-b)
      EVALUATE_SELECTED_E7_B=true
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

echo "[SigD E7 morphology] syntax check"
python -m compileall training/SigD/src training/SigD/scripts training/SigD/tests

echo "[SigD E7 morphology] unit tests"
pytest training/SigD/tests -q

if [[ "$AUDIT_ONLY" == true ]]; then
  echo "[SigD E7 morphology] morphology engine audit"
  python training/SigD/scripts/audit_morphology_engine.py
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
if [[ -n "$LAMBDA_SVRI" ]]; then
  train_args+=(--lambda-svri "$LAMBDA_SVRI")
fi
if [[ -n "$LAMBDA_SQI" ]]; then
  train_args+=(--lambda-sqi "$LAMBDA_SQI")
fi

if [[ "$TRAIN_E7_A" == true ]]; then
  python training/SigD/scripts/train_adaptation.py --config "$E7_A_CONFIG" "${train_args[@]}"
fi

if [[ "$TRAIN_E7_B" == true ]]; then
  python training/SigD/scripts/train_adaptation.py --config "$E7_B_CONFIG" "${train_args[@]}"
fi

if [[ "$SELECT_E7_A" == true || "$SELECT_E7_B" == true || "$EVALUATE_SELECTED_E7_A" == true || "$EVALUATE_SELECTED_E7_B" == true ]]; then
  echo "Selection/final evaluation are available as explicit scripts but are not run in this implementation smoke phase." >&2
  exit 2
fi

echo "[SigD E7 morphology] done"

