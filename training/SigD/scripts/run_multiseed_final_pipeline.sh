#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SEEDS="42 52 123 777 2026"
DEVICE="auto"
AUDIT_ONLY=false
TRAIN_E4=false
EVAL_E4=false
TRAIN_E7A=false
EVAL_E7A=false
SKIP_EXISTING=false
OVERWRITE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit-only)
      AUDIT_ONLY=true
      shift
      ;;
    --train-e4)
      TRAIN_E4=true
      shift
      ;;
    --eval-e4)
      EVAL_E4=true
      shift
      ;;
    --train-e7a)
      TRAIN_E7A=true
      shift
      ;;
    --eval-e7a)
      EVAL_E7A=true
      shift
      ;;
    --all)
      TRAIN_E4=true
      EVAL_E4=true
      TRAIN_E7A=true
      EVAL_E7A=true
      shift
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=true
      shift
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

python -m compileall training/SigD/src training/SigD/scripts training/SigD/tests
pytest training/SigD/tests -q

if [[ "$AUDIT_ONLY" == true ]]; then
  python training/SigD/scripts/audit_multiseed_results.py --seeds "$SEEDS" --allow-missing
  exit 0
fi

overwrite_args=()
if [[ "$OVERWRITE" == true ]]; then
  overwrite_args+=(--overwrite)
fi

e4_config="training/SigD/config/papagei_s_generic_supcon_head_only_seed42.yaml"
e7a_config="training/SigD/config/papagei_s_e7_a_generic_supcon_morph_e4_branch_seed42.yaml"

train_exists() {
  local root_path="$1"
  [[ -f "$root_path/manifest.json" && -f "$root_path/checkpoints/best_projection_head.pt" ]]
}

eval_exists() {
  local root_path="$1"
  [[ -f "$root_path/final_exhaustive_evaluation/test_metrics.json" && -f "$root_path/final_exhaustive_evaluation/test_scores.csv" ]]
}

for seed in $SEEDS; do
  e4_root="training/SigD/results/papagei_s_generic_supcon_head_only/seed${seed}"
  e7a_root="training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/svri0p05_sqi0p05/seed${seed}"
  e7a_legacy_seed42_root="training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/seed42"

  if [[ "$TRAIN_E4" == true ]]; then
    if [[ "$SKIP_EXISTING" == true ]] && train_exists "$e4_root"; then
      echo "skip_existing_train_e4 seed=${seed} root=${e4_root}"
    else
      python training/SigD/scripts/train_adaptation.py --config "$e4_config" --seed "$seed" --device "$DEVICE" "${overwrite_args[@]}"
    fi
  fi

  if [[ "$EVAL_E4" == true ]]; then
    if [[ "$SKIP_EXISTING" == true ]] && eval_exists "$e4_root"; then
      echo "skip_existing_eval_e4 seed=${seed} root=${e4_root}"
    else
      python training/SigD/scripts/evaluate_adapted_model.py --config "$e4_config" --seed "$seed" --checkpoint "$e4_root/checkpoints/best_projection_head.pt" --device "$DEVICE" "${overwrite_args[@]}"
    fi
  fi

  if [[ "$TRAIN_E7A" == true ]]; then
    if [[ "$SKIP_EXISTING" == true ]] && train_exists "$e7a_root"; then
      echo "skip_existing_train_e7a seed=${seed} root=${e7a_root}"
    else
      python training/SigD/scripts/train_adaptation.py --config "$e7a_config" --seed "$seed" --candidate-name svri0p05_sqi0p05 --lambda-svri 0.05 --lambda-sqi 0.05 --device "$DEVICE" "${overwrite_args[@]}"
    fi
  fi

  if [[ "$EVAL_E7A" == true ]]; then
    if [[ "$SKIP_EXISTING" == true ]] && eval_exists "$e7a_root"; then
      echo "skip_existing_eval_e7a seed=${seed} root=${e7a_root}"
    elif [[ "$SKIP_EXISTING" == true && "$seed" == "42" ]] && eval_exists "$e7a_legacy_seed42_root"; then
      echo "skip_existing_eval_e7a seed=42 root=${e7a_legacy_seed42_root} legacy_seed42_eval=true"
    else
      python training/SigD/scripts/evaluate_adapted_model.py --config "$e7a_config" --seed "$seed" --candidate-name svri0p05_sqi0p05 --lambda-svri 0.05 --lambda-sqi 0.05 --checkpoint "$e7a_root/checkpoints/best_projection_head.pt" --device "$DEVICE" "${overwrite_args[@]}"
    fi
  fi
done
