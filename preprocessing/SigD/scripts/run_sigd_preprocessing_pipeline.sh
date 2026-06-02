#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
PREPROCESS_DIR="${ROOT_DIR}/preprocessing/SigD"
PYTHON_BIN="${PYTHON_BIN:-python}"

FULL=0
REFRESH_REFERENCE=0
VERIFY_ALL=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      FULL=1
      shift
      ;;
    --refresh-papagei-reference)
      REFRESH_REFERENCE=1
      shift
      ;;
    --verify-all-npz-hashes)
      VERIFY_ALL=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${PREPROCESS_DIR}" || ! -d "${ROOT_DIR}/dataset/SigD" ]]; then
  echo "Run this script from the SIM_PPG root." >&2
  exit 1
fi

VERBOSE_FLAG=()
if [[ "${VERBOSE}" -eq 1 ]]; then
  VERBOSE_FLAG=(--verbose)
fi

REFERENCE_FLAGS=()
if [[ "${REFRESH_REFERENCE}" -eq 1 ]]; then
  REFERENCE_FLAGS=(--refresh-source)
fi

echo "[SigD preprocessing] 1/5 reference source setup"
"${PYTHON_BIN}" preprocessing/SigD/scripts/setup_papagei_reference.py --root . "${REFERENCE_FLAGS[@]}" "${VERBOSE_FLAG[@]}"

if [[ "${FULL}" -eq 1 ]]; then
  if [[ "${VERIFY_ALL}" -ne 1 ]]; then
    echo "--full requires --verify-all-npz-hashes" >&2
    exit 1
  fi
  echo "[SigD preprocessing] 2/5 full snapshot validation"
  "${PYTHON_BIN}" preprocessing/SigD/scripts/verify_sigd_snapshot.py --root . --verify-all-npz-hashes "${VERBOSE_FLAG[@]}"

  echo "[SigD preprocessing] 3/5 full 10s preprocessing"
  "${PYTHON_BIN}" preprocessing/SigD/scripts/preprocess_sigd_windows.py --root . --window-seconds 10 --verify-all-npz-hashes "${VERBOSE_FLAG[@]}"

  echo "[SigD preprocessing] 4/5 full post-QC audit"
  "${PYTHON_BIN}" preprocessing/SigD/scripts/audit_preprocessed_sigd.py --root . --window-seconds 10 "${VERBOSE_FLAG[@]}"

  echo "[SigD preprocessing] 5/5 done"
else
  echo "[SigD preprocessing] 2/5 smoke snapshot validation"
  "${PYTHON_BIN}" preprocessing/SigD/scripts/verify_sigd_snapshot.py --root . --limit-raw-ranges 2 "${VERBOSE_FLAG[@]}"

  echo "[SigD preprocessing] 3/5 smoke 10s preprocessing"
  "${PYTHON_BIN}" preprocessing/SigD/scripts/preprocess_sigd_windows.py --root . --window-seconds 10 --smoke --verify-selected-npz-hashes "${VERBOSE_FLAG[@]}"

  echo "[SigD preprocessing] 4/5 smoke post-QC audit"
  "${PYTHON_BIN}" preprocessing/SigD/scripts/audit_preprocessed_sigd.py --root . --window-seconds 10 --smoke "${VERBOSE_FLAG[@]}"

  echo "[SigD preprocessing] 5/5 done"
fi
