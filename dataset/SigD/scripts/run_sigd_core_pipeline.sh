#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
SIGD_DIR="${ROOT_DIR}/dataset/SigD"
PYTHON_BIN="${PYTHON_BIN:-python}"

FULL=0
REFRESH_SOURCE=0
SKIP_SMOKE=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      FULL=1
      shift
      ;;
    --refresh-source)
      REFRESH_SOURCE=1
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
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

if [[ ! -d "${SIGD_DIR}" ]]; then
  echo "Run this script from the sim_ppg root; dataset/SigD was not found." >&2
  exit 1
fi

VERBOSE_FLAG=()
if [[ "${VERBOSE}" -eq 1 ]]; then
  VERBOSE_FLAG=(--verbose)
fi

SOURCE_FLAGS=()
if [[ "${REFRESH_SOURCE}" -eq 1 ]]; then
  SOURCE_FLAGS=(--refresh-source)
fi

echo "[SigD-Core] 1/5 source setup"
"${PYTHON_BIN}" dataset/SigD/scripts/setup_sigd_source.py --root . "${SOURCE_FLAGS[@]}" "${VERBOSE_FLAG[@]}"

echo "[SigD-Core] 2/5 annotation parsing"
"${PYTHON_BIN}" dataset/SigD/scripts/parse_sigd_annotations.py --root . "${VERBOSE_FLAG[@]}"

if [[ "${FULL}" -eq 1 ]]; then
  echo "[SigD-Core] 3/5 short header-check"
  "${PYTHON_BIN}" dataset/SigD/scripts/reconstruct_sigd_core.py --root . --header-check --limit-ranges 2 "${VERBOSE_FLAG[@]}"

  echo "[SigD-Core] 4/5 full reconstruction with resume"
  "${PYTHON_BIN}" dataset/SigD/scripts/reconstruct_sigd_core.py --root . --resume "${VERBOSE_FLAG[@]}"

  echo "[SigD-Core] 5/5 audit"
  "${PYTHON_BIN}" dataset/SigD/scripts/audit_sigd_core.py --root . "${VERBOSE_FLAG[@]}"
else
  echo "[SigD-Core] 3/6 dry-run for first 5 raw ranges"
  "${PYTHON_BIN}" dataset/SigD/scripts/reconstruct_sigd_core.py --root . --dry-run --limit-ranges 5 "${VERBOSE_FLAG[@]}"

  echo "[SigD-Core] 4/6 header-check for first 2 raw ranges"
  "${PYTHON_BIN}" dataset/SigD/scripts/reconstruct_sigd_core.py --root . --header-check --limit-ranges 2 "${VERBOSE_FLAG[@]}"

  if [[ "${SKIP_SMOKE}" -eq 0 ]]; then
    echo "[SigD-Core] 5/6 resume smoke extraction for 1 subject / 2 raw ranges"
    "${PYTHON_BIN}" dataset/SigD/scripts/reconstruct_sigd_core.py --root . --resume --limit-subjects 1 --limit-ranges 2 "${VERBOSE_FLAG[@]}"
  else
    echo "[SigD-Core] 5/6 smoke extraction skipped"
  fi

  echo "[SigD-Core] 6/6 audit"
  "${PYTHON_BIN}" dataset/SigD/scripts/audit_sigd_core.py --root . "${VERBOSE_FLAG[@]}"
fi

echo "[SigD-Core] done"
