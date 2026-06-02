#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
PROTOCOL_DIR="${ROOT_DIR}/protocol/SigD"
PYTHON_BIN="${PYTHON_BIN:-python}"

OVERWRITE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${PROTOCOL_DIR}" || ! -d "${ROOT_DIR}/preprocessing/SigD" ]]; then
  echo "Run this script from the SIM_PPG root." >&2
  exit 1
fi

OVERWRITE_FLAG=()
if [[ "${OVERWRITE}" -eq 1 ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "[SigD protocol] 1/3 subject split"
"${PYTHON_BIN}" protocol/SigD/scripts/build_sigd_subject_split.py --root . "${OVERWRITE_FLAG[@]}"

echo "[SigD protocol] 2/3 verification protocol"
"${PYTHON_BIN}" protocol/SigD/scripts/build_sigd_verification_protocol.py --root . "${OVERWRITE_FLAG[@]}"

echo "[SigD protocol] 3/3 leakage/protocol audit"
"${PYTHON_BIN}" protocol/SigD/scripts/audit_sigd_protocol.py --root .

echo "[SigD protocol] done"
