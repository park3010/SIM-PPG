#!/usr/bin/env bash
set -euo pipefail

CONFIG="protocol/SigD/config/sigd_protocol_10s_k5m1_exhaustive_eval_v2.yaml"
OVERWRITE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "protocol/SigD" || ! -d "preprocessing/SigD" ]]; then
  echo "Run this script from the SIM_PPG project root." >&2
  exit 1
fi

echo "[SigD exhaustive protocol] 1/4 syntax check"
python -m compileall protocol/SigD/scripts protocol/SigD/tests

echo "[SigD exhaustive protocol] 2/4 unit tests"
pytest protocol/SigD/tests -q

echo "[SigD exhaustive protocol] 3/4 build exhaustive evaluation protocol"
BUILD_ARGS=(--config "$CONFIG")
if [[ "$OVERWRITE" == true ]]; then
  BUILD_ARGS+=(--overwrite)
fi
python protocol/SigD/scripts/build_sigd_exhaustive_evaluation_protocol.py "${BUILD_ARGS[@]}"

echo "[SigD exhaustive protocol] 4/4 audit exhaustive evaluation protocol"
python protocol/SigD/scripts/audit_sigd_exhaustive_evaluation_protocol.py --config "$CONFIG"

echo "[SigD exhaustive protocol] done"
