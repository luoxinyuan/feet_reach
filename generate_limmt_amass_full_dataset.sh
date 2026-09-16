#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/path/to/AMASS}"
OUT_DIR="${OUT_DIR:-dataset/limmt_no_foot_compliance_full}"
ALLOWLIST="${ALLOWLIST:-scripts/data_process/allowlist_limmt_no_foot_compliance.json}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/data_process/generate_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --allowlist "$ALLOWLIST" \
  --mem-path "$OUT_DIR"
