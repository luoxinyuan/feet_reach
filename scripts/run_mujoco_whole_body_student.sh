#!/usr/bin/env bash
set -euo pipefail
COMPLIANCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPLIANCE_PYTHON="${COMPLIANCE_PYTHON:-python}"
if [[ "${1:-}" == "--web" ]]; then
  shift
  exec "$COMPLIANCE_PYTHON" "$COMPLIANCE_ROOT/scripts/mujoco_whole_body_web.py" "$@"
fi
exec "$COMPLIANCE_PYTHON" "$COMPLIANCE_ROOT/scripts/mujoco_whole_body_student.py" "$@"
