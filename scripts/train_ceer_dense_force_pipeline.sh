#!/usr/bin/env bash
# Teacher -> student-visited-state distillation -> estimated-force PPO.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
root="${PIPELINE_DIR:-$PWD/outputs/ceer-dense-force/$(date +%Y%m%d_%H%M%S)}"
cache_root="${CACHE_ROOT:-$PWD/artifacts/ceer_dense_link/full23}"
if [[ ! -f "$cache_root/train/manifest.json" ]]; then
 "$python_bin" scripts/data_process/prepare_dense_link_guidance.py --output "$cache_root"
fi
export FORCE_CACHE="$cache_root/train"
mkdir -p "$root"
checkpoint=null
for phase in teacher distill finetune; do
 case "$phase" in teacher) frames="${TEACHER_FRAMES:-4000000000}";; distill) frames="${DISTILL_FRAMES:-1000000000}";; finetune) frames="${FINETUNE_FRAMES:-4000000000}";; esac
 OUTPUT_DIR="$root/$phase" CHECKPOINT="$checkpoint" TOTAL_FRAMES="$frames" bash scripts/train_ceer_dense_force_stage.sh "$phase" "$@"
 checkpoint="$("$python_bin" - "$root/$phase" <<'PY'
import sys
from pathlib import Path
paths=sorted({p.resolve() for p in Path(sys.argv[1]).resolve().glob('wandb/*/files/checkpoint_final.pt')})
if len(paths)!=1:raise RuntimeError(f'Expected one final checkpoint, found {paths}')
print(paths[0])
PY
)"
 printf '%s\n' "$checkpoint" > "$root/${phase}_checkpoint.txt"
done
