#!/usr/bin/env bash
# Reuse the completed 9KP GT-force teacher, distill a causal force estimator,
# then PPO-finetune using only estimated force history at deployment.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export SUPER_MIMIC_ROOT="${SUPER_MIMIC_ROOT:-/home/xl521/supermimic_force_transition}"
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
teacher_checkpoint="${TEACHER_CHECKPOINT:-$PWD/outputs/20261110-supermimic-online-estimated-9pt-crouch20-force-history-gpu567/train/wandb/run-20261110_121638-28wvjxpb/files/checkpoint_final.pt}"
teacher_checkpoint="$(realpath "$teacher_checkpoint")"
test -f "$teacher_checkpoint"
root="${PIPELINE_DIR:-$PWD/outputs/20261111-supermimic-9pt-force-estimator-student-gpu567}"
mkdir -p "$root"
cp "$teacher_checkpoint" "$root/teacher_checkpoint.pt"
sha256sum "$teacher_checkpoint" "$SUPER_MIMIC_ROOT/pretrained/e11c_0220000/ckpt_e11c_0220000.pt" \
  "$SUPER_MIMIC_ROOT/pretrained/e11c_0220000/deploy_config.json" > "$root/pipeline_inputs.sha256"

checkpoint="$teacher_checkpoint"
for phase in distill finetune; do
  case "$phase" in
    distill) frames="${DISTILL_FRAMES:-1000000000}" ;;
    finetune) frames="${FINETUNE_FRAMES:-4000000000}" ;;
  esac
  OUTPUT_DIR="$root/$phase" CHECKPOINT="$checkpoint" TOTAL_FRAMES="$frames" \
    bash scripts/train_supermimic_force_estimator_student_stage.sh "$phase" "$@"
  checkpoint="$("$python_bin" - "$root/$phase" <<'PY'
import sys
from pathlib import Path
paths=sorted({p.resolve() for p in Path(sys.argv[1]).resolve().glob('wandb/*/files/checkpoint_final.pt')})
if len(paths)!=1:
    raise RuntimeError(f"Expected one final checkpoint in {sys.argv[1]}, found {paths}")
print(paths[0])
PY
)"
  printf '%s\n' "$checkpoint" > "$root/${phase}_checkpoint.txt"
done
printf '%s\n' "$checkpoint" > "$root/final_checkpoint.txt"
echo "Student pipeline finished: $root"
