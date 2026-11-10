#!/usr/bin/env bash
# Physical force-prompt CEER residual: teacher -> estimator adaptation -> finetune.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$project_dir/dataset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="${ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
cache="${FORCE_CACHE:-$project_dir/artifacts/offline_force_baseline/train37_budget60}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_ROOT:-$project_dir/outputs/${run_stamp}-ceer-force-prompt-physical-student}"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$project_dir/outputs/ceer-force-prompt-physical-student-latest.txt"

test -f "$cache/manifest.json"
"$python_bin" "$project_dir/scripts/offline_force_baseline/validate_training_cache.py" --cache "$cache"
lowlevel="$project_dir/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt"
test -f "$lowlevel"
sha256sum "$lowlevel" > "$output_dir/frozen_lowlevel.sha256"
cp "$cache/manifest.json" "$output_dir/force_manifest.json"

# Snapshot the exact code used by this training so later workspace edits cannot
# alter the running jobs between stages.
"$python_bin" - "$project_dir" "$output_dir/code" <<'PYCODE'
import hashlib, json, shutil, sys
from pathlib import Path
root, dest = Path(sys.argv[1]), Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=False)
for name in ["active_adaptation", "scripts", "cfg"]:
    shutil.copytree(root / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in dest.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}}
(dest / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
PYCODE
export PYTHONPATH="$output_dir/code:$PYTHONPATH"

IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
checkpoint_args=("checkpoint_path=null")
for phase in train adapt finetune; do
  case "$phase" in
    train) algo=ceer_force_prompt_student; frames="${TEACHER_FRAMES:-2000000000}";;
    adapt) algo=ceer_force_prompt_student_adapt; frames="${ADAPT_FRAMES:-100000000}";;
    finetune) algo=ceer_force_prompt_student_finetune; frames="${STUDENT_FRAMES:-2000000000}";;
  esac
  phase_dir="$output_dir/$phase"
  mkdir -p "$phase_dir"
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#gpu_list[@]}" \
    "$output_dir/code/scripts/train.py" \
    task=G1/G1_ceer_offline_force_physical_student "algo=$algo" \
    "task.command.guidance_cache=$cache" \
    "task.num_envs=${NUM_ENVS_PER_GPU:-20480}" "total_frames=$frames" \
    "${checkpoint_args[0]}" vecnorm=train "save_interval=${SAVE_INTERVAL:-50}" skip_final_eval=true \
    "wandb.mode=${WANDB_MODE:-online}" wandb.project=ceer-force-prompt-student \
    "wandb.run_name=ceer-force-prompt-student-$phase-$run_stamp" \
    "hydra.run.dir=$phase_dir" \
    2>&1 | tee "$phase_dir/train.log"

  mapfile -t checkpoints < <(find "$phase_dir" -type f -name checkpoint_final.pt)
  [[ ${#checkpoints[@]} -eq 1 ]] || {
    echo "Expected one final checkpoint under $phase_dir; found ${#checkpoints[@]}" >&2
    exit 1
  }
  checkpoint_args=("checkpoint_path=${checkpoints[0]}")
done

printf '%s\n' "${checkpoint_args[0]#checkpoint_path=}" > "$output_dir/final_checkpoint.txt"
echo "Physical force-prompt teacher/adapt/finetune pipeline finished: $output_dir"
