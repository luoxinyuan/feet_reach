#!/usr/bin/env bash
# SONIC low-level force-prompt teacher -> estimator adaptation -> finetune.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$project_dir/dataset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="${ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
cache="${FORCE_CACHE:-$project_dir/artifacts/offline_force_baseline/train37_budget60}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_ROOT:-$project_dir/outputs/${run_stamp}-sonic-force-prompt-student}"
lowlevel="$project_dir/.cache/sonic"

test -f "$cache/manifest.json"
test -f "$lowlevel/manifest.json"
test -f "$lowlevel/actor.pt"
"$python_bin" "$project_dir/scripts/offline_force_baseline/validate_training_cache.py" --cache "$cache"
if [[ -e "$output_dir" ]]; then
  echo "Output already exists: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$project_dir/outputs/sonic-force-prompt-student-latest.txt"
cp "$cache/manifest.json" "$output_dir/force_manifest.json"
cp "$lowlevel/manifest.json" "$output_dir/sonic_manifest.json"
sha256sum "$lowlevel/actor.pt" > "$output_dir/frozen_sonic_actor.sha256"
dataset_paths_json="$("$python_bin" - "$cache/manifest.json" <<'PYCODE'
import json, sys
print(json.dumps(json.load(open(sys.argv[1]))["dataset_paths"], separators=(",", ":")))
PYCODE
)"

# Snapshot training code. SONIC's fixed, hash-validated resources are symlinked
# into that snapshot so imports resolve against the intended model/dependencies.
"$python_bin" - "$project_dir" "$output_dir/code" "$lowlevel" <<'PYCODE'
import hashlib, json, shutil, sys
from pathlib import Path
root, dest, sonic = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dest.mkdir(parents=True, exist_ok=False)
for name in ["active_adaptation", "scripts", "cfg"]:
    shutil.copytree(root / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
(dest / ".cache").mkdir()
(dest / ".cache" / "sonic").symlink_to(sonic, target_is_directory=True)
hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in dest.rglob("*") if p.is_file() and not p.is_symlink()
          and p.suffix in {".py", ".yaml", ".sh"}}
(dest / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
PYCODE
export PYTHONPATH="$output_dir/code:$PYTHONPATH"

IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
phases=(train adapt finetune)
start_phase="${START_PHASE:-train}"
start_index=-1
for i in "${!phases[@]}"; do
  [[ "${phases[$i]}" == "$start_phase" ]] && start_index="$i"
done
[[ "$start_index" -ge 0 ]] || { echo "START_PHASE must be train, adapt, or finetune" >&2; exit 2; }
checkpoint="${RESUME_CHECKPOINT:-null}"
resume_frames="${RESUME_FRAMES:-0}"
if [[ "$checkpoint" != null && ! -f "$checkpoint" ]]; then
  echo "Resume checkpoint not found: $checkpoint" >&2
  exit 2
fi
for i in "${!phases[@]}"; do
  phase="${phases[$i]}"
  (( i < start_index )) && continue
  case "$phase" in
    train) algo=sonic_force_prompt_student; frames="${TEACHER_FRAMES:-2000000000}";;
    adapt) algo=sonic_force_prompt_student_adapt; frames="${ADAPT_FRAMES:-100000000}";;
    finetune) algo=sonic_force_prompt_student_finetune; frames="${STUDENT_FRAMES:-2000000000}";;
  esac
  remaining_frames="$frames"
  if [[ "$phase" == "$start_phase" && "$resume_frames" -gt 0 ]]; then
    if (( resume_frames >= frames )); then
      echo "RESUME_FRAMES ($resume_frames) must be smaller than $phase budget ($frames)" >&2
      exit 2
    fi
    remaining_frames=$((frames - resume_frames))
  else
    resume_frames=0
  fi
  phase_dir="$output_dir/$phase"
  mkdir -p "$phase_dir"
  export WANDB_DIR="$phase_dir/wandb"
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#gpu_list[@]}" \
    "$output_dir/code/scripts/train.py" \
    task=G1/G1_sonic_offline_force_physical_student "algo=$algo" \
    "task.command.guidance_cache=$cache" \
    "task.command.dataset.mem_paths=$dataset_paths_json" \
    "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=$remaining_frames" \
    "phase_frames=$frames" "resume_frames=$resume_frames" \
    "checkpoint_path=$checkpoint" vecnorm=train "save_interval=${SAVE_INTERVAL:-50}" skip_final_eval=true \
    "wandb.mode=${WANDB_MODE:-online}" wandb.project=sonic-force-prompt-student \
    "wandb.run_name=sonic-force-prompt-student-$phase-$run_stamp" \
    "hydra.run.dir=$phase_dir" "$@" 2>&1 | tee "$phase_dir/train.log"

  mapfile -t checkpoints < <(find "$phase_dir" -type f -name checkpoint_final.pt)
  [[ ${#checkpoints[@]} -eq 1 ]] || {
    echo "Expected exactly one final checkpoint under $phase_dir; found ${#checkpoints[@]}" >&2
    exit 1
  }
  checkpoint="$(realpath "${checkpoints[0]}")"
  printf '%s\n' "$checkpoint" > "$output_dir/${phase}_checkpoint.txt"
  resume_frames=0
done

printf '%s\n' "$checkpoint" > "$output_dir/final_checkpoint.txt"
echo "SONIC force-prompt teacher/adapt/finetune pipeline finished: $output_dir"
