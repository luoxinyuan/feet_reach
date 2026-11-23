#!/usr/bin/env bash
# 37-point, total-force-60N physical force-prompt teacher -> adapt -> student,
# with the released frozen SuperMimic G1 controller as the low-level tracker.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$project_dir/dataset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="${ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
cache="${FORCE_CACHE:-$project_dir/artifacts/offline_force_baseline/train37_budget60}"
release_root="${SUPER_MIMIC_SOURCE:-/home/xl521/supermimic_force_transition}"
release_dir="$release_root/pretrained/e11c_0220000"
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_ROOT:-$project_dir/outputs/${run_stamp}-supermimic-37pt-budget60-force-prompt-student}"
frozen_root="$output_dir/frozen_supermimic"

test -f "$cache/manifest.json"
test -f "$release_dir/ckpt_e11c_0220000.pt"
test -f "$release_dir/deploy_config.json"
"$python_bin" "$project_dir/scripts/offline_force_baseline/validate_training_cache.py" --cache "$cache"
if [[ -e "$output_dir" ]]; then
  echo "Output already exists: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$project_dir/outputs/supermimic-37pt-budget60-force-prompt-student-latest.txt"
cat > "$output_dir/README.md" <<EOF
# SuperMimic low-level, 37-point physical force-prompt teacher/student

Reference configuration: the 2026-11-09 SONIC force-prompt physical student
pipeline, task `G1_sonic_offline_force_physical_student`, adapted by replacing
the frozen SONIC joint-reference adapter with `SuperMimicResidualAction`.
The previous SONIC checkpoint's v1.1 validation was 0.218 rad joint MAE and
96.9% physical completion; it is a reference, not a result for this run.

Force cache: `$cache`. It has 37 application points and 16,021 clips
(4,183,180 frames; 4,167,159 transitions), with per-point force capped at
30 N, sum of point force magnitudes capped at 60 N, and 150 N/s slew. Its
measured average total force is 5.389 N. Dataset paths use weights [0.8, 0.2]
as in the reference pipeline. Current-frame force tokens are provided to the
actor while the same cached forces are applied physically; no force history.

Pipeline budgets are 2.0B teacher + 0.1B adaptation + 2.0B student-finetune
environment steps. GPUs: $CUDA_VISIBLE_DEVICES. Frozen low-level release:
SuperMimic e11c_0220000; the output contains checkpoint, deploy config and code.
EOF
cp "$cache/manifest.json" "$output_dir/force_manifest.json"
cp "$cache/REPORT.md" "$output_dir/force_cache_report.md"
mkdir -p "$frozen_root/pretrained/e11c_0220000"
cp "$release_dir/ckpt_e11c_0220000.pt" "$frozen_root/pretrained/e11c_0220000/"
cp "$release_dir/deploy_config.json" "$frozen_root/pretrained/e11c_0220000/"
cp -a "$release_root/code" "$frozen_root/code"
export SUPER_MIMIC_ROOT="$frozen_root"
sha256sum "$release_dir/ckpt_e11c_0220000.pt" "$release_dir/deploy_config.json" > "$output_dir/supermimic_sha256.txt"
"$python_bin" - "$cache/manifest.json" "$frozen_root" "$output_dir" <<'PYCODE'
import hashlib, json, sys
from pathlib import Path
manifest, frozen, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
m = json.loads(manifest.read_text())
files = [frozen / 'pretrained/e11c_0220000/ckpt_e11c_0220000.pt',
         frozen / 'pretrained/e11c_0220000/deploy_config.json']
files += [p for p in (frozen / 'code').rglob('*') if p.is_file()]
record = {'cache_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
          'cache_dataset_paths': m['dataset_paths'],
          'supermimic_files': {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in files}}
(out / 'input_hashes.json').write_text(json.dumps(record, indent=2) + '\n')
PYCODE

dataset_paths_json="$("$python_bin" - "$cache/manifest.json" <<'PYCODE'
import json, sys
print(json.dumps(json.load(open(sys.argv[1]))['dataset_paths'], separators=(',', ':')))
PYCODE
)"

# Snapshot the exact project sources/configs used for every phase.
"$python_bin" - "$project_dir" "$output_dir/code" <<'PYCODE'
import hashlib, json, shutil, sys
from pathlib import Path
root, dest = Path(sys.argv[1]), Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=False)
for name in ['active_adaptation', 'scripts', 'cfg']:
    shutil.copytree(root / name, dest / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in dest.rglob('*') if p.is_file() and not p.is_symlink()
          and p.suffix in {'.py', '.yaml', '.sh'}}
(dest / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
PYCODE
export PYTHONPATH="$output_dir/code:$PYTHONPATH"

IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
echo "Run root: $output_dir"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
echo "Phase budgets (global environment steps): teacher=${TEACHER_FRAMES:-2000000000}, adapt=${ADAPT_FRAMES:-100000000}, finetune=${STUDENT_FRAMES:-2000000000}"
echo "Force cache: $cache"
echo "Force prompt: 37 points, per-point <=30 N, sum of point magnitudes <=60 N, <=150 N/s slew"
echo "Physical force application: enabled; current-frame force tokens; no force history"
echo "Dataset mixture: cache train paths weighted 0.8/0.2 (whole-body/EE/static mixture plus crouch)"
echo "Low-level: frozen SuperMimic e11c_0220000"
echo "Actor output: 29-DOF joint-reference residual"

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
    train) algo=supermimic_force_prompt_student; frames="${TEACHER_FRAMES:-2000000000}";;
    adapt) algo=supermimic_force_prompt_student_adapt; frames="${ADAPT_FRAMES:-100000000}";;
    finetune) algo=supermimic_force_prompt_student_finetune; frames="${STUDENT_FRAMES:-2000000000}";;
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
    task=G1/G1_supermimic_offline_force_physical_student "algo=$algo" \
    "task.command.guidance_cache=$cache" \
    "task.command.dataset.mem_paths=$dataset_paths_json" \
    task.command.physical_guidance=true \
    "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=$remaining_frames" \
    "phase_frames=$frames" "resume_frames=$resume_frames" \
    "checkpoint_path=$checkpoint" vecnorm=train "save_interval=${SAVE_INTERVAL:-50}" skip_final_eval=true \
    "wandb.mode=${WANDB_MODE:-online}" wandb.project=supermimic-force-prompt-student \
    "wandb.run_name=supermimic-force-prompt-student-$phase-$run_stamp" \
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
echo "SuperMimic force-prompt teacher/adapt/finetune pipeline finished: $output_dir"
