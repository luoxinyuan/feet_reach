#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
phase="${1:?Usage: train_ceer_dense_force_stage.sh teacher|distill|finetune [hydra overrides]}"; shift
case "$phase" in teacher|distill|finetune) ;; *) echo "Unknown phase: $phase" >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
cache="${FORCE_CACHE:-$PWD/artifacts/ceer_dense_link/full23/train}"
cache="$(realpath "$cache")"
"$python_bin" scripts/references/validate_dense_link_cache.py --cache "$cache" --full-hash
source_dataset="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_dataset"])' "$cache/manifest.json")"
checkpoint="${CHECKPOINT:-null}"
if [[ "$phase" != teacher && "$checkpoint" == null ]]; then echo "$phase requires CHECKPOINT from the preceding stage" >&2; exit 2; fi
if [[ "$checkpoint" != null ]]; then checkpoint="$(realpath "$checkpoint")"; test -f "$checkpoint"; fi
norm=train; default_frames=4000000000
if [[ "$phase" != teacher ]]; then norm=eval; fi
if [[ "$phase" == distill ]]; then default_frames=1000000000; fi
output="${OUTPUT_DIR:-$PWD/outputs/ceer-dense-force/$(date +%Y%m%d_%H%M%S)/$phase}"
if [[ -f "$output/train.log" ]]; then echo "Output already used: $output; choose a fresh OUTPUT_DIR (and CHECKPOINT to resume)" >&2; exit 2; fi
mkdir -p "$output"
cp "$cache/manifest.json" "$output/force_manifest.json"
sha256sum .cache/ceer/checkpoint_final.pt > "$output/frozen_ceer.sha256"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
 task=G1/G1_ceer_dense_link_compliance algo=ceer_dense_force "algo.phase=$phase" \
 "task.command.guidance_cache=$cache" "task.command.dataset.mem_paths=[$source_dataset]" \
 "task.num_envs=${NUM_ENVS:-2048}" "total_frames=${TOTAL_FRAMES:-$default_frames}" \
 "checkpoint_path=$checkpoint" "vecnorm=$norm" save_interval=50 \
 "wandb.mode=${WANDB_MODE:-online}" wandb.project=ceer-dense-link-force-compliance \
 "wandb.run_name=dense23-$phase-$(basename "$(dirname "$output")")" \
 "hydra.run.dir=$output" "$@" 2>&1 | tee "$output/train.log"
