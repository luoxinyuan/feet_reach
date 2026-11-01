# CEER online-estimated 9-point compliance

This extends the staged online-estimated-force method from root3 to the earlier
nine-point Jacobian compliance setup. Force channels are ordered as torso,
left shoulder/elbow/wrist/hand, then right shoulder/elbow/wrist/hand. The actor
does not receive ground-truth force; a causal estimator predicts 27 normalized
force values from ten frames of proprioception and low-level action history.

The task inherits the previous nine-point Jacobian no-EE-tracking setup: EE
tracking reward is disabled, while joint and keypoint tracking rewards remain.
The frozen CEER low-level policy and Jacobian force allocation are unchanged.
No offline force cache is used.

Training runs on GPUs 2, 3, and 4 with the same three phases as online-estimated
root3: 4B-frame GT-force teacher, 1B-frame student-visited-state distillation,
then 4B-frame estimated-force PPO. Default is 16,384 environments per GPU.

```bash
CUDA_VISIBLE_DEVICES=2,3,4 NUM_ENVS=16384 WANDB_MODE=online \
  bash scripts/train_ceer_online_estimated_9pt_pipeline.sh
```

Output is placed under `outputs/YYYYMMDD-ceer-online-estimated-9pt-no-ee/`;
the W&B project is `ceer-online-estimated-9pt-compliance`.

## Current run (2026-10-30)

- tmux: `ceer-online-estimated-9pt` (GPUs 2, 3, 4)
- Output: `outputs/20261030-ceer-online-estimated-9pt-no-ee/20261030_170115/`
- Teacher run: [W&B](https://wandb.ai/luoxinyuan-duke-university/ceer-online-estimated-9pt-compliance/runs/bq9d1uhq)
- Initial rollout and PPO update succeeded at about 130k frames/s; estimated total
  pipeline wall time is 19–20 hours, subject to shared GPU load.
