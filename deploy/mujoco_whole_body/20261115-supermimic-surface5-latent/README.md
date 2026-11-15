# SuperMimic observable-surface 5-contact latent policy

This bundle exports the completed teacher policy from `outputs/20261114_230110-surface-observable5-latent-teacher/wandb/run-20261114_230217-dt54sa0h/files/checkpoint_final.pt` (W&B run `dt54sa0h`). It uses a reference-free SuperMimic low-level tracker and a 32-D latent endpoint residual. The actor output is interpolated toward its endpoint over 10 control updates; each update decodes one latent into the 29-joint low-level target. No future motion clip or joint ground-truth is loaded at runtime.

## MuJoCo interaction

Run from the repository root:

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-surface5-latent
```

Left-drag an upper-body surface to apply a point spring (80 N/m, 4 N s/m damping). The selected point moves with its body. Force is capped at 30 N per point and 60 N total, with a 150 N/s slew rate. The viewer currently drives one selected contact at a time; the policy input preserves the trained five-contact token layout and can represent up to five active contacts.

## Input alignment

The high-level actor receives 372 proprioception/latent values and six histories of five 43-value tokens (1662 total). Each contact token is Isaac asset body one-hot, body-local application point, root-relative point, root-frame applied force divided by 30 N, and a visibility bit. Inactive slots are zero and contacts are canonically sorted, matching training.

The frozen SuperMimic release is `e11c_0220000`. The low-level goal is generated from the reset reference and current root state; it is not a prerecorded future trajectory.
