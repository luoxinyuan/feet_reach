# SuperMimic full-body surface 3-contact direct-latent policy

Exported from `outputs/20261117-supermimic-surface3-direct-latent-compliancemix-gpu234/train/wandb/run-20261116_235926-syfbqljg/files/checkpoint_final.pt` (W&B run `syfbqljg`), trained on ComplianceMix-29h-v1. The actor directly predicts the complete 32-D SuperMimic decoder latent as `tanh(actor_output)`. It does not add a motion-reference base latent and does not interpolate actions.

## MuJoCo interaction

Run from the repository root:

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-surface3-direct-latent
```

Left-drag any visible represented body surface. The selected point follows its body and receives an 80 N/m spring with 4 N s/m damping; the cap is 30 N at the point and 60 N total, with 150 N/s slew. The desktop interface supports one active mouse drag at a time. The actor retains the trained three-contact input format.

The force prompt has 6 frame samples at offsets `[0,1,2,4,8,16]`. Each slot contains body one-hot, body-local point, root-relative point, root-frame force normalized by 30 N, and visibility. The frozen low-level release is `e11c_0220000`; it receives current 10-frame proprioception and the direct latent only. No reference-motion dataset is needed at runtime.

The actor input is 363 proprioceptive values plus 774 contact-history values (1137 total). Metadata records the training checkpoint hash, action interface, joint orders, and export parity checks.
