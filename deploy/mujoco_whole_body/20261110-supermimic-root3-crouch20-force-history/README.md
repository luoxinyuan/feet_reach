# SuperMimic root3 force-history policy

Standalone interactive MuJoCo export of the final `supermimic-root3-crouch20-force-history` training checkpoint.
The bundle contains the trained 3KP residual actor and a TorchScript copy of the frozen released SuperMimic
low-level tracker. Runtime loading does not require Isaac Lab, the training checkpoint, the motion datasets,
or the external SuperMimic source checkout.

Run the browser demo:

```bash
python scripts/mujoco_whole_body_web.py --policy supermimic-root3-crouch20-force-history --port 8790
```

Or use the desktop viewer:

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-root3-crouch20-force-history
```

The policy input is 354 proprioceptive/reference values followed by root-local force history for pelvis,
left hand EE and right hand EE. Force-history samples use offsets `[0, 1, 2, 4, 8, 16]` at 50 Hz and
are divided by 30 N. Missing history at reset is zero-filled. Each application point is capped at 30 N,
with a 60 N total force budget and 150 N/s per-point slew limit.

The interactive demo has no motion clip. It gives SuperMimic a 10-frame stationary, upright standing goal
and adds the residual actor's persistent joint-reference offset. This preserves the trained force-to-joint
residual interface without loading future motion targets. As a headless smoke check, the nominal 20 s run
stayed finite with no fall (minimum root height 0.757 m); a 15 N hand-force run also stayed finite with no
fall and the force-token audit error was 0. These are simulator load checks, not a replacement for the
IsaacLab compliance benchmark.
The measured headless results and web-render endpoint checks are in `validation.json`.

`metadata.json` records the training checkpoint hash, the released low-level model revision and hash,
observation order, joint mappings and force settings. `SHA256SUMS.json` covers the exported model and
metadata files.
