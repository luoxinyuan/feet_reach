# Bundled whole-body compliance student

This directory contains everything needed by the standalone MuJoCo demo:

- `policy/student.pt`: deterministic TorchScript student (approximately 5 MB).
- `policy/normalization.pt`: frozen public-observation normalization.
- `policy/metadata.json`: joint order, gains, actuator limits and checkpoint hash.
- `g1/g1_29dof_rev_1_0.xml` and `g1/meshes/`: G1 MuJoCo model and all 35 referenced meshes.
- `SHA256SUMS.json`: SHA-256 manifest of the model and policy files.

These files are ordinary Git files. Git LFS, training checkpoints, Isaac Lab,
AMASS data and the external `ceer_deploy` checkout are not required to run.
Total policy/model size is approximately 25 MB before Git compression.

See [installation and controls](../../scripts/references/README_mujoco_whole_body_student.md).

## Provenance

The student was exported from the final finetune checkpoint of
`outputs/passive-whole-body/20261021_231413`, run `offline-run-20261022_033351-b0ak7xn3`.
The checkpoint SHA-256 is recorded in `policy/metadata.json`. The full checkpoint
is not distributed here. No teacher or reference-motion data is loaded at runtime.
The exporter reconstructs the student architecture using code adapted from
`ceer_deploy/scripts/export_policy_vecnorm_from_checkpoint.py`.

The robot XML and meshes were copied unchanged from
[luoxinyuan/ceer_deploy](https://github.com/luoxinyuan/ceer_deploy), revision
`2c12d143c6dc2a34445f952ff4a21e09e5bbc455`, under `assets/robots/g1/`.
G1 is a Unitree robot. The source repository credits HansZ8 and supplies
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its license notice is
preserved in [CEER_DEPLOY_LICENSE.txt](CEER_DEPLOY_LICENSE.txt).
The simulation script adds the training hand force points in memory and overrides
actuator settings; it does not rewrite the bundled source XML or meshes.

## Residual variants

- `sonic/`: trained residual student (381 public inputs), frozen normalization,
  motor/interface metadata, pinned official ONNX download URLs and SHA-256 hashes.
- `ceer/`: trained residual student, normalization, frozen original CEER controller
  with its joint predictor bypassed, and exported kinematics/motor metadata.
- `jacobian/`: the Jacobian compliance residual student and the exact frozen CEER
  low-level controller used by its training checkpoint, with its own normalization
  and kinematics/motor metadata.
- `20261029-ceer-jacobian-no-ee/`: the completed 9-point online-Jacobian residual
  with the separate EE-tracking reward disabled; kept alongside the older Jacobian bundle.
- `segmented-static-hold/`: the static-hold augmented segmented policy evaluated
  on 2026-10-25 (W&B run `3dlvh2d9`), kept separately from `segmented/`.
- `validation_residual.json`: export, adapter, headless and browser test results.

Run `python scripts/mujoco_whole_body_web.py --policy sonic`, `--policy ceer`,
`--policy jacobian`, or `--policy jacobian-no-ee`. The original `--policy direct` remains the default. All policies
share the bundled G1 meshes. Residual student and CEER weights are ordinary Git
files, without Git LFS.
SONIC's two official ONNX files (about 88 MB) are fetched once into
`.cache/mujoco-sonic/`; no Hugging Face login is needed. For offline use, copy
these two files into that cache or `deploy/mujoco_whole_body/sonic/` beforehand;
the loader still verifies their hashes.

SONIC resources are unmodified exports from
[nvidia/GEAR-SONIC, revision bfd764a](https://huggingface.co/nvidia/GEAR-SONIC/tree/bfd764a),
validated against the native controller used in training.
**Licensed by NVIDIA Corporation under the NVIDIA Open Model License.**
The original dual-license notice is included in [sonic/SONIC_LICENSE.txt](sonic/SONIC_LICENSE.txt).
No NVIDIA source library is imported by the standalone runtime. CEER and all
residual students were exported from this project's trained checkpoints; hashes
and original run identifiers are recorded in each variant's metadata.

See [residual instructions and limitations](../../scripts/references/README_mujoco_residual.md).

## Segmented Jacobian variant

`segmented/` contains the completed 18-point CEER residual model (run `00lci6cu`),
its frozen CEER controller, normalization and application-point metadata. This is
separate from the AMP experiment. Launch with `--policy segmented` in either
interactive entry point; see the [instructions and measured limitations](../../scripts/references/README_mujoco_residual.md#分段施力版segmented18-个候选点).
No reference motion, force cache or training dependencies are needed at runtime.

The static-hold augmented checkpoint is available with
`python scripts/mujoco_whole_body_student.py --policy segmented-static-hold`, or
in the browser with
`python scripts/mujoco_whole_body_web.py --policy segmented-static-hold --port 8780`.
Its paired evaluation found fewer large passive-hand lifts overall; see
`artifacts/ceer_segmented_singlehand_stop_eval_20261025/README.md` for metrics
and the side-specific caveat.

`20261026-segmented-multi10/` is a separate export of the multi10 segmented
CEER residual checkpoint (W&B run `4vqahv90`). Select it with
`--policy segmented-multi10`; the package directory keeps the date prefix.
A standalone frontal force-sequence renderer is available at
`scripts/mujoco_whole_body_force_demo.py`.

## Three-point and 23-point compliance students

Two additional interactive bundles are available:

- `20261028-ceer-ee-root3/`: the CEER residual trained with forces at pelvis and both EEs.
- `20261028-ceer-dense23/`: the full-body 23-point residual with its causal force estimator.
- `20261030-ceer-online-estimated-root3-v2/`: corrected MuJoCo adapter for the online-trained 3-point residual and causal force estimator. The original v1 bundle is retained for comparison.

Start either browser demo with:

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-ee-root3 --port 8782
python scripts/mujoco_whole_body_web.py --policy ceer-dense23 --port 8783
python scripts/mujoco_whole_body_web.py --policy ceer-online-estimated-root3 --port 8784
```

The 3-point student receives the three applied forces in the root-local frame.
The 23-point student does not receive MuJoCo force ground truth: its bundled
estimator predicts forces from ten recent proprioceptive/action frames, then
feeds those predictions to the residual actor. The browser HUD shows the
estimator's total force magnitude and the selected point's estimate; orange
arrows show applied forces and a blue arrow shows the selected estimate. Both
use the same 30 N per-point drag limit and work with the desktop viewer as well.
The 20261030 online-estimated root3 student also uses only its bundled causal
estimator at the pelvis and both EEs; it does not feed MuJoCo force ground truth
to the actor. Its default alias selects the v2 bundle, which restores the
frozen CEER low-level action scales and filters from the training checkpoint.
