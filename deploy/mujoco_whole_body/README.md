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
- `validation_residual.json`: export, adapter, headless and browser test results.

Run `python scripts/mujoco_whole_body_web.py --policy sonic` or `--policy ceer`.
The original `--policy direct` remains the default. All three share the bundled G1
meshes. Residual student and CEER weights are ordinary Git files, without Git LFS.
SONIC's two official ONNX files (about 88 MB) are fetched once into
`.cache/mujoco-sonic/`; no Hugging Face login is needed. For offline use, copy
these two files into that cache or `deploy/mujoco_whole_body/sonic/` beforehand;
the loader still verifies their hashes.

SONIC resources are unmodified exports from
[nvidia/GEAR-SONIC, revision bfd764a](https://huggingface.co/nvidia/GEAR-SONIC/tree/bfd764a),
validated against the native controller used in training.
**Licensed by NVIDIA Corporation under the NVIDIA Open Model License.**
The original dual-license notice is included in [sonic/SONIC_LICENSE.txt](sonic/SONIC_LICENSE.txt).
No NVIDIA source library is imported by the standalone runtime. CEER and both
residual students were exported from this project's trained checkpoints; hashes
and original run identifiers are recorded in each variant's metadata.

See [residual instructions and limitations](../../scripts/references/README_mujoco_residual.md).
