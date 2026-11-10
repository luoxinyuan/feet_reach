# Recent compliance policy candidates

Candidates identified from the recent training runs. Checkpoints marked active were still being trained when this list was assembled; their weights may change as training continues.

| Policy | Training status | Checkpoint / artifact | Notes |
|---|---|---|---|
| Root3 Crouch20, force-history teacher | Complete | [`checkpoint_final.pt`](../outputs/20261103-ceer-online-root3-crouch20-force-history/train/wandb/run-20261103_134204-aej91ijk/files/checkpoint_final.pt) | Online estimated-force student trained with force-history teacher. |
| Root3 Crouch20, torque-only force-history teacher | Active at listing time | [`checkpoint_550.pt`](../outputs/20261104-ceer-online-root3-crouch20-force-history-torque-only/train/wandb/run-20261103_233733-9uqhdhgr/files/checkpoint_550.pt) | Intermediate checkpoint from the torque-only run. |
| Root3 Crouch20, no force-history teacher | Complete; evaluated | [`checkpoint_final.pt`](../outputs/20261103-ceer-3point-residual-teacher-crouch20/teacher/wandb/run-20261103_012239-kx7wbhsh/files/checkpoint_final.pt) | Three-point residual teacher. [Compliance v1.1 report](../artifacts/compliance_eval/20261105-root3-crouch20-no-force-history-corrected-v1_1/REPORT.md) |
| Root3 CrouchTracking, 3-point teacher | Teacher complete; downstream pipeline complete | [`checkpoint_final.pt`](../outputs/20261103-ceer-3point-force-deltaq/full/teacher/wandb/run-20261103_011903-szrvkqj6/files/checkpoint_final.pt) | A separate residual-teacher artifact exists; the compliance runner targets this saved policy checkpoint. |
| Root3 low-level force, GT-joint-target fine-tuning | Complete | [`checkpoint_final.pt`](../outputs/20261103-ceer-lowlevel-force-root3-supervised-residual/resume_from_checkpoint_13_16384/lowlevel_finetune/wandb/run-20261103_154204-dque1v6y/files/checkpoint_final.pt) | Low-level force fine-tuning run. |
| Root3 low-level force, teacher fine-tuning | Teacher complete; downstream rollout active at listing time | [`checkpoint_final.pt`](../outputs/20261104-ceer-lowlevel-force-root3-teacher-supervised-residual/full/teacher_finetune/wandb/run-20261103_194714-uyl5nq45/files/checkpoint_final.pt) | Teacher checkpoint; downstream teacher rollout was still running. |
| Root3 no-EE crouch teacher | Complete | [`checkpoint_final.pt`](../outputs/20261102-ceer-online-root3-no-ee-crouch-teacher/140134/wandb/run-20261102_140254-sbjc6hly/files/checkpoint_final.pt) | Earlier baseline, not the Crouch20 run. |
| MuJoCo exported online-estimated 9-point policy | Exported; full compliance eval complete | [`20261101-ceer-online-estimated-9pt`](../deploy/mujoco_whole_body/20261101-ceer-online-estimated-9pt/) | Source Isaac checkpoint is the `qaxgsgg1` 2026-10-30 finetune run. [Compliance v1.1 report](../artifacts/compliance_eval/20261104-ceer-online-estimated-9pt-compliance-v1_1/REPORT.md); [12 s MuJoCo force-test report](../artifacts/compliance_eval/20261104-ceer-online-estimated-9pt-mujoco/report.json) |
| Segmented Jacobian multi10 (MuJoCo exported) | Exported; full compliance v1.1 eval complete | [`20261026-segmented-multi10`](../deploy/mujoco_whole_body/20261026-segmented-multi10/) | Source Isaac checkpoint SHA256 matched the MuJoCo export record. [Full report](../artifacts/compliance_eval/20261104-segmented-multi10-compliance-v1_1/REPORT.md) |
| CEER EE-root3 segmented (MuJoCo exported) | Exported; full compliance v1.1 eval complete | [`20261028-ceer-ee-root3`](../deploy/mujoco_whole_body/20261028-ceer-ee-root3/) | Source Isaac checkpoint SHA256 matched the MuJoCo export record. [Full report](../artifacts/compliance_eval/20261104-ee-root3-compliance-v1_1/REPORT.md) |
| CEER spring-root3 (MuJoCo exported) | Exported; full compliance v1.1 eval complete | [`20261029-ceer-spring-root3`](../deploy/mujoco_whole_body/20261029-ceer-spring-root3/) | Source Isaac checkpoint SHA256 matched the MuJoCo export record. [Full report](../artifacts/compliance_eval/20261104-spring-root3-compliance-v1_1/REPORT.md) |

The Crouch20 no-force-history teacher and exported MuJoCo 9-point policy were evaluated earlier. The three MuJoCo bundles above were the latest selected candidates.

## Evaluation results

- **Crouch20 no-force-history teacher, compliance v1.1:** 384 validation tracking trials; 96.6% physically completed, mean joint MAE 0.280 rad, mean root XY error 0.098 m, and mean root-relative EE error 0.205 m. The crouch category completed 78.3% of trials. In fixed-force tests, 20 N locomotion survived 60% of trials; the 20 N hand-force case survived 95%. The [full report](../artifacts/compliance_eval/20261105-root3-crouch20-no-force-history-corrected-v1_1/REPORT.md) includes all categories and paired zero-force results.
- **Force-history ablation conclusion:** both corrected Crouch20 checkpoints completed **96.6%** of validation tracking. No-history had lower joint MAE (**0.280 vs 0.294 rad**); force-history had slightly lower EE-relative error (**0.189 vs 0.205 m**) and root XY error (**0.089 vs 0.098 m**). Current v1.1 results do **not** show an overall tracking advantage from history. See [no-history](../artifacts/compliance_eval/20261105-root3-crouch20-no-force-history-corrected-v1_1/REPORT.md) and [force-history](../artifacts/compliance_eval/20261105-root3-crouch20-force-history-corrected-v1_1/REPORT.md).
- **Exported 9-point policy, corrected Isaac compliance v1.1:** source checkpoint SHA256 matches the MuJoCo export record. Across 384 validation tracking trials, physical completion was **96.9%**, mean joint MAE 0.224 rad, mean root XY error 0.099 m, and mean root-relative EE error 0.176 m. The prior low-survival Isaac report was invalidated because the actor checkpoint was not loaded. See the [corrected report](../artifacts/compliance_eval/20261105-9pt-no-ee-corrected-v1_1/REPORT.md).
- **Segmented Jacobian multi10:** 96.9% validation physical completion; joint MAE 0.252 rad. See the [full v1.1 report](../artifacts/compliance_eval/20261104-segmented-multi10-compliance-v1_1/REPORT.md).
- **CEER EE-root3 segmented:** 96.4% validation physical completion; joint MAE 0.250 rad. See the [full v1.1 report](../artifacts/compliance_eval/20261104-ee-root3-compliance-v1_1/REPORT.md).
- **CEER spring-root3:** 96.9% validation physical completion; joint MAE 0.212 rad. See the [full v1.1 report](../artifacts/compliance_eval/20261104-spring-root3-compliance-v1_1/REPORT.md).


## Evaluator correction (2026-11-05)

Several recent DenseForcePPO reports had a checkpoint-loader regression and are invalidated in place. The wrapper stripped a legitimate `TensorDictModule.module` prefix and silently accepted the missing actor weights. Corrected reports use actor key/shape validation; details and the affected report list are in the [loader regression audit](../artifacts/compliance_eval/20261105-loader-regression-audit.md).

- Filterdata force-history: **96.6%** validation completion, **0.290 rad** joint MAE ([corrected report](../artifacts/compliance_eval/20261105-filterdata-force-history-corrected-v1_1/REPORT.md)).
- 3KP spring force-history, filtered step 300: **95.1%**, **0.312 rad** ([corrected report](../artifacts/compliance_eval/20261105-3kp-spring-force-history-filtered-step300-corrected-v1_1/REPORT.md)).
- Torque-only force-history: **96.4%**, **0.298 rad** ([corrected report](../artifacts/compliance_eval/20261105-root3-crouch20-torque-only-corrected-v1_1/REPORT.md)).
- Independent pelvis: **96.1%**, **0.291 rad** ([corrected report](../artifacts/compliance_eval/20261105-independent-pelvis-corrected-v1_1/REPORT.md)).
- Online estimated 9-point: **96.9%**, **0.224 rad** ([corrected report](../artifacts/compliance_eval/20261105-9pt-no-ee-corrected-v1_1/REPORT.md)).
