"""Write a concise, reproducible Markdown report for v2 root-motion labels."""
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);r=json.loads((out/'evaluation/report.json').read_text());m=json.loads((out/'manifest.json').read_text());v=json.loads((out/'validation.json').read_text())
names=['jacobian3','jacobian9','spring3','force37'];labels={'state_only':'State only','jacobian3':'3KP Jacobian','jacobian9':'9KP Jacobian','spring3':'3KP direct spring','force37':'37-point force prompt'}
groups=['joint_delta','root1_translation_m','root1_rotation_rad','root5_translation_m','root5_rotation_rad']
lines=['# Force–motion correspondence audit v2','', 'Offline nominal-reference-state decoder audit. It is not a policy or physics rollout.', '',
'## Protocol','',
'The matched split contains 560 source clips and 80,123 transitions after the five-frame endpoint crop: 363/84/113 clips and 52,541/11,706/15,876 transitions for train/validation/test.', '',
'The matched source-clip split and candidate force methods are the v1 protocol. A shared 41-output MLP predicts next-frame joint Δq, root-relative translation/rotation over one frame, and root-relative translation/rotation over five frames. Translation is expressed in the current root frame; rotation uses the log map of the relative SO(3) rotation. Labels are scaled by 0.02 in their native units during fitting. Scores are reported by target group in physical units. Each force-conditioned decoder is compared to the same state-only architecture.', '',
'3KP direct spring is replayed at nominal reference states using the production 80 N/m spring, 30 N per-point/60 N total caps, 150 N/s force slew, all three pelvis/hand points continuously active, and reset force zero. Actor-visible applied force is used; desired force is retained as a timing diagnostic. Jacobian force is similarly actor-visible after substep slew. The force37 cache is the existing noncausal 64-frame-block prompt.', '',
'## Results','',
'NRMSE is lower-is-better. Force SSE reduction is `1 - SSE(state+force)/SSE(state)`; positive means force adds predictive value over current state. Confidence intervals bootstrap whole held-out clips. Translation is in meters; rotation and joint errors are in radians.', '',
'|Target|Method|NRMSE ↓|RMSE|MAE|Force SSE reduction vs state only (95% clip CI)|',
'|---|---|---:|---:|---:|---:|']
for group in groups:
    for name in ['state_only']+names:
        s=r['summary'][name][group];ci=s['force_sse_reduction_clip_bootstrap95']
        gain='—' if name=='state_only' else f"{s['force_sse_reduction_vs_state']*100:.1f}% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]%"
        lines.append(f"|{group}|{labels[name]}|{s['nrmse_mean']:.3f} ± {s['nrmse_seed_std']:.3f}|{s['rmse_mean']:.5f}|{s['mae_mean']:.5f}|{gain}|")
lines += ['', '## Force magnitude and implementation checks','', '|Method|Mean sum of point-force norms (N)|P95 (N)|Max (N)|Mean active points|', '|---|---:|---:|---:|---:|']
for name in names:
    s=r['force_stats'][name];lines.append(f"|{labels[name]}|{s['total_mean_N']:.2f}|{s['total_p95_N']:.2f}|{s['total_max_N']:.2f}|{s['active_points_mean']:.2f}|")
lines += ['',f"Validation: {v}.", '', '## Interpretation limits','',
'- This score asks whether current state plus force predicts recorded motion targets. It measures decodability/conditional predictive value, not the causal physical response to applying force or closed-loop policy quality.',
'- The force generators are not a matched ablation: point layouts, objectives, active-contact schedules, regularization, and force magnitudes differ. In particular, the 37-point cache is optimized over 64-frame blocks and sees future motion; the Jacobian and spring prompts are causal per-frame methods.',
'- The source corpus is the policy training corpus, not a subject-disjoint/OOD evaluation. Its AMASS subset was previously filtered using a 3KP method. Decoder test clips are held out by source clip only.',
'- The direct spring uses current-to-next reference displacement at nominal state; it does not include perturbed robot position/velocity, spring damping, or physics contacts. Planned versus applied forces are stored separately.',
'- The five-frame root target has fewer valid transitions near clip ends; all methods use the same truncated frames.',
'- Force SSE reduction is a decoder- and protocol-dependent predictive score, not mutual information.', '',
'## Artifacts','', '- `data.npz`, `manifest.json`: matched frames, force arrays, source clip split and target labels.', '- `evaluation/report.json`, `predictions.npz`, `*.pt`: metrics, predictions and decoder checkpoints.', '- `validation.json`: finite-difference, split, target-shape and force-budget checks.', '- `code/`: exact script and production-code snapshots used for this run.', '']
(out/'RESULTS.md').write_text('\n'.join(lines))
print(out/'RESULTS.md')
