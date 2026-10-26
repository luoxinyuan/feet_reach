"""Check original exported CEER network equivalence with explicit joint inputs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import torch
from active_adaptation.utils.ceer_residual import FrozenCEER
from scripts.export_whole_body_student import build_policy_state, build_student_rollout_from_state

torch.manual_seed(42)
frozen = FrozenCEER(device='cuda:0')
ckpt = torch.load(ROOT/'.cache/ceer/checkpoint_final.pt', map_location='cpu', weights_only=False)
state = build_policy_state(ckpt)
native = build_student_rollout_from_state(state)
native.load_state_dict(state, strict=True)
native.cuda().eval().requires_grad_(False)
with torch.no_grad():
    norm_obs = torch.randn(128,257,device='cuda:0')
    raw_obs = norm_obs*frozen.policy_scale+frozen.policy_loc
    predicted_joint = native.adapt_joint(norm_obs)
    raw_joint = predicted_joint*frozen.joint_scale+frozen.joint_loc
    output = frozen(raw_obs,raw_joint)
    prediction_error = (output-native(norm_obs)).abs().max().item()
    torch.testing.assert_close(output,native(norm_obs),rtol=1e-5,atol=1e-5)
    norm_joint = torch.randn(128,145,device='cuda:0')
    raw_joint = norm_joint*frozen.joint_scale+frozen.joint_loc
    expected = native.actor(torch.cat((norm_obs,native.adapt(norm_obs),norm_joint),-1))
    actual = frozen(raw_obs,raw_joint)
    direct_error = (actual-expected).abs().max().item()
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
    single = frozen(raw_obs[:1],raw_joint[:1])
    torch.testing.assert_close(single,actual[:1],rtol=1e-5,atol=1e-5)
assert not any('adapt_joint' in name for name,_ in frozen.named_modules())
assert all(not p.requires_grad for p in frozen.parameters())
report = dict(checkpoint_sha256=frozen.checkpoint_sha256, prediction_equivalence_error=prediction_error,
              direct_reference_error=direct_error, batch_equivalence_error=(single-actual[:1]).abs().max().item())
(ROOT/'artifacts/ceer_residual/native.json').write_text(json.dumps(report,indent=2))
print('CEER_NATIVE_PASS',report)
