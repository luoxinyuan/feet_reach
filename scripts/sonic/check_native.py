"""Verify actual released weights: independent batch inference and ONNX parity."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import time
import numpy as np
import torch
from active_adaptation.utils.sonic import FrozenSonic, motor_parameters, ResidualReference, ProprioceptionHistory

def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = FrozenSonic(device='cuda:0')
    q = motor_parameters()[0].cuda().expand(16, -1) + .1 * torch.randn(16, 29, device='cuda')
    ori = torch.tensor([1., 0, 0, 1, 0, 0], device='cuda').expand(16, -1)
    prop = torch.randn(16, 930, device='cuda') * .1
    prop[:, -30:] = torch.tensor([0., 0, -1], device='cuda').repeat(10)
    batch = model(q, ori, prop)
    single = torch.cat([model(q[i:i+1], ori[i:i+1], prop[i:i+1]) for i in range(16)])
    torch.testing.assert_close(batch, single, atol=3e-5, rtol=3e-5)
    model.train()
    assert not model.training and not model.model.training
    assert all(not p.requires_grad for p in model.parameters())
    # Compare the public official ONNX export of this original checkpoint.
    import onnxruntime as ort
    cache = ROOT / '.cache/sonic'
    sessions = [ort.InferenceSession(str(cache / name), providers=['CPUExecutionProvider'])
                for name in ('model_encoder.onnx', 'model_decoder.onnx')]
    errors = []
    for i in range(8):
        inp = np.zeros((1, 1762), dtype=np.float32)
        inp[:, 0] = 0  # C++ deployment mode index; native API uses one-hot.
        inp[:, 4:294] = q[i].cpu().numpy()[None].repeat(10, axis=0).reshape(1, -1)
        inp[:, 601:661] = ori[i].cpu().numpy()[None].repeat(10, axis=0).reshape(1, -1)
        token = sessions[0].run(None, {sessions[0].get_inputs()[0].name: inp})[0]
        decoder = np.concatenate((token, prop[i:i+1].cpu().numpy()), axis=-1)
        result = sessions[1].run(None, {sessions[1].get_inputs()[0].name: decoder})[0]
        errors.append(float(np.max(np.abs(result - batch[i:i+1].cpu().numpy()))))
    assert max(errors) < 1e-4, errors
    limits = torch.ones(4, 29, device='cuda')
    ref = ResidualReference(4, -limits, limits)
    actual = torch.zeros_like(limits)
    ids = torch.arange(4, device='cuda')
    ref.reset(ids, actual)
    ref.update(torch.ones_like(limits), actual)
    before = ref.q.clone()
    ref.reset(ids[:1], actual)
    torch.testing.assert_close(ref.q[1:], before[1:])
    before = ref.q.clone()
    ref.update(torch.zeros_like(limits), actual + .9)
    torch.testing.assert_close(ref.q, before)
    assert (ref.q.abs() <= 1).all()
    hist = ProprioceptionHistory(4, 'cuda')
    args = [torch.randn(4, d, device='cuda') for d in (3, 29, 29, 29, 3)]
    hist.push(*args)
    previous = [v.clone() for v in hist.values]
    hist.reset(ids[:1])
    for a, b in zip(hist.values, previous):
        torch.testing.assert_close(a[1:], b[1:])
    bench = {}
    for n in (1, 256, 2048, 4096):
        qb = q[:1].expand(n, -1); ob = ori[:1].expand(n, -1); pb = prop[:1].expand(n, -1)
        for _ in range(3): model(qb, ob, pb)
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(20): model(qb, ob, pb)
        torch.cuda.synchronize()
        bench[n] = (time.perf_counter() - start) * 1000 / 20
    report = {'batch_vs_single_max_abs': (batch-single).abs().max().item(),
              'onnx_max_abs': max(errors), 'native_forward_ms': bench,
              'frozen': True, 'reset_isolation': True, 'zero_action_holds_reference': True}
    out = ROOT / 'artifacts/sonic_residual'; out.mkdir(parents=True, exist_ok=True)
    (out / 'native.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
