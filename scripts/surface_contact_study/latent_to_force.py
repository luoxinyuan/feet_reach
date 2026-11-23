"""Decode fitted motion latents into dense and budgeted sparse link forces."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from active_adaptation.utils.body_guidance import limit_force
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, TARGET_SCALE, fit, metrics


def motion_model(width, outputs):
    return nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, 256),
                         nn.SiLU(), nn.Linear(256, outputs)).cuda()


def fit_codes(decoder, state, target, force_width, seed, dim=32):
    torch.manual_seed(seed + 1000)
    codes = []
    for pose, motion in zip(state.split(2048), target.split(2048)):
        code = nn.Parameter(.01 * torch.randn(len(pose), dim, device=pose.device))
        opt = torch.optim.Adam([code], lr=.02)
        for _ in range(200):
            pred = decoder(torch.cat((pose, nn.functional.pad(code, (0, force_width - dim))), 1))
            loss = (pred - motion).square().mean() + .001 * code.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        codes.append(code.detach())
    return torch.cat(codes)


def top_three(force, scores=None):
    if scores is None:
        scores = force.norm(dim=-1)
    chosen = scores.topk(3, dim=-1).indices
    hard = torch.zeros_like(scores).scatter_(1, chosen, 1.)
    if torch.is_grad_enabled():
        soft = (3. * scores.softmax(dim=-1)).clamp(max=1.)
        hard = hard + (soft - soft.detach())
    return limit_force(force * hard.unsqueeze(-1))


def train_force(model, loss_fn, train, val, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    best, weights = float("inf"), None
    for _ in range(epochs):
        order = torch.randperm(len(train), device=train.device)
        for indices in train[order].split(1024):
            loss = loss_fn(model, indices)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            score = loss_fn(model, val).item()
        if score < best:
            best = score
            weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(11, 22, 33))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root, seed = args.input, args.seed
    out = root / "latent_to_force" / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(root / "data.npz") as data:
        split, clip = data["split"], data["clip"]
        target = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
        pose = data["state"]
        width = data["greedy3"].shape[1]
    with np.load(root / "link_spring" / "features.npz") as data:
        force = data["applied"]
    state_mean = pose[split == 0].mean(0)
    state_scale = np.maximum(pose[split == 0].std(0), .01)
    cuda = lambda x: torch.as_tensor(x, device="cuda")
    state = cuda((pose - state_mean) / state_scale)
    motion = cuda(target / TARGET_SCALE)
    force = cuda(force)
    train, val, test = [cuda(np.flatnonzero(split == s)) for s in (0, 1, 2)]
    index = cuda(np.arange(len(split)))
    previous = (index - 1).clamp_min(0)
    adjacent = cuda(np.r_[False, clip[1:] == clip[:-1]])
    flat = force.flatten(1)
    mean = flat[train].mean(0)
    scale = flat[train].std(0, correction=0).clamp_min(.1)
    padding = width - flat.shape[1]

    def features(indices, f):
        channels = (f.flatten(1) - mean) / scale
        return torch.cat((state[indices], nn.functional.pad(channels, (0, padding))), 1)

    dec = motion_model(state.shape[1] + width, motion.shape[1])
    dec.load_state_dict(torch.load(root / "latent_oracle" / "dim32" / f"decoder_{seed}.pt",
                                   weights_only=True))
    dec.eval().requires_grad_(False)
    codes = torch.empty((len(state), 32), device=state.device)
    for indices in (train, val, test):
        codes[indices] = fit_codes(dec, state[indices], motion[indices], width, seed)
    code_mean = codes[train].mean(0)
    code_scale = codes[train].std(0, correction=0).clamp_min(.01)
    prompts = torch.cat((state, (codes - code_mean) / code_scale), 1)
    print(seed, "latents fitted", flush=True)

    def scorer(scorer_seed):
        net = motion_model(state.shape[1] + width, motion.shape[1])
        ckpt = torch.load(root / "link_spring" / f"applied_{scorer_seed}.pt", weights_only=False)
        net.load_state_dict(ckpt["weights"])
        return net.eval().requires_grad_(False)

    proxy = scorer(seed)
    evaluator = scorer({11: 22, 22: 33, 33: 11}[seed])
    force_shape = tuple(force.shape[1:])
    torch.manual_seed(seed)
    dense = nn.Sequential(nn.Linear(prompts.shape[1], 256), nn.SiLU(),
                          nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, flat.shape[1])).cuda()

    def dense_prediction(net, i):
        return limit_force(net(prompts[i]).reshape(-1, *force_shape))

    def dense_loss(net, i):
        return (dense_prediction(net, i) - force[i]).square().mean()

    train_force(dense, dense_loss, train, val, 65)
    print(seed, "dense mapper fitted", flush=True)
    sparse = nn.Sequential(nn.Linear(prompts.shape[1], 256), nn.SiLU(),
                           nn.Linear(256, 256), nn.SiLU(),
                           nn.Linear(256, flat.shape[1] + force_shape[0])).cuda()
    with torch.no_grad():
        for layer in (0, 2):
            sparse[layer].load_state_dict(dense[layer].state_dict())
        sparse[-1].weight[:flat.shape[1]].copy_(dense[-1].weight)
        sparse[-1].bias[:flat.shape[1]].copy_(dense[-1].bias)
        sparse[-1].weight[flat.shape[1]:].zero_()
        sparse[-1].bias[flat.shape[1]:].zero_()

    def sparse_prediction(net, i):
        values, logits = net(prompts[i]).split((flat.shape[1], force_shape[0]), 1)
        values = values.reshape(-1, *force_shape)
        return top_three(values, logits + values.norm(dim=-1))

    with torch.no_grad():
        truncated = top_three(force)

    def sparse_loss(net, i):
        f = sparse_prediction(net, i)
        mse = (proxy(features(i, f)) - motion[i]).square().mean()
        prev_f = sparse_prediction(net, previous[i])
        change = (f - prev_f).square() * adjacent[i, None, None]
        return (mse + .05 * (f - truncated[i]).square().mean() +
                .02 * change.mean() + .0005 * f.norm(dim=-1).sum(dim=-1).mean())

    train_force(sparse, sparse_loss, train, val, 65)
    print(seed, "sparse mapper fitted", flush=True)
    report = {}
    with torch.no_grad():
        dense_all = dense_prediction(dense, index)
        sparse_all = sparse_prediction(sparse, index)
        methods = {"original": force, "original_top3": truncated, "decoded_dense": dense_all,
                   "decoded_dense_top3": top_three(dense_all), "decoded_sparse": sparse_all}
        for name, f in methods.items():
            norms = f[test].norm(dim=-1)
            contiguous = test[adjacent[test]]
            rate = (f[contiguous] - f[previous[contiguous]]).norm(dim=-1).amax(-1) / .02
            pred = evaluator(features(test, f[test])).cpu().numpy() * TARGET_SCALE
            report[name] = dict(
                frozen_scores={k: metrics(target[split == 2, sl], pred[:, sl])
                               for k, sl in GROUPS.items()},
                forces=dict(mean_total=norms.sum(-1).mean().item(),
                            p95_total=norms.sum(-1).quantile(.95).item(),
                            max_total=norms.sum(-1).max().item(),
                            max_point=norms.max().item(),
                            mean_active=(norms > .05).sum(-1).float().mean().item(),
                            max_active=(norms > 1e-5).sum(-1).max().item(),
                            step_rate_p95=rate.quantile(.95).item()),
                force_nrmse=metrics(force[test].cpu().numpy(), f[test].cpu().numpy())["nrmse"],
            )
    for name, f in methods.items():
        if name == "original":
            continue
        # Fit a fresh readout to each force distribution to measure information
        # separately from domain shift in the frozen dense readout.
        x = features(index, f).cpu().numpy()
        pred, _, _ = fit(x, target, split, seed, out / f"{name}_readout.pt")
        report[name]["matched_scores"] = {
            k: metrics(target[split == 2, sl], pred[:, sl]) for k, sl in GROUPS.items()
        }
    torch.save(dict(dense=dense.state_dict(), sparse=sparse.state_dict(),
                    code_mean=code_mean.cpu(), code_scale=code_scale.cpu(),
                    state_mean=state_mean.tolist(), state_scale=state_scale.tolist()), out / "decoders.pt")
    np.savez_compressed(out / "test_forces.npz", rows=test.cpu().numpy(),
                        dense=dense_all[test].cpu().numpy(), sparse=sparse_all[test].cpu().numpy())
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
