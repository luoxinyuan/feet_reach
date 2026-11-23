"""Run and summarize the fixed spatial-masking ablation across physical GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dense_next import paired_ci
from masked_motion import save_json


POLICIES = ("whole", "causal")
MODES = ("direct", "masked", "pretrain")
SEEDS = (20261122, 20261123)


def run(root, gpus, epochs, batch, clip_limit):
    script = Path(__file__).with_name("masked_motion.py")
    cache = root / "cache.pt"
    torch_env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

    def execute(gpu, args, log):
        env = dict(os.environ, **torch_env, CUDA_VISIBLE_DEVICES=str(gpu))
        with log.open("a") as handle:
            subprocess.run([sys.executable, str(script), *map(str, args)], env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    jobs = [(policy, mode, seed) for policy in POLICIES for mode in MODES for seed in SEEDS]
    jobs.extend(("causal", "history", seed) for seed in SEEDS)
    save_json(root / "run.json", {"gpus": gpus, "epochs": epochs, "batch": batch, "clip_limit": clip_limit,
                                 "jobs": [dict(policy=p, mode=m, seed=s) for p, m, s in jobs]})
    if not cache.exists():
        execute(gpus[0], ["prepare", "--cache", cache, "--clip-limit", clip_limit], root / "prepare.log")

    def worker(gpu, assigned):
        failures = []
        for policy, mode, seed in assigned:
            name = f"{policy}-{mode}-{seed}"
            output = root / name
            if (output / "summary.json").exists():
                continue
            for attempt in range(2):
                if output.exists():
                    output.rename(root / f"{name}-incomplete-{time.time_ns()}")
                try:
                    print("start", name, "physical GPU", gpu, flush=True)
                    execute(gpu, ["train", "--cache", cache, "--output", output,
                                  "--policy", policy, "--mode", mode, "--seed", seed,
                                  "--epochs", epochs, "--batch", batch], root / f"{name}.log")
                    print("complete", name, flush=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt == 1:
                        failures.append(name)
                    print("failed", name, "attempt", attempt + 1, flush=True)
        return failures

    with ThreadPoolExecutor(len(gpus)) as workers:
        futures = [workers.submit(worker, gpu, jobs[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        failures = [name for future in futures for name in future.result()]
    if failures:
        raise RuntimeError(f"Jobs failed after retry: {failures}")
    aggregate(root)


def aggregate(root):
    metrics = ("body_fk_cm", "body_fk_first_100ms_cm", "contact_first_100ms_cm", "root_first_100ms_cm",
               "wrist_fk_cm", "foot_fk_cm")
    conditions = ("paired", "shuffled", "absent", "zero", "reversed", "noise_0p1")
    common = None
    report = {}
    for policy in POLICIES:
        report[policy] = {}
        for n in (3, 5):
            all_errors = {}
            record = {}
            for mode in (*MODES, "history"):
                task_policy = "causal" if mode == "history" else policy
                dirs = [root / f"{task_policy}-{mode}-{seed}" for seed in SEEDS]
                summaries = [json.loads((d / "summary.json").read_text())["test"][str(n)] for d in dirs]
                rows = []
                for d in dirs:
                    with np.load(d / "per_window.npz") as vals:
                        candidate = (vals["test_ids"], vals["recordings"], vals["categories"])
                        if common is None:
                            common = candidate
                        else:
                            assert all(np.array_equal(x, y) for x, y in zip(common, candidate))
                        rows.append({
                            (case, metric): vals[f"{n}__{case}__{metric}"]
                            for case in conditions for metric in metrics
                        })
                errors = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
                assert all(np.isfinite(v).all() for v in errors.values())
                all_errors[mode] = errors
                record[mode] = {
                    "conditions": {case: {metric: float(errors[case, metric].mean()) for metric in metrics}
                                   for case in conditions},
                    "direction_response_cm_per_seed": [s["reversal_projected_cm"] for s in summaries],
                    "mean_observed_force_N_per_seed": [s["force_total_N_mean_including_masked_frames"] for s in summaries],
                    "seeds": list(SEEDS),
                    "paired_fk_cm_per_seed": errors["paired", "body_fk_cm"].mean(1).tolist(),
                }
            _, recordings, categories = common
            for mode in MODES:
                errors = all_errors[mode]
                record[mode]["paired_improvement_cm"] = {}
                for comparator in ("direct", "history", "shuffled"):
                    if comparator == mode:
                        continue
                    comparison = all_errors[comparator] if comparator != "shuffled" else errors
                    case = "shuffled" if comparator == "shuffled" else "paired"
                    record[mode]["paired_improvement_cm"][comparator] = {
                        metric: {"mean": float((comparison[case, metric] - errors["paired", metric]).mean()),
                                 "difference_by_seed": (comparison[case, metric] - errors["paired", metric]).mean(1).tolist(),
                                 "recording_bootstrap_ci95": paired_ci(errors["paired", metric].mean(0),
                                     comparison[case, metric].mean(0), recordings, SEEDS[0])}
                        for metric in metrics
                    }
                record[mode]["paired_by_category"] = {
                    cat: {metric: float(errors["paired", metric][:, categories == cat].mean()) for metric in metrics}
                    for cat in sorted(set(categories))
                }
            report[policy][str(n)] = record
    save_json(root / "results.json", {"test_windows": len(common[0]), "test_recordings": len(set(common[1])),
                                     "units": "centimeters", "results": report})
    print("Wrote", root / "results.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--clip-limit", type=int, default=900)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("Physical GPU indices must be distinct")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run(root, args.gpus, args.epochs, args.batch, args.clip_limit)


if __name__ == "__main__":
    main()
