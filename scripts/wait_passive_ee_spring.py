"""Launch one spring experiment after either four-GPU group is released."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/passive-ee-spring'
GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7)]


def gpu_snapshot():
    def query(arg):
        return list(csv.reader(subprocess.check_output(
            ['nvidia-smi', arg, '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()))
    rows = query('--query-gpu=index,uuid,memory.used,utilization.gpu')
    active = {r[0].strip() for r in query('--query-compute-apps=gpu_uuid') if r}
    return {int(i): uuid.strip() not in active and int(mem) < 1024 and int(util) < 10
            for i, uuid, mem, util in rows}


def pipeline_reservations():
    # A live pipeline reserves its GPUs even between training/evaluation phases.
    reserved = set()
    for path in (ROOT / 'outputs').glob('passive-*/launcher.json'):
        if path.parent == OUT:
            continue
        meta = json.loads(path.read_text())
        pid = meta.get('pid')
        if not pid:
            continue
        proc = Path('/proc') / str(pid)
        try:
            cmd = (proc / 'cmdline').read_bytes()
            if b'train_passive_' in cmd:
                reserved.update(meta.get('gpus', []))
        except FileNotFoundError:
            pass
    return reserved


def available_groups(snapshot, reserved):
    return [g for g in GROUPS if all(snapshot.get(i, False) and i not in reserved for i in g)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'wait.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = OUT / 'queue.json'
        if state.exists() and json.loads(state.read_text()).get('status') in ('launched', 'finished', 'failed'):
            print('This queue has already launched; refusing duplicate launch.', flush=True)
            return
        counts = {g: 0 for g in GROUPS}
        while True:
            try:
                ready = available_groups(gpu_snapshot(), pipeline_reservations())
                counts = {g: counts[g] + 1 if g in ready else 0 for g in GROUPS}
            except Exception as error:
                print(f'GPU check failed; keep waiting: {error}', flush=True)
                counts = {g: 0 for g in GROUPS}
            record = dict(status='waiting', waiter_pid=os.getpid(), checked_at=time.time(),
                          idle_checks={','.join(map(str,g)): n for g,n in counts.items()})
            state.write_text(json.dumps(record, indent=2) + '\n')
            selected = next((g for g in GROUPS if counts[g] >= 3), None)
            if selected is not None:
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, selected)), NUM_ENVS='12288',
                           TEACHER_FRAMES='1200000000', ADAPT_FRAMES='100000000', STUDENT_FRAMES='400000000')
                for key in ('RESUME_CHECKPOINT', 'TEACHER_DONE_FRAMES'):
                    env.pop(key, None)
                with (OUT / 'launch.log').open('a') as log:
                    child = subprocess.Popen(['bash', 'scripts/train_passive_ee_spring_pipeline.sh'],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record.update(status='launched', pid=child.pid, gpus=list(selected), started_at=time.time(),
                              num_envs_per_gpu=12288, teacher_force_enabled=True)
                state.write_text(json.dumps(record, indent=2) + '\n')
                (OUT / 'launcher.json').write_text(json.dumps(record, indent=2) + '\n')
                print(f'Launched PID {child.pid} on GPUs {selected}', flush=True)
                result = child.wait()
                record.update(status='finished' if result == 0 else 'failed', returncode=result)
                state.write_text(json.dumps(record, indent=2) + '\n')
                return
            print(f'Waiting: {record["idle_checks"]}', flush=True)
            time.sleep(30)


if __name__ == '__main__':
    main()
