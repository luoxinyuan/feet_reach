"""Persistent, restartable local watcher for explicitly configured training runs."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def alive(pid,start):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and fields[19]==start
    except (FileNotFoundError,ProcessLookupError):return False

def training_processes(run):
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:cmd=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if 'scripts/train.py' in cmd and str(run) in cmd:return True
    return False

def checkpoint_ready(run):
    found=list((run/'train'/'wandb').glob('run-*/files/checkpoint_final.pt'))
    if len(found)>1:raise RuntimeError(f'Ambiguous final checkpoint: {run}')
    if not found:return None
    ckpt=found[0]
    # Completed launchers imply torch.save has returned; also allow filesystem settling.
    if ckpt.stat().st_size==0 or time.time()-ckpt.stat().st_mtime<30:return None
    return ckpt

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text());folder=a.config.parent
    lock=(folder/'watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=folder/'status.json'
    states=json.loads(status_path.read_text()) if status_path.exists() else {}
    running={}
    def save():
        tmp=folder/'status.tmp';tmp.write_text(json.dumps(states,indent=2));tmp.replace(status_path)
    while True:
        for job in config['jobs']:
            name=job['name'];state=states.setdefault(name,{'status':'waiting_training'})
            if name in running:
                proc,log=running[name];code=proc.poll()
                if code is not None:
                    log.close();del running[name]
                    state.update(status='complete' if code==0 else 'eval_failed',exit_code=code,finished_at=time.time())
                continue
            if state['status'] in ('complete','eval_failed','training_failed'):continue
            if state.get('status')=='evaluating' and alive(state['eval_pid'],state['eval_start']):continue
            run=Path(job['run'])
            if alive(job['launcher_pid'],job['launcher_start']) or training_processes(run):continue
            ckpt=checkpoint_ready(run)
            if ckpt is None:
                first=state.setdefault('training_exit_seen',time.time())
                if time.time()-first>180:state.update(status='training_failed',reason='Training exited without a finalized checkpoint; no partial checkpoint evaluated.')
                continue
            out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(ROOT/'scripts/compliance_eval/run.py'),'--checkpoint',str(ckpt),'--manifest',str(ROOT/'cfg/compliance_eval_v1_5.json'),'--output',str(out),'--gpus',job['gpus'],'--videos']
            log=(out/'auto_eval.log').open('a');env=os.environ.copy();env['MEMPATH']=str(ROOT/'dataset');env['PYTHONUNBUFFERED']='1'
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            running[name]=(proc,log);state.update(status='evaluating',eval_pid=proc.pid,eval_start=start,checkpoint=str(ckpt),output=str(out),command=cmd,started_at=time.time())
        save()
        if all(s['status'] in ('complete','eval_failed','training_failed') for s in states.values()) and len(states)==len(config['jobs']):break
        time.sleep(config.get('poll_seconds',30))
if __name__=='__main__':main()
