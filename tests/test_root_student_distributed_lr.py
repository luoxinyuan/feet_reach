"""A shared DDP gradient must not produce rank-dependent optimizer updates."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from active_adaptation.learning.hierarchical.root_student_ppo import RootStudentPPOPolicy


def worker(rank, rendezvous, output):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(42)
        model=DDP(torch.nn.Linear(1,1,bias=False))
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
        policy=SimpleNamespace(progress=.5,current_lr=.0003,cfg=SimpleNamespace(desired_kl=.01),
            opt_teacher=optimizer,opt_student=optimizer,opt_estimator=optimizer,opt_critic=optimizer,
            opt_direct_estimator=None,world_size=2)
        RootStudentPPOPolicy._do_lr_schedule(policy,.04 if rank==0 else .001)
        optimizer.zero_grad();model(torch.ones(1,1)).sum().backward();optimizer.step()
        local=torch.tensor([policy.current_lr,model.module.weight.item()])
        values=[torch.zeros_like(local) for _ in range(2)];dist.all_gather(values,local)
        if rank==0:Path(output).write_text(json.dumps([v.tolist() for v in values]))
    finally:dist.destroy_process_group()


class DistributedLearningRateTest(unittest.TestCase):
    def test_rank_local_kl_does_not_split_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=str(Path(tmp)/'result.json')
            mp.spawn(worker,args=(str(Path(tmp)/'rendezvous'),result),nprocs=2,join=True)
            values=json.loads(Path(result).read_text())
        self.assertEqual(values[0],values[1])
        self.assertAlmostEqual(values[0][0],.0003/1.1,places=10)


if __name__=='__main__':unittest.main()
