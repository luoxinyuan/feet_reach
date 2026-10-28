"""Regression tests execute the real reward bodies without starting Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

source=Path(__file__).resolve().parents[1]/'active_adaptation/envs/mdp/rewards/locomotion.py'
tree=ast.parse(source.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('action_rate_l2','action_rate2_l2')]
for node in functions:node.decorator_list=[]
namespace={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)

def reward(name,buffer):
    return namespace[name](SimpleNamespace(action_manager=SimpleNamespace(action_buf=buffer)))

class ActionRateTests(unittest.TestCase):
    def test_constant_unequal_joint_actions_have_no_time_penalty(self):
        q=torch.arange(29,dtype=torch.float32)[None,None,:].expand(4,3,29)
        for name in ('action_rate_l2','action_rate2_l2'):
            torch.testing.assert_close(reward(name,q),torch.zeros(4,1))

    def test_change_in_any_joint_is_penalized_and_batches_are_independent(self):
        q=torch.zeros(29,3,29)
        for i in range(29):q[i,0,i]=i+1
        expected=-torch.arange(1,30,dtype=torch.float32).square()[:,None]
        for name in ('action_rate_l2','action_rate2_l2'):
            torch.testing.assert_close(reward(name,q),expected)
            torch.testing.assert_close(reward(name,q.flip(-1)),expected)

    def test_constant_velocity_has_zero_second_difference(self):
        slopes=torch.arange(1,30,dtype=torch.float32)
        q=torch.stack([2*slopes,slopes,torch.zeros(29)])[None]
        torch.testing.assert_close(reward('action_rate2_l2',q),torch.zeros(1,1))
        torch.testing.assert_close(reward('action_rate_l2',q),-slopes.square().sum().reshape(1,1))

    def test_gradient_only_uses_correct_time_slices(self):
        q=torch.zeros(1,3,29);q[0,0,17]=1;q.requires_grad_()
        reward('action_rate_l2',q).sum().backward()
        expected=torch.zeros_like(q);expected[0,0,17]=-2;expected[0,1,17]=2
        torch.testing.assert_close(q.grad,expected)

if __name__=='__main__':unittest.main()
