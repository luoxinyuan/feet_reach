"""Small metric point encoder. Input [..., N*4]: root-frame xyz + valid mask."""
import torch
from torch import nn

class GeometryPointNet(nn.Module):
    def __init__(self, points=128, latent=64):
        super().__init__()
        self.points=points
        self.per_point=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.project=nn.Linear(64,latent)

    def forward(self, packed):
        cloud=packed.reshape(*packed.shape[:-1],self.points,4)
        valid=cloud[...,3]>0.5
        features=self.per_point(cloud[...,:3])
        features=features.masked_fill(~valid.unsqueeze(-1),-1e6)
        pooled=features.amax(dim=-2)
        pooled=torch.where(valid.any(-1,keepdim=True),pooled,torch.zeros_like(pooled))
        return self.project(pooled)
