"""CPU mathematical checks; no physics simulator needed."""
import torch
from active_adaptation.learning.hierarchical.flow_latent import LatentFlow

def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    flow=LatentFlow(4, action_dim=2, horizon=1, width=64)
    opt=torch.optim.Adam(flow.parameters(), lr=0.002)
    condition=torch.randn(256,4)
    target=(condition[:,:2].tanh()*.4)
    noise=torch.randn_like(target)
    t=torch.rand(256,1)
    x=(1-t)*noise+t*target
    truth=target-noise
    first=None
    for i in range(500):
        loss=(flow(condition,x,t)-truth).square().mean()
        if first is None:first=loss.item()
        opt.zero_grad();loss.backward();opt.step()
    assert loss.item()<first*.08,(first,loss.item())
    chunk=flow.sample(condition,steps=8,noise=noise)
    assert chunk.shape==(256,1,2) and torch.isfinite(chunk).all()
    torch.testing.assert_close(torch.atanh(chunk).tanh(),chunk)
    # Each ten-step label must stay inside a single episode.
    valid=torch.ones(2,32,dtype=torch.bool)
    valid[0,12]=False
    starts=valid.unfold(1,10,1).all(-1)
    assert not starts[0,3:13].any() and starts[0,:3].all() and starts[0,13:].all()
    assert starts[1].all()
    print({'passed':True,'initial_mse':first,'final_mse':loss.item()})
if __name__=='__main__':main()
