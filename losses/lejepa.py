"""LEJEPA: mean-view invariance and Gaussian SIGReg (from Retrieval)."""

import torch
from torch import nn


class SIGReg(torch.nn.Module):
    def __init__(self, num_projections=1024,  t_max=5, knots=51, seed=0):
        super().__init__()
        self.num_projections = num_projections
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self._seed = seed
        g = torch.Generator()
        g.manual_seed(self._seed)
        self._generator = g

    def forward(self, proj):
        if self._generator.device != proj.device:
            self._generator = torch.Generator(device=proj.device).manual_seed(int(self._seed))

        #proj : (B,d)
        A = torch.randn(proj.size(-1), self.num_projections, device=proj.device, dtype=proj.dtype, generator=self._generator) #(d,nb_proj)
        A = A.div_(A.norm(p=2, dim=0)) #(d,nb_proj)
        x_t = (proj @ A).unsqueeze(-1) * self.t #(B, nb_proj, T)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square() #(nb_proj, T)
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
    
class LEJEPA(nn.Module):
    def __init__(self, params):
        """
        """
        super().__init__()  
        self.params = params
        self.lambd = float(getattr(self.params, 'lambda_reg', 0.05))
        self.Vg = getattr(self.params, 'global_views', getattr(self.params, 'Vg', 2))
        self.loss_reg = SIGReg(
            num_projections=getattr(self.params, 'num_projections', 1024),
            knots=getattr(self.params, 'knots', 51),
            seed=getattr(self.params, 'seed', 0),
        )
             
    def forward(self, zs, mode='train'):
        """
        """
        if isinstance(zs, (list, tuple)):
            zs = torch.stack(zs, dim=0)

        metrics = {}
        mu = zs[:self.Vg].mean(dim=0, keepdim=True) #(1,B,D)
        Linv = (zs-mu).pow(2).mean() #(1,)
        Lreg = torch.stack([self.loss_reg(zi) for zi in zs]).mean()
        L = (1 - self.lambd) * (Linv) + self.lambd * Lreg 
        metrics[f'{mode}/Linv'] = Linv.item()
        metrics[f'{mode}/Lreg'] = Lreg.item()
        metrics[f'{mode}/L'] = L.item()    
        return L, metrics
