import math
import torch

from tools import *
from constants import MUON_MASS_GEV


class BasePropagator:
    def __init__(
        self,
    ):
        self.low = None
        self.high = None

    def init_box(self,dataset):
        self.low = dataset.low
        self.high = dataset.high

    def propagate(
        self,
        mapper,
        ri,
        di,
        pi,
        eps = 1e-12,
    ):
        assert self.low is not None
        assert self.high is not None

        dev = mapper.device
        N = ri.shape[0]
        pos = ri.clone()
        vec = di.clone() / di.norm(dim=-1, keepdim=True)
        mom = pi.clone()

        rf = torch.full_like(ri, float("nan"))
        df = torch.full_like(di, float("nan"))
        pf = torch.full_like(pi, float("nan"))
        active = torch.arange(N)

        next_progress = 1
        progress_bar = ""
        counter = 0
        while active.numel() > 0:
            # select still active muons
            x, t, p = pos[active], vec[active], mom[active]
            # Scattering over one step
            x_new, t_new, p_new = self.propagate_step(
                lam = mapper.process(x).cpu(),
                x = x,
                t = t,
                p = p,
            )
            # Muons whose step ended outside the box: find where the step crossed the wall
            out = ((x_new < self.low) | (x_new > self.high)).any(-1)
            if out.any():
                i = out.nonzero().squeeze(-1)
                seg = x_new[i] - x[i]
                target = torch.where(seg > 0, self.high, self.low)
                moving = seg.abs() > eps
                tau = torch.where(moving,
                                  (target - x[i]) / torch.where(moving, seg, torch.ones_like(seg)),
                                  torch.full_like(seg, float("inf"))).min(-1).values.clamp(0, 1)
                rf[active[i]] = x[i] + tau[:, None] * seg
                df[active[i]] = t_new[i]
                pf[active[i]] = p_new[i]

            keep = ~out
            pos[active[keep]] = x_new[keep]
            vec[active[keep]] = t_new[keep]
            mom[active[keep]] = p_new[keep]
            active = active[keep]

            progress = int(100 * (N - active.numel()) / N)
            counter += 1

            while progress >= next_progress and next_progress <= 100:
                progress_bar += "="
                print(f"\rPropagation : [{progress_bar:<100}] {next_progress}%", end="", flush=True)
                next_progress += 1
        print (f"\nPropagation done in {counter} steps, max distance {(ri-rf).norm(dim=-1).max():.3f}")
        return rf, df, pf

    def propagate_step(self,lam,x,t,p):
        raise NotImplementedError


class RossiPropagator(BasePropagator):
    """ Statistical Reconstruction for Cosmic Ray Muon Tomography """
    def __init__(
        self,
        step_size,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.step_size = step_size

    def propagate_step(self,lam,x,t,p):
        if torch.any(lam<0):
            raise RuntimeError(f'There are {(lam<0).sum()} negative lambda values')
        if torch.any(torch.isnan(lam)):
            raise RuntimeError(f'There are {torch.isnan(lam).sum()} nan lambda values')
        n = x.shape[0]
        h = self.step_size

        # Get sigma and generate delta theta
        beta = p / torch.sqrt(p.square() + MUON_MASS_GEV**2)
        sigma = torch.sqrt(lam * h) * 0.015 / (beta * p)
        a = sigma * torch.randn(n, 2)
        # Get axes and update direction #
        e1, e2 = get_axes(t)
        t_new = t + a[:, :1] * e1 + a[:, 1:] * e2
        t_new = t_new / t_new.norm(dim=-1, keepdim=True)
        # Update position #
        xi = torch.randn(n, 2)
        s = (h / 2) * a + (sigma * h / (2 * 3**0.5)) * xi
        x_new = x + h * t + s[:, :1] * e1 + s[:, 1:] * e2
        return x_new, t_new, p
