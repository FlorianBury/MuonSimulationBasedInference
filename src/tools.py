import math
import torch
from constants import MUON_MASS_GEV

def get_axes(t):
    # (Frisvad on n = -t, so only singular point is at +z)
    nx, ny, nz = (-t).unbind(-1)
    k = 1.0 / (1.0 + nz).clamp(min=1e-6)
    b = -nx * ny * k
    e1 = torch.stack([1.0 - nx * nx * k, b, -nx], dim=-1)
    e2 = torch.cross(t, e1, dim=-1)
    return e1, e2

def project_track(t,ri,lo,hi,eps=1e-9):
    # Find 3 possible walls (depending on direction)
    target = torch.where(t > 0, hi, lo)

    # Find smallest lambda and project point
    moving = t.abs() > eps
    lam = torch.where(
        moving,
        (target - ri) / torch.where(moving,t,torch.ones_like(t)),
        torch.full_like(t, float("inf")),
    )
    L = lam.min(dim=-1).values
    rp = ri + L[:, None] * t
    return rp


def positions_to_observables(
    ri,
    rf,
    di,
    df,
    lo,
    hi,
):
    # Make sure directions are normed
    di = di / di.norm(dim=-1, keepdim=True)
    df = df / df.norm(dim=-1, keepdim=True)

    # Find local frame for each muon
    e1,e2 = get_axes(di)

    # Deflection components
    dft = (df * di).sum(-1)
    a1 = (df * e1).sum(-1) / dft
    a2 = (df * e2).sum(-1) / dft

    # Project ri along track by di
    rp = project_track(di,ri,lo,hi)

    # Offset: extend outgoing line to the plane through xp perpendicular to di
    lam_star = ((rp - rf) * di).sum(-1) / dft
    X = rf + lam_star[:, None] * df
    d = X - rp
    s1 = (d * e1).sum(-1)
    s2 = (d * e2).sum(-1)

    return a1, a2, s1, s2

def observables_to_positions(
    a1,
    a2,
    s1,
    s2,
    ri,
    di,
    lo,
    hi,
    eps=1e-9,
):
    # Make sure direction is normed
    di = di / di.norm(dim=-1, keepdim=True)

    # Find local frame for each muon
    e1,e2 = get_axes(di)

    # Project ri along track by di
    rp = project_track(di,ri,lo,hi)

    # Get outgoing direction
    df = di + a1[:, None] * e1 + a2[:, None] * e2
    df = df / df.norm(dim=-1, keepdim=True)

    # Point where df meets rp perpendicular to di
    X = rp + s1[:, None] * e1 + s2[:, None] * e2

    # rf is where line of X+lam*df meets the box
    rf = project_track(df, X, lo, hi)
    #hit = ((rf >= lo - tol) & (rf <= hi + tol)).all(dim=-1)
    return rf,df

def gaussian_nll(mapper, ri, rf, pi, obs, step_size, jitter=False):
    """
    mapper    : needs .process(x[M,3]) -> lambda [M] or [M,1]  (differentiable)
    ri, rf    : (B,3) entry / exit positions of the track; the path is the straight segment ri -> rf
    pi        : (B,1) incoming momentum [GeV]
    obs       : (B,4) = (a1, a2, s1, s2) from positions_to_observables
    step_size : max sub-step length along the segment
    jitter    : if True, draw one random point per stratum at every call (stratified jittering,
                as in the paper); if False, use stratum midpoints.
    Returns (loss, q) with q (B,2) the per-projection Mahalanobis distances (E[q] = 2 at truth).
    """
    seg = rf - ri                                               # (B,3)
    Ltot = seg.norm(dim=-1)                                     # (B,) length of the straight segment
    ok = Ltot > 1e-6
    ri, seg, pi, obs, Ltot = ri[ok], seg[ok], pi[ok], obs[ok], Ltot[ok]
    B = ri.shape[0]

    # n strata per muon (n is common to the batch), each of length h = L / n
    n = max(1, math.ceil(Ltot.max().item() / step_size))
    h = (Ltot / n)[:, None]                                     # (B,1)
    k = torch.arange(n, device=ri.device, dtype=ri.dtype)[None, :]          # (1,n)
    u = torch.rand(B, n, device=ri.device, dtype=ri.dtype) if jitter else 0.5
    t = (k + u) / n                                             # path fraction in [0,1], (B,n) or (1,n)
    pts = ri[:, None, :] + t[:, :, None] * seg[:, None, :]      # (B,n,3) points on the segment
    lam = mapper.process(pts.reshape(-1, 3)).reshape(B, n).double()         # grads flow here

    h = h.double()
    T = (n - 1 - k.double()) * h                                # path left after each stratum
    wA = h.expand(B, n)
    wB = h * (h / 2 + T)
    wC = h * (h**2 / 3 + h * T + T**2)

    p = pi.reshape(B).double()
    bp = p**2 / torch.sqrt(p**2 + MUON_MASS_GEV**2)             # beta * p
    KAPPA_GEV, EPS_HL = 0.0136, 0.038                          # 13.6 MeV in GeV, Highland epsilon
    sl = ((KAPPA_GEV / bp) ** 2)[:, None] * lam
    c00, c01, c11 = (sl * wA).sum(1), (sl * wB).sum(1), (sl * wC).sum(1)
    T_tot = (lam * wA).sum(1)                                  # total thickness in radiation lengths
    g = (1 + EPS_HL * torch.log(T_tot.clamp_min(1e-30))).clamp_min(1e-3)   # physical branch g > 0
    c00, c01, c11 = c00 * g**2, c01 * g**2, c11 * g**2
    det = (c00 * c11 - c01**2).clamp_min(1e-300)

    o = obs.double()
    th, x = o[:, 0:2], o[:, 2:4]                                # projections (a1,s1) and (a2,s2)
    q = (c11[:, None] * th**2 - 2 * c01[:, None] * th * x + c00[:, None] * x**2) / det[:, None]
    loss = (torch.log(det) + 0.5 * q.sum(1)).mean()
    return loss, q

def energy_distance_loss(
    x,                      # Shape: (B, S_x, F) - True/Reference samples
    y,                      # Shape: (B, S_y, F) - Simulated/Surrogate samples
    gamma = 0.,
    p = 2,
    eps = 1e-8,
):
    B, S_y, F = y.shape
    K = S_y

    # 1. Weights based on simulated samples y (assuming samp_obs was meant to be y)
    norms = torch.norm(y, dim=-1)                # Shape: (B, K)
    weights = 1.0 + gamma * (norms ** p)         # Shape: (B, K)

    # 2. Attractive Term: x vs y (Shape: B, S_x, S_y)
    # x[:, :, None, :] -> (B, S_x, 1, F)
    # y[:, None, :, :] -> (B, 1, S_y, F)
    diff_sq = ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(-1) # Shape: (B, S_x, S_y)

    # weights needs to align with y's sample dimension (axis 2)
    # weights[:, None, :] has shape (B, 1, K)
    weighted_diff_sq = weights[:, None, :] * diff_sq # Shape: (B, S_x, S_y)

    # Mean over samples (dimensions 1 and 2), keeping batch dimension (returns shape B)
    att = torch.sqrt(weighted_diff_sq + eps).mean(dim=(1, 2))

    # 3. Repulsive Term: y vs y (Pairwise distances within simulated samples)
    # y[:, :, None, :] -> (B, K, 1, F)
    # y[:, None, :, :] -> (B, 1, K, F)
    rep_diff_sq = ((y[:, :, None, :] - y[:, None, :, :]) ** 2).sum(-1) # Shape: (B, K, K)
    rep_matrix = torch.sqrt(rep_diff_sq + eps)

    # Exclude diagonal (self-distances) using mask
    off = ~torch.eye(K, dtype=torch.bool, device=y.device) # Shape: (K, K)

    # Apply mask across batch and average over valid pairs
    # rep_matrix[:, off] results in shape (B, K * (K - 1))
    rep = rep_matrix[:, off].reshape(B, K, K - 1).sum(dim=(1, 2)) / (K * (K - 1))

    # 4. Energy distance component: 2 * Attractive - Repulsive (or return components depending on your loss formulation)
    # Usually Energy Distance = 2 * att - rep (or similar depending on convention)
    return 2 * att.mean(), -rep.mean()
