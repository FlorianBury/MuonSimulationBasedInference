import torch

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





