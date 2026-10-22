import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm

def plot_3d_hist2d_faces(
    x, y, z,
    xlim, ylim, zlim,
    bins=30,
    cmap='viridis',
    tol=1e-6,
):
    """
    Plot 2D histograms on the six faces of a 3D rectangular box.

    Each point is assigned to the face(s) it lies on, within tolerance.
    Points are NOT projected onto faces they do not belong to.

    Parameters
    ----------
    x, y, z : array-like
        Point coordinates. NumPy arrays or PyTorch tensors.
    xlim, ylim, zlim : tuple
        Bounds of the rectangular box.
    bins : int
        Number of bins per histogram dimension.
    cmap : str
        Matplotlib colormap.
    tol : float
        Absolute tolerance for assigning points to faces,
        in the same units as the coordinates.
    """

    def to_numpy(a):
        if a is None:
            return None
        if hasattr(a, 'detach'):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    x = to_numpy(x).ravel()
    y = to_numpy(y).ravel()
    z = to_numpy(z).ravel()

    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, and z must have the same length.")

    # Keep only finite points inside the box
    valid = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(z) &
        (x >= xlim[0] - tol) & (x <= xlim[1] + tol) &
        (y >= ylim[0] - tol) & (y <= ylim[1] + tol) &
        (z >= zlim[0] - tol) & (z <= zlim[1] + tol)
    )

    x, y, z = x[valid], y[valid], z[valid]

    # Identify points belonging to each face
    masks = {
        'x_min': np.abs(x - xlim[0]) <= tol,
        'x_max': np.abs(x - xlim[1]) <= tol,
        'y_min': np.abs(y - ylim[0]) <= tol,
        'y_max': np.abs(y - ylim[1]) <= tol,
        'z_min': np.abs(z - zlim[0]) <= tol,
        'z_max': np.abs(z - zlim[1]) <= tol,
    }

    # Define the coordinates used for each face histogram
    face_specs = {
        'z_min': (x, y, xlim, ylim, 'xy', zlim[0]),
        'z_max': (x, y, xlim, ylim, 'xy', zlim[1]),
        'y_min': (x, z, xlim, zlim, 'xz', ylim[0]),
        'y_max': (x, z, xlim, zlim, 'xz', ylim[1]),
        'x_min': (y, z, ylim, zlim, 'yz', xlim[0]),
        'x_max': (y, z, ylim, zlim, 'yz', xlim[1]),
    }

    # Calculate histograms using only points on the relevant face
    histograms = {}

    for face, (a, b, range_a, range_b, plane, fixed) in face_specs.items():
        mask = masks[face]

        H, ea, eb = np.histogram2d(
            a[mask],
            b[mask],
            bins=bins,
            range=[range_a, range_b],
        )

        histograms[face] = (H, ea, eb, plane, fixed)

    # Shared colour scale
    vmax = max(
        (H.max() for H, *_ in histograms.values()),
        default=0
    )

    color_map = plt.get_cmap(cmap)

    # Logarithmic colour scale (zero-count bins remain transparent)
    norm = LogNorm(
        vmin=1,
        vmax=max(vmax, 1)
    )

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot histograms on their respective faces
    for face, (H, ea, eb, plane, fixed) in histograms.items():

        A, B = np.meshgrid(ea, eb, indexing='ij')
        colors = color_map(norm(H))
        colors[H == 0, 3] = 0

        if plane == 'xy':
            C = np.full_like(A, fixed)
            ax.plot_surface(
                A, B, C,
                facecolors=colors,
                shade=False,
                edgecolor='none',
                antialiased=False,
            )

        elif plane == 'xz':
            C = np.full_like(A, fixed)
            ax.plot_surface(
                A, C, B,
                facecolors=colors,
                shade=False,
                edgecolor='none',
                antialiased=False,
            )

        elif plane == 'yz':
            C = np.full_like(A, fixed)
            ax.plot_surface(
                C, A, B,
                facecolors=colors,
                shade=False,
                edgecolor='none',
                antialiased=False,
            )

    # Draw volume box
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0],
        [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1],
        [x1, y1, z1], [x0, y1, z1]
    ])

    # Connect the 12 edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for i, j in edges:
        ax.plot(
            *zip(corners[i], corners[j]),
            color='gray',
            linewidth=2.,
            alpha=0.7,
        )

    # Axis limits and proportions
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect((
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        zlim[1] - zlim[0],
    ))

    # Colour bar
    sm = cm.ScalarMappable(norm=norm, cmap=color_map)
    sm.set_array([])

    fig.colorbar(
        sm,
        ax=ax,
        shrink=0.65,
        label='Point count per bin',
    )

    plt.tight_layout()

    return fig, ax


def plot_particles_3d(
    ri,
    rf,
    di,
    df,
    pi,
    pf,
    low,
    high,
    N=100,
    indices=None,
    arrow_scale=1.0,
    show_direction=True,
    show_path=False,
    seed=None,
):
    """
    Plot initial and final particle positions and momentum-scaled
    direction vectors in 3D.

    Parameters
    ----------
    ri, rf : array-like, shape (n, 3)
        Initial and final positions.
    di, df : array-like, shape (n, 3)
        Initial and final direction vectors.
    pi, pf : array-like, shape (n,) or (n, 1)
        Initial and final scalar momentum magnitudes.
    low, high : array-like, shape (3,)
        Lower and upper box boundaries.
    N : int
        Maximum number of particles to plot.
    indices : array-like or None
        Specific particle indices. If None, sample randomly.
    arrow_scale : float
        Overall scale factor for direction arrows.
    show_direction : bool
        Whether to plot direction arrows.
    show_path : bool
        Whether to connect initial and final positions.
    seed : int or None
        Random seed for reproducible sampling.

    Returns
    -------
    fig, ax
        Matplotlib figure and 3D axes.
    """

    # Convert PyTorch tensors or NumPy arrays to NumPy.
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    ri, di, pi = map(to_numpy, (ri, di, pi))
    rf, df, pf = map(to_numpy, (rf, df, pf))
    low, high = map(to_numpy, (low, high))

    # Ensure momenta are one-dimensional scalar magnitudes.
    pi = pi.reshape(-1)
    pf = pf.reshape(-1)

    n_particles = len(ri)

    # Validate input shapes.
    for name, arr in {
        "di": di,
        "pi": pi,
        "rf": rf,
        "df": df,
        "pf": pf,
    }.items():
        expected_shape = (n_particles, 3) if name in {"di", "rf", "df"} else (n_particles,)

        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {arr.shape}."
            )

    if low.shape != (3,) or high.shape != (3,):
        raise ValueError("low and high must each have shape (3,).")

    if np.any(high <= low):
        raise ValueError("Every component of high must exceed low.")

    if arrow_scale < 0:
        raise ValueError("arrow_scale must be non-negative.")

    # Select particles.
    if indices is None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            n_particles,
            size=min(N, n_particles),
            replace=False,
        )
    else:
        indices = np.asarray(indices, dtype=int).reshape(-1)

        if np.any(indices < 0) or np.any(indices >= n_particles):
            raise IndexError("indices contains invalid particle indices.")

    ri, di, pi = ri[indices], di[indices], pi[indices]
    rf, df, pf = rf[indices], df[indices], pf[indices]

    # Normalise direction vectors.
    def normalise(v):
        norms = np.linalg.norm(v, axis=1, keepdims=True)

        return np.divide(
            v,
            norms,
            out=np.zeros_like(v, dtype=float),
            where=norms > 0,
        )

    di = normalise(di)
    df = normalise(df)

    # Logarithmically scale directions by scalar momentum.
    arrow_i = arrow_scale * np.log1p(np.maximum(pi, 0))[:, None] * di
    arrow_f = arrow_scale * np.log1p(np.maximum(pf, 0))[:, None] * df

    # Create figure.
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Initial positions.
    ax.scatter(
        *ri.T,
        color="forestgreen",
        label="Initial positions",
        s=20,
        alpha=0.8,
    )

    # Final positions.
    ax.scatter(
        *rf.T,
        color="darkred",
        label="Final positions",
        s=20,
        alpha=0.8,
    )

    # Optional paths.
    if show_path:
        for r0, r1 in zip(ri, rf):
            ax.plot(
                [r0[0], r1[0]],
                [r0[1], r1[1]],
                [r0[2], r1[2]],
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )

    # Direction arrows.
    if show_direction:
        ax.quiver(
            *ri.T,
            *arrow_i.T,
            color="forestgreen",
            normalize=False,
            arrow_length_ratio=0.2,
        )

        ax.quiver(
            *rf.T,
            *arrow_f.T,
            color="darkred",
            normalize=False,
            arrow_length_ratio=0.2,
        )

    # Box dimensions.
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(high - low)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Initial and Final Particle States")
    ax.legend()

    plt.tight_layout()
    return fig, ax
