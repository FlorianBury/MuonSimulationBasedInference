import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader,Subset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from tools import *
from constants import *
from plotting import plot_3d_hist2d_faces


class ModelParameterChecker:
    def __init__(self, model):
        self.model = model
        self.saved_params = self._snapshot()

    def _snapshot(self):
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

    def has_changed(self):
        """Return True if any model parameter has changed."""
        for name, param in self.model.named_parameters():
            if not torch.equal(param.detach(), self.saved_params[name]):
                return True
        return False

    def compare(self):
        """Report which parameters have changed."""
        changes = {}

        for name, param in self.model.named_parameters():
            current = param.detach()
            saved = self.saved_params[name]

            if not torch.equal(current, saved):
                changes[name] = {
                    "max_abs_diff": (current - saved).abs().max().item(),
                    "num_changed": (current != saved).sum().item(),
                }

        return changes

    def save(self):
        """Update the snapshot to the current model parameters."""
        self.saved_params = self._snapshot()

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out = None,
        neurons = [],
        hidden_activation = None,
        output_activation = None,
        dropout = 0.,
        batch_norm = False,
    ):
        super().__init__()

        # Public attributes #
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.neurons = neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Make layers #
        layers = []
        out_neurons = self.dim_in
        for i in range(len(self.neurons)):
            in_neurons = self.neurons[i - 1] if i > 0 else self.dim_in
            out_neurons = self.neurons[i]
            layers.append(nn.Linear(in_neurons, out_neurons))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(out_neurons))
            if self.hidden_activation is not None:
                layers.append(self.hidden_activation())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        if self.dim_out is not None:
            layers.append(nn.Linear(out_neurons, dim_out))
            if self.output_activation is not None:
                if isinstance(self.output_activation,nn.Softmax):
                    layers.append(self.output_activation(dim=-1))
                else:
                    layers.append(self.output_activation())
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)


class BaseDensityMapper(nn.Module):
    def __init__(
        self,
        xlow,
        xhigh,
        ylow,
        yhigh,
        device = 'cpu',
    ):
        super().__init__()

        if not torch.is_tensor(xlow):
            xlow = torch.tensor(xlow)
        if not torch.is_tensor(xhigh):
            xhigh = torch.tensor(high)
        self.register_buffer("xlow", xlow)
        self.register_buffer("xhigh", xhigh)
        if not torch.is_tensor(ylow):
            ylow = torch.tensor(ylow)
        if not torch.is_tensor(yhigh):
            yhigh = torch.tensor(yhigh)
        self.register_buffer("ylow", torch.log10(ylow))
        self.register_buffer("yhigh", torch.log10(yhigh))

        self.device = device

    def set_lr(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def set_to_device(self,device):
        self.device = device
        self.to(self.device)
        if hasattr(self, "optimizer"):
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)

    def normalize(self, x):
        return 2 * (x - self.xlow) / (self.xhigh - self.xlow) - 1

    def process(self,x):
        y = self(x)
        y = self.ylow + (y + 1.) / 2. * (self.yhigh - self.ylow)
        mask = torch.isnan(torch.pow(10,y)).ravel()
        if mask.sum() > 0:
            print ('pow creates nans')
            print ('x')
            print (x.to(mask.device)[mask])
            print ('y')
            print (y[mask])
        return torch.pow(10,y)

    def extra_loss(self):
        return None

    def loss(self,x,t):
        t = torch.log10(t)
        t = 2 * (t - self.ylow) / (self.yhigh - self.ylow) - 1
        loss = (self(x)-t).square().mean()
        extra_loss = self.extra_loss()
        if extra_loss is not None:
            loss = loss + extra_loss
        return loss

    def param_stats(self, top=10):
        rows = []
        for name, p in self.named_parameters():
            w = p.detach()
            g = p.grad.detach() if p.grad is not None else None
            rows.append((name, w.abs().max().item(), w.norm().item(),
                         g.abs().max().item() if g is not None else float("nan"),
                         torch.isfinite(w).all().item()))
        rows.sort(key=lambda r: -r[1])
        for name, wmax, wnorm, gmax, ok in rows[:top]:
            print(f"{name:40s} |w|max {wmax:9.3e}  |w|norm {wnorm:9.3e}  |g|max {gmax:9.3e}  finite {ok}")
        total = torch.sqrt(sum((p.detach() ** 2).sum() for p in self.parameters()))
        print("total weight norm:", total.item())

    def optimize(self,loss,max_norm=1.0):
        self.optimizer.zero_grad()
        loss.backward()
        self.param_stats()
        #grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        #print ('grad norm',grad_norm)
        self.optimizer.step()

    @torch.no_grad()
    def predict(self, xyz, batch_size=1024):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        xyz = torch.as_tensor(xyz, dtype=torch.float32, device="cpu")

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")

        if xyz.shape[0] == 0:
            return torch.empty((0, 4), dtype=torch.float32)

        self.eval()
        predictions = []

        for batch in xyz.split(batch_size):
            batch = batch.to(self.device)
            values = self.process(batch)
            predictions.append(values.cpu())

        predictions = torch.cat(predictions, dim=0)
        if torch.any(predictions<0):
            raise RuntimeError(f'There are {(predictions<0).sum()}/{len(predictions)} negative outputs')

        return predictions

    def plot(
        self,
        low,
        high,
        points=100,
        batch_size=1024,
        figsize=(10, 8),
        cmap="viridis",
        alpha=0.8,
    ):
        if points < 2:
            raise ValueError("points must be at least 2")

        low = torch.as_tensor(low, dtype=torch.float32).flatten()
        high = torch.as_tensor(high, dtype=torch.float32).flatten()

        if low.numel() == 1:
            low = low.repeat(3)
        if high.numel() == 1:
            high = high.repeat(3)

        if low.numel() != 3 or high.numel() != 3:
            raise ValueError("low and high must be scalars or length-3 sequences")

        if not torch.all(high > low):
            raise ValueError("Every high coordinate must exceed its low coordinate")

        # Generate the 3D grid on the CPU.
        axes = [
            torch.linspace(low[i].item(), high[i].item(), points)
            for i in range(3)
        ]

        grid = torch.stack(
            torch.meshgrid(*axes, indexing="ij"),
            dim=-1,
        )

        xyz = grid.reshape(-1, 3)
        x,y,z = xyz.unbind(dim=-1)

        # Evaluate in batches; results are returned on the CPU.
        values = self.predict(xyz, batch_size=batch_size)

        # Choose marker size based on total point count.
        n = len(values)
        marker_size = max(1, min(30, 20000 / n))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            x.numpy(),
            y.numpy(),
            z.numpy(),
            c=values.numpy(),
            cmap=cmap,
            s=marker_size,
            alpha=alpha,
            linewidths=0,
        )

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.75)
        cbar.set_label(r"Scattering density $\lambda$",fontsize=16)

        ax.set_xlabel("X",fontsize=16)
        ax.set_ylabel("Y",fontsize=16)
        ax.set_zlabel("Z",fontsize=16)
        ax.set_title("Mapper density")

        plt.tight_layout()
        return fig

    def plot_slice(
        self,
        direction,
        value,
        low,
        high,
        points=100,
        batch_size=1024,
        figsize=None,
        cmap="viridis",
        show_materials=True,
    ):
        if direction not in ("x", "y", "z"):
            raise ValueError("direction must be 'x', 'y', or 'z'")

        if points < 2:
            raise ValueError("points must be at least 2")

        fixed_axis = {"x": 0, "y": 1, "z": 2}[direction]
        varying_axes = [i for i in range(3) if i != fixed_axis]

        low = torch.as_tensor(low, dtype=torch.float32).flatten()
        high = torch.as_tensor(high, dtype=torch.float32).flatten()

        if low.numel() == 1:
            low = low.repeat(2)
        if high.numel() == 1:
            high = high.repeat(2)

        if low.numel() != 2 or high.numel() != 2:
            raise ValueError("low and high must be scalars or length-2 sequences")

        if not torch.all(high > low):
            raise ValueError("Every high bound must exceed its low bound")

        # Generate the 2D sampling grid.
        axes = [
            torch.linspace(low[i].item(), high[i].item(), points)
            for i in range(2)
        ]

        grid = torch.stack(
            torch.meshgrid(*axes, indexing="ij"),
            dim=-1,
        )

        uv = grid.reshape(-1, 2)

        # Construct the corresponding 3D coordinates.
        xyz = torch.zeros((uv.shape[0], 3), dtype=torch.float32)
        xyz[:, fixed_axis] = value
        xyz[:, varying_axes[0]] = uv[:, 0]
        xyz[:, varying_axes[1]] = uv[:, 1]

        # Predict lambda.
        predictions = self.predict(
            xyz,
            batch_size=batch_size,
        ).reshape(-1)

        density = predictions.numpy().reshape(points, points)

        # Use the same orientation as imshow:
        # horizontal axis = first varying coordinate.
        density_plot = density.T

        u = np.linspace(low[0].item(), high[0].item(), points)
        v = np.linspace(low[1].item(), high[1].item(), points)
        U, V = np.meshgrid(u, v, indexing="xy")

        # ---------------------------------------------------------
        # Convert radiation lengths into reference lambda values.
        # ---------------------------------------------------------

        material_lambda = {
            name: 1. / x0
            for name, x0 in X0s.items()
            if x0 > 0
        }

        # Select only materials inside the predicted lambda range.
        lambda_min = float(np.nanmin(density_plot))
        lambda_max = float(np.nanmax(density_plot))

        selected_materials = {
            name: lam
            for name, lam in material_lambda.items()
            if lambda_min <= lam <= lambda_max
        } if show_materials else {}

        # ---------------------------------------------------------
        # Plot the predicted density map.
        # ---------------------------------------------------------
        if figsize is None:
            width = high[0].item() - low[0].item()
            height = high[1].item() - low[1].item()

            fig_height = 6
            figsize = (fig_height * width / height, fig_height)

        fig, ax = plt.subplots(figsize=figsize)

        image = ax.imshow(
            density_plot,
            origin="lower",
            extent=[
                low[0].item(),
                high[0].item(),
                low[1].item(),
                high[1].item(),
            ],
            aspect="equal",
            cmap=cmap,
            interpolation="nearest",
        )

        cbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label(
            r"Scattering density $\lambda$ (rad$^2$/m)",
            fontsize=14,
        )
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        cbar.formatter = formatter
        cbar.update_ticks()

        # ---------------------------------------------------------
        # Overlay material contours and colorbar markers.
        # ---------------------------------------------------------
        if selected_materials:
            # Different materials can have nearly identical lambda.
            # Group numerically identical contour levels.
            grouped_levels = {}

            for name, lam in selected_materials.items():
                grouped_levels.setdefault(lam, []).append(name)

            # Matplotlib requires distinct contour levels.
            levels = sorted(grouped_levels.keys())

            if len(levels) >= 1:
                # Contour only the selected material reference values.
                # Handle a single level separately for robust plotting.
                if len(levels) == 1:
                    lam = levels[0]

                    # Draw a black contour for a single material value.
                    ax.contour(
                        U,
                        V,
                        density_plot,
                        levels=[lam],
                        colors="black",
                        linewidths=1.5,
                    )
                else:
                    ax.contour(
                        U,
                        V,
                        density_plot,
                        levels=levels,
                        colors="black",
                        linewidths=1.5,
                    )

                # Add a black marker to the colorbar for each level.
                for lam, names in grouped_levels.items():
                    cbar.ax.plot(
                        [0, 1],
                        [lam, lam],
                        transform=cbar.ax.get_yaxis_transform(),
                        color="black",
                        linewidth=2,
                        clip_on=False,
                    )

                    # Label each marker with its material symbol.
                    cbar.ax.text(
                        1.3,
                        lam,
                        ", ".join(names),
                        transform=cbar.ax.get_yaxis_transform(),
                        va="center",
                        ha="left",
                        fontsize=9,
                        clip_on=False,
                    )

        # ---------------------------------------------------------
        # Axes and formatting.
        # ---------------------------------------------------------
        axis_names = ["x [m]", "y [m]", "z [m]"]

        ax.set_xlabel(axis_names[varying_axes[0]], fontsize=16)
        ax.set_ylabel(axis_names[varying_axes[1]], fontsize=16)
        ax.set_title(f"Scattering density slice: {direction} = {value}")

        # Leave room for the material labels beside the colorbar.
        fig.subplots_adjust(
            left = 0.15,
            right = 0.65,
            bottom = 0.15,
            top = 0.8,
        )
        return fig


class DenseDensityMapper(BaseDensityMapper):
    def __init__(
        self,
        neurons,
        activation = nn.GELU,
        batch_norm = False,
        lr = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp = MLP(
            dim_in = 3,
            dim_out = 1,
            neurons = neurons,
            hidden_activation = activation,
            output_activation = None,
            batch_norm = batch_norm,
        )
        self.optimizer = torch.optim.AdamW(self.mlp.parameters(),lr=lr)
        self.set_to_device(self.device)

    def forward(self,x):
        y = self.mlp(self.normalize(x.to(self.device)))
        return y

class FourierDensityMapper(BaseDensityMapper):
    def __init__(
        self,
        wavelengths,
        neurons,
        activation = nn.GELU,
        batch_norm = False,
        isotropic = False,
        gated = False,
        lr = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        wavelengths = torch.as_tensor(
            wavelengths, dtype=torch.float32
        )

        self.mlp = MLP(
            dim_in = 3 * 2 *len(wavelengths),
            dim_out = 1,
            neurons = neurons,
            hidden_activation = activation,
            output_activation = None,
            batch_norm = batch_norm,
        )
        if gated:
            self.gate = MLP(
                dim_in = 3,
                dim_out = len(wavelengths),
                neurons = neurons,
                hidden_activation = activation,
                output_activation = nn.Sigmoid,
                batch_norm = batch_norm,
            )
        else:
            self.gate = None

        dimensions = self.xhigh - self.xlow
        if isotropic:
            dimensions = dimensions.max().expand_as(dimensions)
        self.register_buffer(
            "wavelengths_norm",
            2 * wavelengths[:, None] / dimensions[None, :]
        )

        self.optimizer = torch.optim.Adam(self.mlp.parameters(),lr=lr)
        self.set_to_device(self.device)

    def forward(self,x):
        x = self.normalize(x.to(self.device))

        # x: [batch, 3]
        # wavelengths_norm: [num_wavelengths, 3]
        # angles: [batch, 3, num_wavelengths]
        angles = (
            2 * math.pi
            * x.unsqueeze(-1)
            / self.wavelengths_norm.T.unsqueeze(0)
        )

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        if self.gate is not None:
            alpha = self.gate(x).unsqueeze(1)
            sin = sin * alpha
            cos = cos * alpha

        features = torch.cat(
            [sin,cos],
            dim=-1,
        ).flatten(start_dim=1)

        y = self.mlp(features)
        return y

class VoxelisedMapper(BaseDensityMapper):
    """
    Direct voxelized 3D grid density mapper for diagnostic testing.
    Replaces the MLP with a learnable 3D tensor of parameters,
    eliminating global weight-sharing and gradient conflicts.
    """
    def __init__(
        self,
        resolution = (32, 32, 32), # (nx, ny, nz) grid resolution
        lr = 1e-2,
        initial_val = 0.0,         # Initial value in normalized space
        alpha_tv = 0.,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.alpha_tv = alpha_tv

        # Create learnable 3D voxel parameters of shape (nx, ny, nz)
        # Initialized to initial_val (e.g. 0.0 maps to middle of ylow/yhigh range)
        if not torch.is_tensor(initial_val):
            initial_val = torch.tensor(initial_val)
        initial_val = torch.log10(initial_val)
        initial_val = 2 * (initial_val - self.ylow) / (self.yhigh - self.ylow) - 1
        raw_voxels = torch.full(resolution, initial_val, dtype=torch.float32)
        self.raw_voxels = nn.Parameter(raw_voxels)

        # Optimizer targeting only the voxel grid
        self.optimizer = torch.optim.Adam([self.raw_voxels], lr=lr)
        self.set_to_device(self.device)

    @property
    def voxels(self):
        return torch.pow(10,self.ylow + (self.raw_voxels + 1.) / 2. * (self.yhigh - self.ylow))

    def extra_loss(self):
        """
        Computes L1 Total Variation loss on the voxel grid
        to encourage spatial smoothness and contiguous structures.
        """
        # Get the processed density tensor (same range as your forward pass)
        voxels = torch.tanh(self.raw_voxels)  # Shape: (nx, ny, nz)

        # Calculate absolute differences between adjacent voxels along each axis
        diff_x = torch.abs(voxels[1:, :, :] - voxels[:-1, :, :]).mean()
        diff_y = torch.abs(voxels[:, 1:, :] - voxels[:, :-1, :]).mean()
        diff_z = torch.abs(voxels[:, :, 1:] - voxels[:, :, :-1]).mean()

        # Total TV loss is the sum of variations across all three dimensions
        return self.alpha_tv * (diff_x + diff_y + diff_z)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 3) containing spatial coordinates (x, y, z)
        returns: Tensor of shape (batch_size, 1) in range [-1, 1] expected by process()
        """
        x = x.to(self.device)

        # Normalize spatial coordinates to [-1, 1] for grid_sample
        x = self.normalize(x)

        # Apply tanh to keep voxel outputs bounded within [-1, 1]
        # (matching the range expected by BaseDensityMapper.process)
        #voxels = torch.tanh(self.raw_voxels)
        voxels = self.raw_voxels

        # Prepare for F.grid_sample:
        # Input tensor needs shape (N, C, D, H, W) -> (1, 1, nx, ny, nz)
        voxels = voxels.unsqueeze(0).unsqueeze(0)

        # Grid needs shape (N, D_out, H_out, W_out, 3) where the last dim is (x, y, z)
        # For point-wise sampling of batch B: shape (1, 1, 1, B, 3)
        grid = x.view(1, 1, 1, -1, 3)

        # Trilinear interpolation using PyTorch's native grid_sample
        sampled = F.grid_sample(
            voxels,
            grid,
            mode='bilinear',          # In PyTorch, 3D mode='bilinear' executes trilinear interpolation
            padding_mode='border',
            align_corners=True
        )
        # sampled shape: (1, 1, 1, 1, B) -> reshape to (B, 1)
        return sampled.view(-1, 1)
