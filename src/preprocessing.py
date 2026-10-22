import torch
from torch import nn


class ConditionPreprocessing(nn.Module):
    def __init__(self, a=0.0, b=1.0):
        super().__init__()

        self.a = a
        self.b = b

        self.register_buffer("x_min", torch.empty(0))
        self.register_buffer("x_max", torch.empty(0))

        self.fitted = False

    @torch.no_grad()
    def fit(self, x):
        """
        Fit the scaler on x of shape (N, F).
        Compute min and max independently for each feature.
        """
        self.x_min = x.amin(dim=0).detach().clone()
        self.x_max = x.amax(dim=0).detach().clone()

        self.fitted = True
        return self

    def forward(self, x):
        """Scale each feature to [a, b]."""
        if not self.fitted:
            raise RuntimeError("Call fit(x) before using the scaler.")

        denominator = self.x_max - self.x_min

        # Avoid division by zero for constant features
        denominator = torch.where(
            denominator == 0,
            torch.ones_like(denominator),
            denominator
        )

        x_scaled = self.a + (
            (x - self.x_min) / denominator
        ) * (self.b - self.a)

        # Constant features map to the midpoint of [a, b]
        constant_features = (self.x_max == self.x_min)
        midpoint = (self.a + self.b) / 2

        return torch.where(
            constant_features,
            torch.full_like(x_scaled, midpoint),
            x_scaled
        )

class ObservablePreprocessing(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c = c

    @torch.no_grad()
    def fit(self, x):
        pass

    def forward(self,x):
        return torch.asinh(x/self.c)

    def inverse(self,x):
        return self.c * torch.sinh(x)

