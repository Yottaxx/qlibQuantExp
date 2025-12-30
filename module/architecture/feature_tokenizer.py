import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    FT-Transformer style feature tokenizer for numeric inputs.

    Maps each scalar feature x_{t,n} to a D-dim token using per-feature
    affine parameters: x * weight[n] + bias[n].
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        *,
        bias: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.init_std = float(init_std)

        self.weight = nn.Parameter(torch.empty(self.num_features, self.d_model))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_features, self.d_model))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"FeatureTokenizer expects x as [B,T,N], got {tuple(x.shape)}")
        _, _, n = x.shape
        if int(n) != self.num_features:
            raise ValueError(
                f"FeatureTokenizer expects N={self.num_features}, got N={int(n)}."
            )
        out = x.unsqueeze(-1) * self.weight.view(1, 1, self.num_features, self.d_model)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, self.num_features, self.d_model)
        return out
