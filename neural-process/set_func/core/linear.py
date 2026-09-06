import math

import einops
import torch
import torch.nn as nn


class GroupLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        groups = groups or 1
        assert (
            in_features % groups == 0
        ), "in_features must be divisible by groups"
        assert (
            out_features % groups == 0
        ), "out_features must be divisible by groups"

        self.groups = groups
        self.in_feat_per_group = in_features // groups
        self.out_feat_per_group = out_features // groups

        self.weight = nn.Parameter(
            torch.empty(
                groups, self.in_feat_per_group, self.out_feat_per_group
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(groups, self.out_feat_per_group)
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mirrors nn.Linear's default (kaiming-uniform, a=sqrt(5)) with the
        # per-group fan: each group is an independent in_feat_per_group ->
        # out_feat_per_group map, and torch's fan calculator misreads the 3-D
        # (groups, in, out) weight as fan_in = in * out, so the standard init
        # helpers must not be applied to it directly.
        bound = 1.0 / math.sqrt(self.in_feat_per_group)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = einops.rearrange(
            x, "... (g c) -> ... g c", g=self.groups, c=self.in_feat_per_group
        )  # [B, G, in_feat_per_group]
        out = torch.einsum(
            "...gi,gio->...go", x, self.weight
        )  # [B, G, out_feat_per_group]
        if self.bias is not None:
            out += self.bias
        out = einops.rearrange(
            out,
            "... g c -> ... (g c)",
        )
        return out
