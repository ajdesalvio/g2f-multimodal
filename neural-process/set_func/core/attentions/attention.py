import einops
import torch
import torch.nn as nn

from .base import BaseMultiHeadAttention


class MultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._build()

    def _build(self) -> None:
        inner_dim = self.head_dim * self.num_heads
        project_out = not (self.num_heads == 1 and self.head_dim == self.v_dim)

        self.to_q = nn.Linear(self.q_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(self.k_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(self.v_dim, inner_dim, bias=False)
        # p_dropout is attention-weight dropout (applied in forward); output
        # dropout belongs to the wrapping attention layer's attn_dropout.
        # The Sequential wrapper keeps the projection's state-dict key at
        # to_out.0.* so checkpoints from the Linear+Dropout era still load.
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, self.v_dim))
            if project_out
            else nn.Identity()
        )

    def forward(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(zq)
        k = self.to_k(zk)
        v = self.to_v(zv)

        q, k, v = map(
            lambda x: einops.rearrange(
                x, "b n (h d) -> b h n d", h=self.num_heads
            ),
            (q, k, v),
        )

        if mask is not None:
            mask = einops.repeat(
                mask, "b nq nkv -> b h nq nkv", h=self.num_heads
            )

        out = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            scale=self.scale,
            dropout_p=self.p_dropout if self.training else 0.0,
        )

        out = einops.rearrange(out, "b h nq d -> b nq (h d)")
        out = self.to_out(out)
        return out
