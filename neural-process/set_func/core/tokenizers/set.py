"""Set tokenizer front-end — encode a coord/channel pair into token embeddings.

This is the front half of :class:`~set_func.core.encoders.TransformerSetEncoder`
(steps 1–3: encode coords, encode channels, concatenate, embed), factored into a
single shared :func:`encode_set` implementation so the encoder has one
tokenization path (no drift).
"""

import torch
from torch import nn


def encode_set(
    x_encoder: nn.Module,
    y_encoder: nn.Module,
    xy_encoder: nn.Module,
    coords: torch.Tensor,
    channels: torch.Tensor,
) -> torch.Tensor:
    """Steps 1–3 of the set front-end: ``xy_encoder(cat(x_enc(coords), y_enc(ch)))``.

    Args:
        x_encoder: Encodes the coordinate axis ``(B, N, D) -> (B, N, d_x)``.
        y_encoder: Encodes the channels ``(B, N, C) -> (B, N, d_y)``.
        xy_encoder: Embeds the concatenation ``(B, N, d_x + d_y) -> (B, N, d)``.
        coords: Index/metric axis values, shape ``(B, N, D)``.
        channels: Signal values, shape ``(B, N, C)``.

    Returns:
        Token embeddings of shape ``(B, N, d)``.
    """
    x_encoded = x_encoder(coords)
    y_encoded = y_encoder(channels)
    xy_concat = torch.cat((x_encoded, y_encoded), dim=-1)
    return xy_encoder(xy_concat)
