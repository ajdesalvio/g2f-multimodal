"""Transformer Neural Process model: per-sample tokenizer + TNP encoder.

The unit of computation is a **task** (a context set + a query set). A
:class:`SampleTokenizer` turns each canonical ``[B, n, ...]`` task batch into one
token per sample ``[B, n, d_model]`` (NP construction: a density channel
flags missing ``y``). A TNP cross-sample encoder
(:mod:`set_func.core.np`) then refines the query tokens against the context, and
a decoder + output head produce a predictive
:class:`~set_func.predictions.Prediction` over the query yields.

Contract: ``forward(NPTaskBatch) -> Prediction``. The head is config-selected
(``head:`` in the model YAML) — the heteroscedastic default returns a
:class:`~set_func.predictions.GaussianPrediction` (``.mean`` are the query μ,
``.log_prob`` the NLL); a point model would use a
:class:`~set_func.heads.PointHead` (``PointPrediction``, no density). The
full-vs-pseudo-token attention variant is purely *which*
``transformer_encoder`` is instantiated — there is no ``mode`` flag in the model.
"""

from collections.abc import Sequence

import einops
import torch
import torch.nn as nn

from ..core.decoders import BaseDecoder
from ..core.encoders import BaseSetEncoder
from ..core.transformers.base import BaseTransformerEncoder
from ..heads import BaseHead
from ..predictions import Prediction


class SampleTokenizer(nn.Module):
    """Turn a canonical ``[B, n, ...]`` task batch into one token per sample.

    The per-modality set encoders accept only a single leading batch dim, so the
    tokenizer privately collapses ``(B, n) -> B*n`` (via ``einops``), runs the
    encoders, and restores the task dim — the batch/model stay canonical
    ``[B, n, ...]``.

    Follows the standard NP construction: a **density channel is appended to ``y``**
    flagging whether ``y`` is missing (polarity: ``1`` = query/missing,
    ``0`` = context/observed); the query's ``y`` value itself is zeroed. Context
    and query then go through the **same** ``y_encoder`` and the **same**
    ``token_proj`` (the xy-encoder); only ``(y, density)`` differ. There is
    no learned ``y_missing`` placeholder.

    Args:
        vi_encoder: Set encoder over the VI view (``out_dim = d_vi``).
        fuse_x: Fuses the per-modality ``x`` embeddings -> ``d_x``
            (concat -> MLP by default).
        token_proj: the xy-encoder — MLP ``[d_x + d_y] -> d_model``.
        d_model: Output token width.
        weather_encoder: Optional set encoder over the weather view.
        geno_encoder: Optional MLP over ``derived_features[geno_key]``.
        geno_key: Which derived-feature key(s) the geno branch reads
            (default ``"genomic_add"``; a list concatenates the keys).
        y_encoder: Optional MLP over ``(y, density)``; ``None`` passes the raw
            pair through (``d_y = 2``).
        vi_view: Name of the VI view (default ``"main"``).
        weather_view: Name of the weather view (default ``"weather"``).
    """

    def __init__(
        self,
        vi_encoder: BaseSetEncoder,
        fuse_x: nn.Module,
        token_proj: nn.Module,
        d_model: int,
        weather_encoder: BaseSetEncoder | None = None,
        geno_encoder: nn.Module | None = None,
        geno_key: str | Sequence[str] = "genomic_add",
        y_encoder: nn.Module | None = None,
        vi_view: str = "main",
        weather_view: str = "weather",
    ):
        super().__init__()
        self.vi_encoder = vi_encoder
        self.weather_encoder = weather_encoder
        self.geno_encoder = geno_encoder
        self.geno_key = geno_key
        self.fuse_x = fuse_x
        self.y_encoder = y_encoder
        self.token_proj = token_proj
        self.d_model = d_model
        self.vi_view = vi_view
        self.weather_view = weather_view

    def _encode_view(self, view, encoder: BaseSetEncoder) -> torch.Tensor:
        """Run a per-modality set encoder over a ``[B, n, T, ...]`` view."""
        coords = einops.rearrange(view.coords, "b n t d -> (b n) t d")
        channels = einops.rearrange(view.channels, "b n t c -> (b n) t c")
        pad_mask = einops.rearrange(view.pad_mask, "b n t -> (b n) t")
        return encoder(coords, channels, mask=pad_mask)

    def _gather_geno(self, task) -> torch.Tensor:
        """Flatten and concat the configured geno feature key(s) to ``[B*n, F]``."""
        keys = (
            [self.geno_key]
            if isinstance(self.geno_key, str)
            else list(self.geno_key)
        )
        feats = [
            einops.rearrange(task.derived_features[k], "b n f -> (b n) f")
            for k in keys
        ]
        return torch.cat(feats, dim=-1)

    def forward(self, task, *, is_context: bool) -> torch.Tensor:
        """Tokenize one canonical task batch.

        Args:
            task: A ``G2FBatch`` whose tensors carry a leading ``(task, sample)``
                pair — ``views[v].coords [B, n, T, D]`` etc., ``s [B, n]``.
            is_context: ``True`` => observed ``y`` + density 0; ``False`` =>
                missing ``y`` (zeroed) + density 1.

        Returns:
            One token per sample, ``[B, n, d_model]``.
        """
        main = task.views[self.vi_view]
        batch_size = main.coords.shape[0]

        parts = [self._encode_view(main, self.vi_encoder)]
        if self.weather_encoder is not None:
            parts.append(
                self._encode_view(
                    task.views[self.weather_view], self.weather_encoder
                )
            )
        if self.geno_encoder is not None:
            parts.append(self.geno_encoder(self._gather_geno(task)))

        x = self.fuse_x(torch.cat(parts, dim=-1))  # [B*n, d_x]

        if is_context:
            # task.s is [B, n] (scalar yield) or [B, n, 1] (collated yield) —
            # flatten the (task, sample) pair to [B*n, 1] either way.
            s = task.s
            if s.ndim == 2:
                y = einops.rearrange(s, "b n -> (b n) 1")  # observed
            else:
                y = einops.rearrange(s, "b n d -> (b n) d")
            density = x.new_zeros((x.shape[0], 1))  # density polarity: 0 = context
        else:
            y = x.new_zeros((x.shape[0], 1))  # y missing
            density = x.new_ones((x.shape[0], 1))  # density polarity: 1 = query

        y_in = torch.cat([y, density], dim=-1)  # density rides with y
        y_enc = self.y_encoder(y_in) if self.y_encoder is not None else y_in

        tok = self.token_proj(torch.cat([x, y_enc], dim=-1))  # [B*n, d_model]
        return einops.rearrange(tok, "(b n) d -> b n d", b=batch_size)


class TransformerNeuralProcess(nn.Module):
    """TNP yield model: tokenize -> cross-sample encode -> decode -> likelihood.

    The full/pseudo-token attention variant is selected purely by **which**
    ``transformer_encoder`` is instantiated (config-selected) — there is
    no ``mode`` flag in the model.

    Args:
        tokenizer: :class:`SampleTokenizer`.
        transformer_encoder: A :mod:`set_func.core.np` encoder
            (``EfficientQueryTransformerEncoder`` default | ``ISTransformerEncoder``
            | ``PerceiverEncoder``).
        decoder: ``d_model -> 1`` (homoscedastic) or ``2`` (heteroscedastic).
        head: Maps the decoder output to a predictive ``Prediction``.
    """

    def __init__(
        self,
        tokenizer: SampleTokenizer,
        transformer_encoder: BaseTransformerEncoder,
        decoder: BaseDecoder,
        head: BaseHead,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer_encoder = transformer_encoder
        self.decoder = decoder
        self.head = head

    def forward(self, batch) -> Prediction:
        """Predict the query yields conditioned on the context.

        Args:
            batch: An ``NPTaskBatch`` with canonical ``[B, n, ...]`` ``context``
                and ``query`` ``G2FBatch``es.

        Returns:
            A :class:`Prediction` over the ``B * nq`` query yields (leading axes
            ``[B, nq, ...]`` preserved).
        """
        zc = self.tokenizer(batch.context, is_context=True)  # [B, nc, d]
        zq = self.tokenizer(batch.query, is_context=False)  # [B, nq, d]
        zq_out = self.transformer_encoder(zc, zq)  # [B, nq, d]
        out = self.decoder(zq_out)  # [B, nq, 1 or 2]
        return self.head(out)
