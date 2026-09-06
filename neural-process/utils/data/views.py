"""View spec + assembly: turn a flat sample dict into a coords/channels pair.

A :class:`View` is the config-level description of *which* per-sample columns
play the coordinate axes (``coords``) vs. the signal channels (``channels``),
plus an axis transform to apply. :func:`assemble_view` realises it against one
sample dict, producing a :class:`ViewArrays` (coords ``[T,D]`` / channels
``[T,C]`` + names + the per-cell mask) that both the DL collate path and the
FPCA R-bridge consume.

Channel/coord selectors (resolved by :func:`resolve_channels`):
- ``"vi.*"`` / ``"vi:*"`` — every channel in the sample's ``channel_names``.
- ``"vi:<name>"`` — one named VI channel.
- ``"weather.*"`` / ``"weather:*"`` — every weather variable in the sample's
  ``weather_channel_names`` (the ``raw_weather`` curve matrix
  ``sample["weather_values"]``); ``"weather:<name>"`` selects one.
- an **axis name** (``dap`` / ``gdd`` / ``agdd`` / ``weather_dap``) — use that
  axis *as a channel* (the "extra feature" / separate-FPCA strategy).
- a bare name present in ``channel_names`` — treated as that VI channel.

Coordinates are always axis names (resolved against the axes attached to the
sample). The VI views read the VI grid (``dap``/``gdd``/``agdd`` coords +
``channels`` matrix); the weather view reads the independent daily weather grid
(``weather_dap`` coord + ``weather_values`` matrix) — the minimal
namespace extension that folds the weather peer into the View path.

The mask (reader's ``vi_nan_mask``, ``True`` = originally-missing) is carried
into ``ViewArrays.mask`` so transforms move it in lockstep (G3); axis columns
used as channels and weather channels are never missing (mask ``False``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .axis_transforms import ViewArrays, get_transform

# Axis names that may serve as coordinates (or as channels). The VI-grid axes
# (``dap``/``gdd``/``agdd``) are duplicated from
# ``processing.axis_source.AXIS_NAMES`` to avoid an import cycle (views is
# imported by dataset/collate, which axis_source must not depend on).
# ``weather_dap`` is the independent daily weather grid attached by
# ``raw_weather`` — a coord for the ``weather`` view.
# ``weather_gdd`` / ``weather_agdd`` are the gdd/agdd time axes on that same
# daily weather grid (un-deduped, raw cumsum), attached by ``raw_weather``'s
# ``extra_axes`` knob so the weather set-encoder can take them as extra
# channels (or coords) — the weather-grid analog of the VI-grid gdd/agdd.
AXIS_NAMES: tuple[str, ...] = (
    "dap", "gdd", "agdd", "weather_dap", "weather_gdd", "weather_agdd",
)

# Targets a view can feed.
VIEW_TARGETS: tuple[str, ...] = ("dl", "fpca")


@dataclass
class View:
    """A named coords/channels selection + transform.

    Parameters
    ----------
    name : view name (e.g. ``"main"`` / ``"weather"`` / ``"agdd"``).
    coords : ordered list of axis names forming the coordinate columns.
    channels : ordered list of channel selectors (see module docstring).
    transform : registered axis-transform name (default ``"identity"``).
    transform_args : kwargs passed to the transform constructor.
    target : ``"dl"`` or ``"fpca"`` — which consumer this view feeds.
    """

    name: str
    coords: list[str]
    channels: list[str]
    transform: str = "identity"
    transform_args: dict[str, Any] = field(default_factory=dict)
    target: str = "dl"

    def __post_init__(self) -> None:
        if not self.coords:
            raise ValueError(f"view {self.name!r}: coords must be non-empty.")
        if not self.channels:
            raise ValueError(
                f"view {self.name!r}: channels must be non-empty."
            )
        if self.target not in VIEW_TARGETS:
            raise ValueError(
                f"view {self.name!r}: target must be one of {VIEW_TARGETS}, "
                f"got {self.target!r}."
            )

    @classmethod
    def from_config(cls, name: str, cfg: Any) -> "View":
        """Build a :class:`View` from a (DictConfig-like) mapping.

        ``cfg`` keys: ``coords``, ``channels`` (required); ``transform``,
        ``transform_args``, ``target`` (optional).
        """
        def _get(key, default):
            if hasattr(cfg, "get"):
                val = cfg.get(key, default)
            else:
                val = getattr(cfg, key, default)
            return val

        coords = _get("coords", None)
        channels = _get("channels", None)
        if coords is None or channels is None:
            raise ValueError(
                f"view {name!r}: config must define 'coords' and 'channels'."
            )
        ta = _get("transform_args", None)
        return cls(
            name=name,
            coords=list(coords),
            channels=list(channels),
            transform=str(_get("transform", "identity")),
            transform_args=dict(ta) if ta else {},
            target=str(_get("target", "dl")),
        )


def resolve_channels(
    selectors: list[str],
    channel_names: list[str],
    axis_names: list[str],
    weather_channel_names: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Expand channel selectors into ordered ``(kind, name)`` pairs.

    ``kind`` is ``"vi"`` (slice from the sample's ``channels`` matrix),
    ``"weather"`` (slice from the sample's ``weather_values`` matrix), or
    ``"axis"`` (read the sample's axis column ``sample[name]``). Raises
    ``ValueError`` on an unknown selector or a glob that matches nothing.
    """
    weather_channel_names = weather_channel_names or []
    resolved: list[tuple[str, str]] = []
    for sel in selectors:
        if sel in ("vi.*", "vi:*"):
            if not channel_names:
                raise ValueError(
                    f"selector {sel!r} matched no channels (sample has no "
                    f"channel_names)."
                )
            resolved.extend(("vi", n) for n in channel_names)
        elif sel in ("weather.*", "weather:*"):
            if not weather_channel_names:
                raise ValueError(
                    f"selector {sel!r} matched no weather channels (sample "
                    f"has no weather_channel_names — is raw_weather enabled?)."
                )
            resolved.extend(("weather", n) for n in weather_channel_names)
        elif sel.startswith("vi:"):
            name = sel[len("vi:"):]
            if name not in channel_names:
                raise ValueError(
                    f"selector {sel!r}: VI {name!r} not in channel_names "
                    f"{channel_names}."
                )
            resolved.append(("vi", name))
        elif sel.startswith("weather:"):
            name = sel[len("weather:"):]
            if name not in weather_channel_names:
                raise ValueError(
                    f"selector {sel!r}: weather var {name!r} not in "
                    f"weather_channel_names {weather_channel_names}."
                )
            resolved.append(("weather", name))
        elif sel in axis_names:
            resolved.append(("axis", sel))
        elif sel in channel_names:
            resolved.append(("vi", sel))
        else:
            raise ValueError(
                f"channel selector {sel!r} resolves to nothing: not a glob, "
                f"not an axis {axis_names}, not a VI {channel_names}, not a "
                f"weather var {weather_channel_names}."
            )
    return resolved


def _axes_on_sample(sample: dict[str, Any]) -> list[str]:
    """Axis names present on the sample (always ``dap``; ``gdd``/``agdd`` when
    attached by ``axis_source``)."""
    return [a for a in AXIS_NAMES if a in sample]


def _column(sample: dict[str, Any], name: str) -> torch.Tensor:
    """A ``[T, 1]`` axis column from the sample (handles ``[T]`` or ``[T,1]``)."""
    col = sample[name]
    if col.ndim == 1:
        col = col.unsqueeze(-1)
    return col


def assemble_view(sample: dict[str, Any], view: View) -> ViewArrays:
    """Realise ``view`` against one sample dict → :class:`ViewArrays`.

    Selects the coordinate axes and channel columns by name, carries the
    per-cell ``vi_nan_mask`` (axis-as-channel columns get ``False``), then
    applies the view's axis transform. Coordinates may be any width ``D >= 1``;
    TE encoders consume multi-axis coords via D-agnostic pairwise differences.
    """
    axis_names = _axes_on_sample(sample)

    # ── Coordinates ────────────────────────────────────────────────
    coord_cols: list[torch.Tensor] = []
    for name in view.coords:
        if name not in axis_names:
            raise KeyError(
                f"view {view.name!r}: coord {name!r} not available on sample "
                f"(axes present: {axis_names}). Enable the axis_source "
                f"processor to attach gdd/agdd."
            )
        coord_cols.append(_column(sample, name))
    coords = torch.cat(coord_cols, dim=1)

    # ── Channels (+ mask) ──────────────────────────────────────────
    channel_names_full = list(sample.get("channel_names") or [])
    weather_channel_names = list(sample.get("weather_channel_names") or [])
    resolved = resolve_channels(
        view.channels, channel_names_full, axis_names, weather_channel_names
    )

    vi_matrix = sample["channels"]
    if vi_matrix.ndim == 1:
        vi_matrix = vi_matrix.unsqueeze(-1)
    weather_matrix = sample.get("weather_values")
    if weather_matrix is not None and weather_matrix.ndim == 1:
        weather_matrix = weather_matrix.unsqueeze(-1)
    vi_mask = sample.get("vi_nan_mask")
    has_mask = vi_mask is not None

    chan_cols: list[torch.Tensor] = []
    mask_cols: list[torch.Tensor] = []
    out_channel_names: list[str] = []
    T = coords.shape[0]
    for kind, name in resolved:
        if kind == "vi":
            idx = channel_names_full.index(name)
            chan_cols.append(vi_matrix[:, idx:idx + 1])
            if has_mask:
                mask_cols.append(vi_mask[:, idx:idx + 1])
            out_channel_names.append(name)
        elif kind == "weather":  # weather curve column — never missing
            idx = weather_channel_names.index(name)
            chan_cols.append(weather_matrix[:, idx:idx + 1])
            if has_mask:
                mask_cols.append(torch.zeros((T, 1), dtype=torch.bool))
            out_channel_names.append(name)
        else:  # axis-as-channel — never missing
            chan_cols.append(_column(sample, name))
            if has_mask:
                mask_cols.append(
                    torch.zeros((T, 1), dtype=torch.bool)
                )
            out_channel_names.append(name)

    channels = torch.cat(chan_cols, dim=1)
    mask = torch.cat(mask_cols, dim=1) if has_mask else None

    arrays = ViewArrays(
        coords=coords,
        channels=channels,
        coord_names=list(view.coords),
        channel_names=out_channel_names,
        mask=mask,
    )

    # ── Transform ──────────────────────────────────────────────────
    arrays = get_transform(view.transform, **view.transform_args).apply(arrays)

    return arrays
