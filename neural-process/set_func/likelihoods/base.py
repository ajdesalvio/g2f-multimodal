from abc import abstractmethod

import torch
import torch.nn.functional as F

from set_func.heads.base import BaseHead
from set_func.predictions import Prediction

# Custom type hint for transform configurations
TransformConfig = str | dict | list


class BaseLikelihood(BaseHead):
    """A distributional head: maps a decoder output to a distributional
    :class:`~set_func.predictions.Prediction` (one carrying a density).

    Subclasses build a concrete distribution (e.g. a ``Normal``) and wrap it in
    the matching :class:`Prediction` so the model output is uniform across
    distributional and point heads.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_transform_config(self, transform_config: TransformConfig):
        """
        Parse transformation configuration that can be a string, dict, or list.

        Parameters
        ----------
        transform_config : TransformConfig
            - str: transformation name
            - dict: {transformation_name: parameters}
            - list: sequence of transformations (each can be str or dict)

        Returns
        -------
        list
            List of (transform_name, transform_params) tuples
        """
        if isinstance(transform_config, str):
            return [(transform_config, {})]
        elif isinstance(transform_config, dict):
            if len(transform_config) != 1:
                raise ValueError(
                    "Transform config dict must have exactly one key"
                )
            transform_name = list(transform_config.keys())[0]
            transform_params = transform_config[transform_name] or {}
            return [(transform_name, transform_params)]
        elif isinstance(transform_config, list):
            parsed_transforms = []
            for config in transform_config:
                parsed_transforms.extend(self._parse_transform_config(config))
            return parsed_transforms
        else:
            raise ValueError(
                f"Transform config must be str, dict, or list, got {type(transform_config)}"
            )

    def _apply_transform(
        self, x: torch.Tensor, transform_config: TransformConfig
    ):
        """
        Apply transformation(s) to tensor with optional parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        transform_config : TransformConfig
            Transform configuration(s)

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        transform_sequence = self._parse_transform_config(transform_config)
        result = x
        for transform_name, transform_params in transform_sequence:
            transform_fn = getattr(self, f"_{transform_name}")
            result = transform_fn(result, **transform_params)
        return result

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        """Identity transformation - returns input unchanged."""
        return x

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid transformation - maps to (0, 1)."""
        return torch.sigmoid(x)

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        """Softplus transformation - maps to (0, inf)."""
        return F.softplus(x)

    def _exp(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential transformation - maps to (0, inf)."""
        return torch.exp(x)

    def _tanh(self, x: torch.Tensor) -> torch.Tensor:
        """Tanh transformation - maps to (-1, 1)."""
        return torch.tanh(x)

    def _relu(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU transformation - maps to [0, inf)."""
        return F.relu(x)

    def _abs(self, x: torch.Tensor) -> torch.Tensor:
        """Absolute value transformation - maps to [0, inf)."""
        return torch.abs(x)

    def _clamp(
        self,
        x: torch.Tensor,
        min: float | None = None,
        max: float | None = None,
    ) -> torch.Tensor:
        """Clamp to [min, max] range."""
        return torch.clamp(x, min, max)

    def _add(self, x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        """Addition transformation - adds a constant value."""
        return x + value

    def _multiply(self, x: torch.Tensor, value: float = 1.0) -> torch.Tensor:
        """Multiplication transformation - multiplies by a constant value."""
        return x * value

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Prediction:
        pass
