import torch

from ..tokenizers.set import encode_set
from .base import BaseSetEncoder


class TransformerSetEncoder(BaseSetEncoder):
    """
    Set encoder using a Transformer encoder with an aggregation function.

    The transformer operates on the set elements before aggregation, applying
    self-attention to capture interactions between elements.
    """

    @staticmethod
    def padding_mask_to_attention_mask(
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts a padding mask to an attention mask for self-attention.

        Only masks based on key positions so that padding queries still attend
        to valid keys (avoiding NaN from all-masked softmax rows). The padding
        query outputs are harmless since they get excluded during aggregation.

        Args:
            padding_mask (torch.Tensor): Binary mask of shape (batch, set_size) where
                True indicates padding and False indicates valid elements.

        Returns:
            torch.Tensor: Attention mask of shape (batch, 1, set_size) where
                True indicates valid key positions (can be attended to).
                Broadcasts over the query dimension.
        """
        # Invert: True=padding -> False; False=valid -> True (attend)
        # Unsqueeze to (batch, 1, set_size) to broadcast over queries
        return (~padding_mask).unsqueeze(1)

    def forward(
        self,
        coords: torch.Tensor,
        channels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes a set view (coords, channels) using Transformer architecture.

        Pipeline:
        1. Encode coords and channels separately (optional)
        2. Concatenate encoded features
        3. Apply xy_encoder (optional)
        4. Apply transformer (latent_encoder) with self-attention on set elements
        5. Aggregate across the set
        6. Apply output_encoder (optional) to aggregated representation

        Args:
            coords (torch.Tensor): Index/metric axis, shape (batch, set_size, D).
            channels (torch.Tensor): Signal values, shape (batch, set_size, C).
            mask (Optional[torch.Tensor]): Binary mask of shape (batch, set_size)
                where True indicates padding elements. This will be converted to
                the appropriate attention mask format for the transformer and
                used to exclude padding from aggregation.

        Returns:
            torch.Tensor: Aggregated scalar representation of shape (batch, latent_dim).
        """
        # Step 1-3: Encode coords, channels and concatenate. Delegated to the
        # shared `encode_set` helper so there is a single tokenization
        # implementation (no drift).
        xy_embedded = encode_set(
            self.x_encoder, self.y_encoder, self.xy_encoder, coords, channels
        )

        # Step 4: Apply transformer (set-level encoding with self-attention)
        # Convert padding mask to attention mask if needed
        if mask is not None:
            attention_mask = self.padding_mask_to_attention_mask(mask)
        else:
            attention_mask = None

        latent = self.latent_encoder(xy_embedded, mask=attention_mask)

        # Step 5: Aggregate across set
        aggregated = self.aggregator(latent, mask=mask)

        # Step 6: Apply output encoder (post-aggregation)
        output = self.output_encoder(aggregated)

        return output
