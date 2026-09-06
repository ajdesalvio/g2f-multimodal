import copy

import torch.nn as nn


def get_clones(
    module: nn.Module,
    n: int,
) -> nn.ModuleList:
    """Create N identical layers.
    Args:
        module (nn.Module): Module to clone.
        n (int): Number of clones.
    Returns:
        nn.ModuleList: List of cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
