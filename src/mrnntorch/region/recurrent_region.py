import torch
import torch.nn as nn
from mrnntorch.region.region_base import Region, DEFAULT_REC_REGIONS


class RecurrentRegion(Region):
    def __init__(
        self,
        num_units,
        base_firing=DEFAULT_REC_REGIONS["base_firing"],
        init=DEFAULT_REC_REGIONS["init"],
        sign=DEFAULT_REC_REGIONS["sign"],
        parent_region=DEFAULT_REC_REGIONS["parent_region"],
        learnable_bias=DEFAULT_REC_REGIONS["learnable_bias"],
        device=DEFAULT_REC_REGIONS["device"],
    ):
        super(RecurrentRegion, self).__init__(num_units, sign=sign, device=device)
        """Recurrent region (inherits from :class:`Region`).

        Args:
            num_units (int): Number of units in the region.
            base_firing (float): Baseline firing for each unit.
            init (float): Initial pre-activation value for units.
            sign (str): "pos" or "neg" indicating excitatory/inhibitory outputs.
            device (str): Torch device string.
            parent_region (str | None): Optional parent identifier.
            learnable_bias (bool): If True, make ``base_firing`` a trainable parameter.
        """

        self.init = init * torch.ones(size=(self.num_units,))
        self.learnable_bias = learnable_bias
        self.parent_region = parent_region

        if learnable_bias is True:
            self.base_firing = nn.Parameter(base_firing * torch.ones(size=(num_units,)))
        else:
            self.base_firing = base_firing * torch.ones(size=(num_units,))
