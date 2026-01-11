from mrnntorch.region.region_base import Region, DEFAULT_REGION_BASE


class InputRegion(Region):
    def __init__(
        self,
        num_units,
        sign=DEFAULT_REGION_BASE["sign"],
        device=DEFAULT_REGION_BASE["device"],
    ):
        # Implements base region class
        super(InputRegion, self).__init__(num_units, sign=sign, device=device)
        """Input region (inherits from :class:`Region`).

        Args:
            num_units (int): Number of input channels.
            sign (str): "pos" or "neg" indicating sign mask for inputs.
            device (str): Torch device string.
        """
