"""Region definitions for mRNN.

Provides base :class:`Region` and concrete :class:`RecurrentRegion` and
:class:`InputRegion` containers that own connection parameters and masks.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Self

DEFAULT_REGION_BASE = {"sign": "pos", "device": "cuda"}

DEFAULT_REC_REGIONS = {
    # Name of the region
    "name": "region_",
    # Initial condition of region (xn)
    "init": 0,
    # Whether connection is excitatory or inhibitory
    "sign": "pos",
    # bias or baseline firing of region
    "base_firing": 0,
    # Parent class or family region belongs to (for cell types)
    "parent_region": None,
    # Whether or not the base firing will be trainable
    "learnable_bias": False,
    # device
    "device": "cuda",
}

DEFAULT_CONNECTIONS = {
    # How sparse the connections will be (float from 0-1)
    "sparsity": None,
    # How sparse the connections will be (float from 0-1)
    "zero_connection": False,
}


@dataclass
class Connection:
    parameter: torch.Tensor | None = None
    weight_mask: torch.Tensor | None = None
    sign_matrix: torch.Tensor | None = None
    zero_connection: torch.Tensor | None = None


class Region(nn.Module):
    """
    Base class for regions used by mRNN.

    Models outgoing connections to other regions along with simple region
    properties. Each region maintains its own connection parameters and masks
    (including sign masks enforcing Dale's Law when used by mRNN).

    Attributes:
        num_units (int): Number of units in the region.
        sign (str): "pos" for excitatory or "neg" for inhibitory outputs.
        device (str): Torch device string for tensors.
        connections (dict): Mapping of destination region name -> dict with
            keys {"parameter", "weight_mask", "sign_matrix", "zero_connection"}.
        masks (dict): Convenience masks including "ones" and "zeros" of length ``num_units``.
    """

    def __init__(
        self,
        num_units: int,
        sign: str = DEFAULT_REGION_BASE["sign"],
        device: str = DEFAULT_REGION_BASE["device"],
    ):
        """Construct a region.

        Args:
            num_units (int): Number of units in this region.
            sign (str): "pos" for excitatory or "neg" for inhibitory outputs.
            device (str): Torch device string (e.g., "cpu" or "cuda").
        """
        super(Region, self).__init__()

        self.num_units = num_units
        self.sign = sign
        self.device = device

        self.connections = {}
        self.masks = {}

        self._generate_masks()

    def __setitem__(self, idx: str | int, connection: Connection):
        """
        Implements the assignment operator

        connection_info must be of the same structure as shown in add_connection
        Usage:
            cur_region[idx] = new_connection_info
        """
        if isinstance(idx, int):
            # Get the current indexed connection
            connections_list = list(self.connections.keys())
            # Assign it override connections in idx_region
            cur_connection = connections_list[idx]
            self.connections[cur_connection] = connection
        elif isinstance(idx, str):
            # Directly access connection information
            self.connections[idx] = connection
        else:
            raise Exception("Improper indexing type")

    def __getitem__(self, idx: str | int) -> Connection:
        """
        Indexes into the connection properties of region, using dict order

        Usage:
            region_sub = region[idx]
        """

        if isinstance(idx, int):
            # Get the current indexed connection
            connections_list = list(self.connections.keys())
            # Assign it override connections in idx_region
            cur_connection = connections_list[idx]
            return self.connections[cur_connection]
        elif isinstance(idx, str):
            # Directly access connection information
            return self.connections[idx]
        else:
            raise Exception("Improper indexing type")

    def add_connection(
        self,
        dst_region_name: str,
        dst_region_units: int,
        sparsity: float | None = DEFAULT_CONNECTIONS["sparsity"],
        zero_connection: bool = DEFAULT_CONNECTIONS["zero_connection"],
    ):
        """Add a connection from this region to ``dst_region``.

        Creates a trainable weight parameter and associated non-trainable masks.
        If ``sparsity`` is provided, a binary mask is sampled to achieve the
        requested sparsity.

        Args:
            dst_region_name (str): Name of the destination region.
            dst_region_units (int): Number of units in destination region
            sparsity (float | None): Fraction of nonzero connections (0-1).
            zero_connection (bool): If True, registers a fixed zero connection
                (no trainable parameters are created for this edge).
        """

        """
        Check to make sure users are not duplicating connections
        Currently, this may lead to complications since the parameter will not be registered if it already exists
        Additionally, users will likely never need to duplicate a connection and this may signal an error on their part
        Therefore this can also act as a check to ensure proper connectivity is maintained
        Only raise this exception when the connection is not a zero connection
        """

        if dst_region_name in self.connections:
            if self.connections[dst_region_name]["zero_connection"] is False:
                raise Exception("Connection is already registered as parameter")
        """
        connections should only be made from:
            1. recurrent region -> recurrent region
            2. input region -> recurrent region
        """
        # Store all connection parameters in this dataclass
        connection_properties = Connection()

        # Initialize connection parameters
        parameter = torch.zeros(
            size=(dst_region_units, self.num_units), device=self.device
        )
        # Even though parameter is zero use this specifically to zero out the weight and sign matrix if zero connection
        # This is just to be extra safe to ensure everything about this connection is zero
        zero_con_mask = torch.zeros_like(parameter)

        # Initialize sparse mask if sparsity is given
        if sparsity is not None:
            weight_mask = torch.empty_like(parameter, device=self.device)
            nn.init.sparse_(weight_mask, sparsity)
            weight_mask[weight_mask != 0] = 1
        else:
            weight_mask = torch.ones_like(parameter, device=self.device)

        # Adjust the sign matrix for inhibitory connections
        if self.sign == "pos":
            sign_matrix = weight_mask
        elif self.sign == "neg":
            sign_matrix = -weight_mask
        else:
            raise ValueError("sign can only be (pos) or (neg)")

        """ In the case of zero connection, everything from parameter, mask, and sign will be zero in
            connections_dict. Additionally, they won't be registered as parameters, so nothing will be 
            initialized in the mRNN class either. This should ensure everything is always zero.
        """

        if zero_connection:
            weight_mask *= zero_con_mask
            sign_matrix *= zero_con_mask

        # Store weight mask and sign matrix
        # Store trainable parameter
        connection_properties.parameter = parameter
        connection_properties.weight_mask = weight_mask
        connection_properties.sign_matrix = sign_matrix
        connection_properties.zero_connection = zero_connection

        # Add all of the properties to define the connection in Region class
        self.connections[dst_region_name] = connection_properties

    def has_connection_to(self, region: str) -> bool:
        """
        Checks if there is a connection from the current region to the specified region.

        Args:
            region (str): Name of the region to check for connection.
        Returns:
            bool: True if there is a connection, otherwise False.
        """
        return region in self.connections

    def _generate_masks(self):
        """Generate reusable full and zero masks for this region."""
        full_mask = torch.ones(size=(self.num_units,)).to(self.device)
        zero_mask = torch.zeros(size=(self.num_units,)).to(self.device)

        self.masks["ones"] = full_mask
        self.masks["zeros"] = zero_mask
