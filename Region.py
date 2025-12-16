"""Region definitions for mRNN.

Provides base :class:`Region` and concrete :class:`RecurrentRegion` and
:class:`InputRegion` containers that own connection parameters and masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import json


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

    def __init__(self, num_units, sign="pos", device="cuda"):
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

        self.__generate_masks()

    def add_connection(
        self,
        dst_region_name,
        dst_region,
        sparsity,
        zero_connection=False,
    ):
        """Add a connection from this region to ``dst_region``.

        Creates a trainable weight parameter and associated non-trainable masks.
        If ``sparsity`` is provided, a binary mask is sampled to achieve the
        requested sparsity.

        Args:
            dst_region_name (str): Name of the destination region.
            dst_region (Region): Destination region object.
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
        Here we will assert that users are only making connections from:
            1. recurrent region -> recurrent region
            2. input region -> recurrent region
        """
        self.__assert_projection_type(dst_region)
        # Store all connection parameters in this dictionary
        connection_properties = {}

        # Initialize connection parameters
        parameter = torch.zeros(
            size=(dst_region.num_units, self.num_units), device=self.device
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
        connection_properties["parameter"] = parameter
        connection_properties["weight_mask"] = weight_mask
        connection_properties["sign_matrix"] = sign_matrix
        connection_properties["zero_connection"] = zero_connection

        # Add all of the properties to define the connection in Region class
        self.connections[dst_region_name] = connection_properties

    def __generate_masks(self):
        """Generate reusable full and zero masks for this region."""
        full_mask = torch.ones(size=(self.num_units,)).to(self.device)
        zero_mask = torch.zeros(size=(self.num_units,)).to(self.device)

        self.masks["ones"] = full_mask
        self.masks["zeros"] = zero_mask

    def __assert_projection_type(self, dst_region):
        """Ensure that projections only target recurrent regions."""
        assert isinstance(dst_region, RecurrentRegion)

    def has_connection_to(self, region):
        """
        Checks if there is a connection from the current region to the specified region.

        Args:
            region (str): Name of the region to check for connection.
        Returns:
            bool: True if there is a connection, otherwise False.
        """
        return region in self.connections


#############################################################################################################
# Recurrent Region


class RecurrentRegion(Region):
    def __init__(
        self,
        num_units,
        base_firing,
        init,
        sign="pos",
        device="cuda",
        parent_region=None,
        learnable_bias=False,
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


#############################################################################################################
# Input Region


class InputRegion(Region):
    def __init__(self, num_units, sign="pos", device="cuda"):
        # Implements base region class
        super(InputRegion, self).__init__(num_units, sign=sign, device=device)
        """Input region (inherits from :class:`Region`).

        Args:
            num_units (int): Number of input channels.
            sign (str): "pos" or "neg" indicating sign mask for inputs.
            device (str): Torch device string.
        """
