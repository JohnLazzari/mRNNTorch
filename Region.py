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
    A class representing a region in a neural network that models connections 
    to other regions along with other properties such as cell types and firing rates.
    
    Attributes:
        num_units (int): Number of neurons in the region.
        base_firing (torch.Tensor): Baseline firing rate for each neuron in the region.
        device (torch.device): The device on which to store the tensors (e.g., 'cpu' or 'cuda').
        cell_type_info (dict): Dictionary specifying each cell type and the proportion of neurons each type occupies.
        connections (dict): Dictionary to store the connections to other regions.
        masks (dict): Masks for each cell type and region properties (e.g., full mask, zero mask).
    """
    
    def __init__(self, num_units, sign="pos", device="cuda"):
        """
        Initializes the Region class.
        
        Args:
            num_units (int): Number of neurons in the region.
            sign (str): sign of weights, should be None, exc, or inhib
            device (torch.device): The device ('cpu' or 'cuda').
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
        """
        Adds a connection from the current region to a specified projection region.
        
        Args:
            dst_region_name (str):                  Name of the region that the current region connects to.
            dst_region (Region):                    The target region to which the connection is made.
            sign (str):                             Specifies if the connection is excitatory or inhibitory ('inhib' for inhibitory).
            sparsity (float):                       Specifies how sparse the connections are (defualts to none otherwise)
            zero_connection (bool, optional):       If True, no connections are created (default is False).
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
        parameter = torch.zeros(size=(dst_region.num_units, self.num_units), device=self.device)
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
        connection_properties["parameter"] = nn.Parameter(parameter) if not zero_connection else parameter
        connection_properties["weight_mask"] = nn.Parameter(weight_mask, requires_grad=False) if not zero_connection else weight_mask
        connection_properties["sign_matrix"] = nn.Parameter(sign_matrix, requires_grad=False) if not zero_connection else sign_matrix
        connection_properties["zero_connection"] = zero_connection

        # Add all of the properties to define the connection in Region class
        self.connections[dst_region_name] = connection_properties

        # Manually register parameters if not a zero connection
        if not zero_connection:
            # Register the main parameter denoting weights of model
            # Next, register the mask and sign matrix as parameters to ensure they're loaded in the same
            self.register_parameter(dst_region_name, self.connections[dst_region_name]["parameter"])
            self.register_parameter(f"{dst_region_name}_weight_mask", self.connections[dst_region_name]["weight_mask"])
            self.register_parameter(f"{dst_region_name}_sign_matrix", self.connections[dst_region_name]["sign_matrix"])

    def __generate_masks(self):
        """
        Generates masks for the region, including full and zero masks, and specific cell-type masks.
        """
        full_mask = torch.ones(size=(self.num_units,)).to(self.device)
        zero_mask = torch.zeros(size=(self.num_units,)).to(self.device)

        self.masks["ones"] = full_mask
        self.masks["zeros"] = zero_mask

    def __assert_projection_type(self, dst_region):
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
    def __init__(self, num_units, base_firing, init, sign="pos", device="cuda", parent_region=None, learnable_bias=False):
        super(RecurrentRegion, self).__init__(num_units, sign=sign, device=device)
        """ Recurrent Region Class (mostly inherits from base Region class)
        
            Params:
                num_units:
                base_firing:
                init:
                device:
                parent_region:
                learnable_bias:
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