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
    
    def __init__(self, num_units, base_firing, init, device, cell_types=None):
        """
        Initializes the Region class.
        
        Args:
            num_units (int): Number of neurons in the region.
            base_firing (float): Baseline firing rate for the region.
            device (torch.device): The device ('cpu' or 'cuda').
            cell_types (dict, optional): A dictionary specifying the proportions of different cell types in the region.
        """
        super(Region, self).__init__()

        self.num_units = num_units
        self.init = init * torch.ones(size=(self.num_units,))
        self.device = device
        self.base_firing = base_firing * torch.ones(size=(num_units,))
        self.cell_type_info = cell_types if cell_types is not None else {}
        self.connections = {}
        self.masks = {}

        self.__generate_masks()

    def add_connection(
        self, 
        dst_region_name, 
        dst_region, 
        src_region_cell_type, 
        dst_region_cell_type, 
        sign, 
        sparsity,
        zero_connection=False, 
    ):
        """
        Adds a connection from the current region to a specified projection region.
        
        Args:
            proj_region_name (str):                 Name of the region that the current region connects to.
            proj_region (Region):                   The target region to which the connection is made.
            src_region_cell_type (str):             The source region's cell type.
            dst_region_cell_type (str):             The destination region's cell type.
            sign (str):                             Specifies if the connection is excitatory or inhibitory ('inhib' for inhibitory).
            sparsity (float):                       Specifies how sparse the connections are (defualts to none otherwise)
            zero_connection (bool, optional):       If True, no connections are created (default is False).
            lower_bound (float, optional):          Lower bound for uniform weight initialization.
            upper_bound (float, optional):          Upper bound for uniform weight initialization.
        """
        connection_properties = {}

        # Initialize connection parameters
        if not zero_connection:
            parameter = nn.Parameter(torch.empty(size=(dst_region.num_units, self.num_units), device=self.device))
        else:
            parameter = torch.zeros(size=(dst_region.num_units, self.num_units), device=self.device)
        
        # Initialize sparse mask if sparsity is given
        if sparsity is not None:
            sparse_tensor = torch.empty_like(parameter, device=self.device)
            nn.init.sparse_(sparse_tensor, sparsity)
            sparse_tensor[sparse_tensor != 0] = 1

            sparse_tensor_src = sparse_tensor
            sparse_tensor_dst = sparse_tensor.T
        else:
            sparse_tensor_src = torch.ones_like(parameter, device=self.device)
            sparse_tensor_dst = torch.ones_like(parameter, device=self.device).T

        # Store trainable parameter
        connection_properties["parameter"] = parameter

        # Initialize connection tensors (1s for active connections, 0s for no connections)
        connection_tensor_src = torch.ones_like(parameter, device=self.device) if not zero_connection else torch.zeros_like(parameter, device=self.device)
        connection_tensor_dst = torch.ones_like(parameter, device=self.device).T if not zero_connection else torch.zeros_like(parameter, device=self.device).T

        # Create weight masks based on cell types, if specified
        weight_mask_src, sign_matrix_src = self.__get_weight_and_sign_matrices(src_region_cell_type, connection_tensor_src, sparse_tensor_src)
        weight_mask_dst, sign_matrix_dst = dst_region.__get_weight_and_sign_matrices(dst_region_cell_type, connection_tensor_dst, sparse_tensor_dst)

        # Combine masks
        # Transpose the dst matrices since they should correspond to row operations
        weight_mask = weight_mask_src * weight_mask_dst.T
        sign_matrix = sign_matrix_src * sign_matrix_dst.T

        # Adjust the sign matrix for inhibitory connections
        if sign == "inhib":
            sign_matrix *= -1
        elif sign is None:
            sign_matrix = torch.zeros_like(parameter).to(self.device)

        # Store weight mask and sign matrix
        connection_properties["weight_mask"] = weight_mask.to(self.device)
        connection_properties["sign_matrix"] = sign_matrix.to(self.device)

        # Update connections dictionary
        if dst_region_name in self.connections:
            # If current region already has connection with dst_region
            # Simply update the make to account for a new connection (such as to a new cell type)
            self.connections[dst_region_name]["weight_mask"] += connection_properties["weight_mask"]
            self.connections[dst_region_name]["sign_matrix"] += connection_properties["sign_matrix"]
        else:
            self.connections[dst_region_name] = connection_properties

            # Manually register parameters
            if not zero_connection:
                self.register_parameter(dst_region_name, self.connections[dst_region_name]["parameter"])

    def __generate_masks(self):
        """
        Generates masks for the region, including full and zero masks, and specific cell-type masks.
        """
        full_mask = torch.ones(size=(self.num_units,)).to(self.device)
        zero_mask = torch.zeros(size=(self.num_units,)).to(self.device)

        self.masks["full"] = full_mask
        self.masks["zero"] = zero_mask

        for key in self.cell_type_info:
            mask = self.__generate_cell_type_mask(key)
            self.masks[key] = mask.to(self.device)

    def __generate_cell_type_mask(self, key):
        """
        Generates a mask for a specific cell type based on its proportion in the region.

        Args:
            key (str): The cell type identifier.

        Returns:
            torch.Tensor: Mask for the specified cell type.
        """
        cur_masks = []
        for prev_key in self.cell_type_info:
            if prev_key == key:
                cur_masks.append(torch.ones(size=(int(round(self.num_units * self.cell_type_info[prev_key])),)))
            else:
                cur_masks.append(torch.zeros(size=(int(round(self.num_units * self.cell_type_info[prev_key])),)))
        mask = torch.cat(cur_masks)
        return mask

    def __get_weight_and_sign_matrices(self, cell_type, connection_tensor, sparse_tensor):
        """
        Retrieves the weight mask and sign matrix for a specified cell type.

        Args:
            cell_type (str): The cell type to generate the mask for.
            connection_tensor (torch.Tensor): Tensor indicating whether connections are active.

        Returns:
            tuple: weight mask and sign matrix.
        """
        if cell_type is not None:
            weight_mask = sparse_tensor * connection_tensor * self.masks.get(cell_type)
            sign_matrix = sparse_tensor * connection_tensor * self.masks.get(cell_type)
        else:
            weight_mask = sparse_tensor * connection_tensor
            sign_matrix = sparse_tensor * connection_tensor

        return weight_mask, sign_matrix

    def has_connection_to(self, region):
        """
        Checks if there is a connection from the current region to the specified region.
        
        Args:
            region (str): Name of the region to check for connection.
        
        Returns:
            bool: True if there is a connection, otherwise False.
        """
        return region in self.connections