import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import json
from mRNNTorch.Region import Region

def linear(x):
    return x

class mRNN(nn.Module):
    """
    A Multi-Regional Recurrent Neural Network (mRNN) that implements interactions between brain regions.
    This model is designed to simulate neural interactions between different brain areas, with support
    for region-specific properties and inter-regional connections.

    Key Features:
    - Supports multiple brain regions with distinct properties
    - Implements Dale's Law for biological plausibility
    - Handles region-specific cell types
    - Includes noise injection for both hidden states and inputs
    - Supports tonic (baseline) firing rates for each region

    Args:
        config (str): Path to JSON configuration file specifying network architecture
        inp_dim (int): Dimension of the input
        noise_level_act (float, optional): Noise level for activations. Defaults to 0.01
        noise_level_inp (float, optional): Noise level for inputs. Defaults to 0.01
        constrained (bool, optional): Whether to apply Dale's Law constraints. Defaults to True
        t_const (float, optional): Time constant for network dynamics. Defaults to 0.1
        device (str, optional): Computing device to use. Defaults to "cuda"
    """

    def __init__(
        self, 
        config,
        activation="relu",
        noise_level_act=0.01, 
        noise_level_inp=0.01, 
        constrained=True, 
        t_const=0.1,
        batch_first=True,
        lower_bound_rec=0,
        upper_bound_rec=10,
        lower_bound_inp=0,
        upper_bound_inp=10,
        device="cuda",
    ):
        super(mRNN, self).__init__()

        # TODO allow initialization through spectral radius

        # Initialize network parameters
        self.region_dict = {}
        self.inp_dict = {}
        self.region_mask_dict = {}
        self.constrained = constrained
        self.device = device
        self.t_const = t_const
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation
        self.lower_bound_rec = lower_bound_rec
        self.upper_bound_rec = upper_bound_rec
        self.lower_bound_inp = lower_bound_inp
        self.upper_bound_inp = upper_bound_inp
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "linear":
            self.activation == linear()
        else:
            raise Exception("Only relu, tanh, sigmoid, or linear activations are implemented")

        # Load and process configuration
        with open(config, 'r') as f:
            config = json.load(f)
        
        # Generate network structure
        self.__create_def_values(config)

        # Generate recurrent regions
        for region in config["recurrent_regions"]:
            self.region_dict[region["name"]] = Region(
                num_units=region["num_units"],
                base_firing=region["base_firing"],
                init=region["init"],
                device=self.device,
                cell_types=region["cell_types"]
            )

        # Generate recurrent connections
        for connection in config["recurrent_connections"]:
            self.region_dict[connection["src_region"]].add_connection(
                dst_region_name=connection["dst_region"],
                dst_region=self.region_dict[connection["dst_region"]],
                src_region_cell_type=connection["src_region_cell_type"],
                dst_region_cell_type=connection["dst_region_cell_type"],
                sign=connection["sign"],
                sparsity=connection["sparsity"],
            )

        # Generate input regions
        for inp in config["input_regions"]:
            self.inp_dict[inp["name"]] = Region(
                num_units=inp["num_units"],
                base_firing=0,
                init=0,
                device=self.device,
                cell_types={}
            )
        
        # Generate input connections
        for inp in config["input_connections"]:
            self.inp_dict[inp["input"]].add_connection(
                dst_region_name=inp["dst_region"],
                dst_region=self.region_dict[inp["dst_region"]],
                src_region_cell_type=None,
                dst_region_cell_type=inp["dst_region_cell_type"],
                sign=inp["sign"],
                sparsity=inp["sparsity"],
            )
        
        # This completes the connections matrix between regions 
        # By adding zeros where explicity connections are not specified.
        # Does so for both recurrent and input connections
        self.finalize_connectivity()
        
        # General network parameters
        self.total_num_units = self.__get_total_num_units(self.region_dict)
        self.total_num_inputs = self.__get_total_num_units(self.inp_dict)
        self.baseline_inp = self.__get_tonic_inp()

        # Register all parameters 
        for region in self.region_dict:
            for name, param in self.region_dict[region].named_parameters():
                self.register_parameter(f"{region}_{name}", param)
                # Default initialization for recurrent weights
                nn.init.uniform_(param, 0, np.sqrt(1 / (2*self.total_num_units)))

        for inp in self.inp_dict:
            for name, param in self.inp_dict[inp].named_parameters():
                self.register_parameter(f"{inp}_{name}", param)
                # Default initialization for inputs
                nn.init.uniform_(param, 0, np.sqrt(1 / (self.total_num_units + self.total_num_inputs)))

        # Get indices for specific regions
        for region in self.region_dict:
            # Get the mask for the whole region, regardless of cell type
            self.region_mask_dict[region] = {}
            self.region_mask_dict[region]["full"] = self.__gen_region_mask(region)
            # Loop through the cell type of each region if not empty
            for cell_type in self.region_dict[region].cell_type_info:
                # Generate a mask for the cell type in region_mask_dict
                self.region_mask_dict[region][cell_type] = self.__gen_region_mask(region, cell_type=cell_type)
    
    def finalize_connectivity(self):
        # Fill rest of recurrent connections with zeros
        for region in self.region_dict:
            self.__get_full_connectivity(self.region_dict[region])

        # Fill rest of input connections with zeros
        for inp in self.inp_dict:
            self.__get_full_connectivity(self.inp_dict[inp])

    def gen_w(self, dict_):
        """
        Generates the full recurrent connectivity matrix and associated masks.
        
        Returns:
            tuple: (W_rec, W_rec_mask, W_rec_sign_matrix)
                - W_rec: Learnable weight matrix
                - W_rec_mask: Binary mask for allowed connections
                - W_rec_sign_matrix: Sign constraints for Dale's Law
        """

        region_connection_columns = []
        region_weight_mask_columns = []
        region_sign_matrix_columns = []

        for cur_region in dict_:

            # Collect connections, masks, and sign matrices for current region
            connections_from_region = []
            weight_mask_from_region = []
            sign_matrix_from_region = []

            for connection in self.region_dict:
                region_data = dict_[cur_region].connections[connection]
                connections_from_region.append(region_data["parameter"])
                weight_mask_from_region.append(region_data["weight_mask"])
                sign_matrix_from_region.append(region_data["sign_matrix"])
            
            # Concatenate region-specific matrices
            region_connection_columns.append(torch.cat(connections_from_region, dim=0))
            region_weight_mask_columns.append(torch.cat(weight_mask_from_region, dim=0))
            region_sign_matrix_columns.append(torch.cat(sign_matrix_from_region, dim=0))
        
        # Create final matrices
        W_rec = torch.cat(region_connection_columns, dim=1)
        W_rec_mask = torch.cat(region_weight_mask_columns, dim=1)
        W_rec_sign = torch.cat(region_sign_matrix_columns, dim=1)

        return W_rec, W_rec_mask, W_rec_sign

    def apply_dales_law(self, W_rec, W_rec_mask, W_rec_sign_matrix, lower_bound=0, upper_bound=10):
        """
        Applies Dale's Law constraints to the recurrent weight matrix.
        Dale's Law states that a neuron can be either excitatory or inhibitory, but not both.
        
        Returns:
            torch.Tensor: Constrained weight matrix
        """
        if lower_bound < 0:
            raise ValueError("Lower bounds below zero not allowed for Dale's Law")
        return W_rec_mask * F.hardtanh(W_rec, lower_bound, upper_bound) * W_rec_sign_matrix

    def forward(self, xn, input, *args, noise=True):
        """
        Forward pass through the network.

        Args:
            xn (torch.Tensor): Pre-activation hidden state
            input: Weighted input into the network (must be same shape as specified in config)
            args: Additional unweighted network input
            noise (bool): Whether to apply noise. Defaults to True

        Returns:
            torch.Tensor: Network output sequence
        """
        assert len(self.region_dict) > 0

        # Apply Dale's Law if constrained
        W_rec, W_rec_mask, W_rec_sign_matrix = self.gen_w(self.region_dict)
        if self.constrained:
            W_rec = self.apply_dales_law(
                W_rec, 
                W_rec_mask, 
                W_rec_sign_matrix, 
                lower_bound=self.lower_bound_rec, 
                upper_bound=self.upper_bound_rec
            )

        # Apply to input weights as well
        W_inp, W_inp_mask, W_inp_sign_matrix = self.gen_w(self.inp_dict)
        if self.constrained:
            W_inp = self.apply_dales_law(
                W_inp, 
                W_inp_mask, 
                W_inp_sign_matrix,
                lower_bound=self.lower_bound_inp, 
                upper_bound=self.upper_bound_inp
            )

        xn_next = xn
        hn_next = self.activation(xn)

        # Create lists for xs and hns
        new_hs = []
        new_xs = []

        # Specify sequence length defined by input
        if self.batch_first:
            seq_len = input.shape[1]
        else:
            seq_len = input.shape[0]
        
        # Process sequence
        for t in range(seq_len):

            # Calculate noise terms
            if noise:
                # Calculate noise constants
                const_hid = np.sqrt(2 * self.t_const * self.sigma_recur**2)
                const_inp = np.sqrt(2 * self.t_const * self.sigma_input**2)
                # Sample from normal distribution and scale by constant term
                # Separate noise levels will be applied to each neuron/input
                perturb_hid = const_hid * torch.randn(size=(self.total_num_units,), device=self.device)
                perturb_inp = const_inp * torch.randn(size=(self.total_num_inputs,), device=self.device)
                # Apply input noise
                input = input + perturb_inp
            else:
                perturb_hid = perturb_inp = 0

            # Update hidden state
            xn_next = (xn_next 
                        + self.t_const 
                        * (-xn_next
                            + (W_rec @ hn_next.T).T
                            + self.baseline_inp
                            + perturb_hid
                        )
            )

            # Add input to the network
            if self.batch_first:
                xn_next = xn_next + self.t_const * (W_inp @ input[:, t, :].T).T
                # Add any remaining inputs without weights
                # Example of when this would be useful is for optogenetic manipulations
                for idx in range(len(args)):
                    xn_next = xn_next + self.t_const * args[idx][:, t, :]
            else:
                # Same as above but for different shaped input
                xn_next = xn_next + self.t_const * (W_inp @ input[t, :, :].T).T
                for idx in range(len(args)):
                    xn_next = xn_next + self.t_const * args[idx][t, :, :]

            # Compute activation
            # Gather activation and pre-activation into lists
            hn_next = self.activation(xn_next)
            new_xs.append(xn_next)
            new_hs.append(hn_next)
        
        return torch.stack(new_xs, dim=1), torch.stack(new_hs, dim=1)
    
    def __create_def_values(self, config):
        """Generate default values for configuration

        Args:
            config (json): Network configuration file
        """
        
        # Set default values for recurrent region connections
        for i, region in enumerate(config["recurrent_regions"]):
            if "name" not in region:
                region["name"] = f"region_{i}"
            if "num_units" not in region:
                region["num_units"] = 100
            if "cell_types" not in region:
                region["cell_types"] = {}
            if "init" not in region:
                region["init"] = 0

        # Set default values for recurrent region connections
        for connection in config["recurrent_connections"]:
            if "src_region_cell_type" not in connection:
                connection["src_region_cell_type"] = None
            if "dst_region_cell_type" not in connection:
                connection["dst_region_cell_type"] = None
            if "sign" not in connection:
                connection["sign"] = "exc"
            if "sparsity" not in connection:
                connection["sparsity"] = 0

        # Set default values for input regions
        for i, region in enumerate(config["input_regions"]):
            if "name" not in region:
                region["name"] = f"input_{i}"

        # Set default values for input region connections
        for connection in config["input_connections"]:
            if "dst_region_cell_type" not in connection:
                connection["dst_region_cell_type"] = None
            if "sign" not in connection:
                connection["sign"] = "exc"
            if "sparsity" not in connection:
                connection["sparsity"] = 0

    def __gen_region_mask(self, region, cell_type=None):
        """
        Generates a mask for a specific region and optionally a cell type.

        Args:
            region (str): Region name
            cell_type (str, optional): Cell type within region. Defaults to None

        Returns:
            torch.Tensor: Binary mask
        """
        mask_type = "full" if cell_type is None else cell_type
        mask = []
        
        for next_region in self.region_dict:
            if region == next_region:
                mask.append(self.region_dict[region].masks[mask_type])
            else:
                mask.append(self.region_dict[next_region].masks["zero"])
        
        return torch.cat(mask).to(self.device)

    def __get_full_connectivity(self, region):
        """
        Ensures all possible connections are defined for a region, adding zero
        connections where none are specified.

        Args:
            region (Region): Region object to complete connections for
        """
        for other_region in self.region_dict:
            if not region.has_connection_to(other_region):
                region.add_connection(
                    dst_region_name=other_region,
                    dst_region=self.region_dict[other_region],
                    src_region_cell_type=None,
                    dst_region_cell_type=None,
                    sign=None,
                    sparsity=None,
                    zero_connection=True
                )

    def __get_total_num_units(self, dict_):
        """
        Calculates total number of units across all regions.

        Returns:
            int: Total number of units
        """
        return sum(region.num_units for region in dict_.values())

    def __get_tonic_inp(self):
        """
        Collects baseline firing rates for all regions.

        Returns:
            torch.Tensor: Vector of baseline firing rates
        """
        return torch.cat([region.base_firing for region in self.region_dict.values()]).to(self.device) 