"""mRNN core module.

Implements the multi-regional recurrent neural network (mRNN) building blocks
and step-wise dynamics, along with helpers for connectivity, constraints, and
initialization."""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Tuple
from collections import OrderedDict
import warnings
from mrnntorch.region.region_base import Region
from mrnntorch.region.recurrent_region import RecurrentRegion
from mrnntorch.region.input_region import InputRegion
from mrnntorch.region.region_base import (
    DEFAULT_REC_REGIONS,
    DEFAULT_REGION_BASE,
    DEFAULT_CONNECTIONS,
)


DEFAULTS_MRNN = {
    "config": None,
    "activation": "relu",
    "noise_level_act": 0.01,
    "noise_level_inp": 0.01,
    "rec_constrained": True,
    "inp_constrained": True,
    "dt": 10,
    "tau": 100,
    "batch_first": True,
    "spectral_radius": None,
    "config_finalize": True,
    "device": "cuda",
}


def linear(x):
    return x


class mRNN(nn.Module):
    def __init__(
        self,
        config: str = DEFAULTS_MRNN["config"],
        activation: str = DEFAULTS_MRNN["activation"],
        noise_level_act: float = DEFAULTS_MRNN["noise_level_act"],
        noise_level_inp: float = DEFAULTS_MRNN["noise_level_inp"],
        rec_constrained: bool = DEFAULTS_MRNN["rec_constrained"],
        inp_constrained: bool = DEFAULTS_MRNN["inp_constrained"],
        dt: float = DEFAULTS_MRNN["dt"],
        tau: float = DEFAULTS_MRNN["tau"],
        batch_first: bool = DEFAULTS_MRNN["batch_first"],
        spectral_radius: float = DEFAULTS_MRNN["spectral_radius"],
        config_finalize: bool = DEFAULTS_MRNN["config_finalize"],
        device: str = DEFAULTS_MRNN["device"],
    ):
        super(mRNN, self).__init__()
        """
        Multi-Regional Recurrent Neural Network (mRNN).

        Simulates interactions between multiple recurrent "regions" with optional
        input regions. Supports Dale's Law constraints, tonic inputs, noise in
        hidden state and inputs, and basic step-wise dynamics with configurable
        discretization parameters.

        Key features:
        - Multiple regions with independent sizes and signs (excitatory/inhibitory)
        - Dale's Law constraints via sign masks on weights
        - Optional noise on hidden state and input
        - Tonic (baseline) input per region
        - JSON-based configuration or fully manual construction

        Args:
            config (str | None): Path to a JSON configuration file describing
                recurrent regions, input regions, and their connections. If None,
                build the network manually by calling the add_* methods.
            activation (str): One of {"relu", "tanh", "sigmoid", "softplus", "linear"}.
            noise_level_act (float): Std of hidden-state noise term. Default: 0.01.
            noise_level_inp (float): Std of input noise term. Default: 0.01.
            rec_constrained (bool): If True, apply Dale's Law to rec regions. Default: True.
            inp_constrained (bool): If True, apply Dale's Law to inp regions. Default: True.
            dt (float): Discrete step in ms used for the Euler update. Default: 10.
            tau (float): Time constant in ms; alpha = dt / tau. Default: 100.
            batch_first (bool): If True, sequences are [B, T, ...]; else [T, B, ...].
            spectral_radius (float | None): If set, scales recurrent weights so the
                spectral radius equals this value after finalization.
            config_finalize (bool): If True and a config is supplied, finalize
                connectivity after reading config. Default: True.
            device (str): Torch device string (e.g., "cpu" or "cuda"). Default: "cuda".
        """

        # Initialize network parameters
        self.region_dict = {}
        self.inp_dict = {}
        self.region_mask_dict = {}
        self.rec_constrained = rec_constrained
        self.inp_constrained = inp_constrained
        self.device = device
        self.dt = dt
        self.tau = tau
        self.alpha = self.dt / self.tau
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation
        self.spectral_radius = spectral_radius
        self.config_finalize = config_finalize
        self.rec_finalized = False
        self.inp_finalized = False

        # Specify activation function
        # Only common activations are implemented
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "linear":
            self.activation = linear
        else:
            raise Exception(
                "Only relu, tanh, sigmoid, or linear activations are implemented"
            )

        # Allow configuration file to be optional
        # If configuration is given, network will be automatically generated using it
        # Otherwise, user will manually build a network in their own class
        if config is not None:
            # Load and process configuration
            with open(config, "r") as f:
                config_file = json.load(f)

            # Default everything to empty dict
            # Nothing inherently needs to be specified and can instead be created in custom network
            config_file.setdefault("recurrent_regions", {})
            config_file.setdefault("input_regions", {})
            config_file.setdefault("recurrent_connections", {})
            config_file.setdefault("input_connections", {})

            """
                Configuration file protocol:
                
                    Here we are allowing flexibility between using configs vs. custom network definitions.
                    None of the regions or connections in the network need to be fully specified in the config.
                    Users can specify some regions and connections in the config, then define the rest manually if
                    they choose.
                    
                    config_finalize is defaulted to True, this assumes that the connections in the configuration
                    define all of the connections in the network, and that the network will be finalized automatically 
                    after passing the config. Users can set this to False to continue to build regions and 
                    connections in their custom model after passing the config.

                    Lastly, an empty config that is passed in(or empty parts of the config) default to an empty dictionary {}. 
                    In this case, any key in the json file that is empty will not affect the network or be defined at all.
                    These missing pieces must then be defined in the custom model. Connections that are defined in the 
                    config without the corresponding regions being defined will give an error.
                    Additionally, the config file itself defaults to None, which would then imply the user needs to
                    manually enter all regions and connections in the custom model.
            """

            # Generate network structure
            self._create_def_values(config_file)

            if len(config_file["recurrent_regions"]) >= 1:
                # Generate recurrent regions
                for region in config_file["recurrent_regions"]:
                    self.add_recurrent_region(
                        name=region["name"],
                        num_units=region["num_units"],
                        sign=region["sign"],
                        base_firing=region["base_firing"],
                        init=region["init"],
                        device=self.device,
                        parent_region=region["parent_region"],
                        learnable_bias=region["learnable_bias"],
                    )

            if len(config_file["input_regions"]) >= 1:
                # Generate input regions
                for region in config_file["input_regions"]:
                    self.add_input_region(
                        name=region["name"],
                        num_units=region["num_units"],
                        sign=region["sign"],
                        device=self.device,
                    )

            # Now checking whether or not connections are specified in config
            if len(config_file["recurrent_connections"]) >= 1:
                # Generate recurrent connections
                for connection in config_file["recurrent_connections"]:
                    self.add_recurrent_connection(
                        src_region=connection["src_region"],
                        dst_region=connection["dst_region"],
                        sparsity=connection["sparsity"],
                    )
                """
                    Finalization for Configuration:

                        This completes the connections matrix between regions 
                        by padding with zeros where explicit connections are not specified.
                """

                if self.config_finalize:
                    self.finalize_rec_connectivity()

            if len(config_file["input_connections"]) >= 1:
                # Generate input connections
                for connection in config_file["input_connections"]:
                    self.add_input_connection(
                        src_region=connection["src_region"],
                        dst_region=connection["dst_region"],
                        sparsity=connection["sparsity"],
                    )
                # Finalization input regions
                if self.config_finalize:
                    self.finalize_inp_connectivity()

    def __setitem__(self, idx: str, region: RecurrentRegion | InputRegion):
        """Assign a recurrent region or input region to a valid index in mRNN

        Args:
            idx (str): an input or recurrent region in the mRNN
            region (RecurrentRegion | InputRegion): the new region used for assignment
        """
        assert isinstance(idx, str), "Only string indexing to regions is allowed"
        if idx in self.region_dict:
            if isinstance(region, RecurrentRegion):
                self.region_dict[idx] = region
            else:
                raise ValueError(
                    "Not a RecurrentRegion object, \
                    cannot assign to recurrent region"
                )
        elif idx in self.inp_dict:
            if isinstance(region, InputRegion):
                self.inp_dict[idx] = region
            else:
                raise ValueError(
                    "Not an InputRegion object, \
                    cannot assign to input region"
                )
        else:
            raise ValueError("Index not a valid recurrent or input region")

    def __getitem__(self, idx: str) -> RecurrentRegion | InputRegion:
        """
        Index a recurrent region or input region in mRNN

        Args:
            idx (str): an input or recurrent region in the mRNN
        """
        assert isinstance(idx, str), "Only string indexing to regions is allowed"
        if idx in self.region_dict:
            return self.region_dict[idx]
        elif idx in self.inp_dict:
            return self.inp_dict[idx]
        else:
            raise ValueError("Index not a valid recurrent or input region")

    def add_recurrent_region(
        self,
        name: str,
        num_units: int,
        sign: str = DEFAULT_REC_REGIONS["sign"],
        base_firing: float = DEFAULT_REC_REGIONS["base_firing"],
        init: float = DEFAULT_REC_REGIONS["init"],
        device: str = DEFAULT_REC_REGIONS["device"],
        parent_region: str = DEFAULT_REC_REGIONS["parent_region"],
        learnable_bias: bool = DEFAULT_REC_REGIONS["learnable_bias"],
    ):
        """Add a recurrent region to the network.

        Args:
            name (str): Region name (unique key).
            num_units (int): Number of units in this region.
            sign (str): "pos" for excitatory or "neg" for inhibitory outputs.
            base_firing (float | torch.Tensor): Baseline firing per unit.
            init (float): Initial pre-activation value for units in this region.
            device (str): Device where tensors for this region live.
            parent_region (str | None): Optional parent region identifier.
            learnable_bias (bool): If True, baseline firing is trainable.
        """
        if self.rec_finalized:
            raise Exception(
                "Recurrent connectivity already finalized, please \
                include all regions and connections beforehand"
            )

        # Create region
        self.region_dict[name] = RecurrentRegion(
            num_units=num_units,
            base_firing=base_firing,
            init=init,
            sign=sign,
            device=device,
            parent_region=parent_region,
            learnable_bias=learnable_bias,
        )
        # General network parameters
        self.total_num_units = self._get_total_num_units(self.region_dict)
        # Get indices for specific regions
        for region in self.region_dict:
            # Get the mask for the whole region, regardless of cell type
            self.region_mask_dict[region] = {}
            self.region_mask_dict[region] = self._gen_region_mask(region)

    def add_input_region(
        self,
        name: str,
        num_units: int,
        sign: str = DEFAULT_REGION_BASE["sign"],
        device: str = DEFAULT_REGION_BASE["device"],
    ):
        """Add an input region to the network.

        Args:
            name (str): Input region name (unique key).
            num_units (int): Number of input channels in this region.
            sign (str): "pos" or "neg"; used to set sign mask for inputs.
            device (str): Device where tensors for this region live.
        """
        if self.inp_finalized:
            raise Exception(
                "Input connectivity already finalized, \
                please include all regions and connections beforehand"
            )

        # Create region
        self.inp_dict[name] = InputRegion(num_units=num_units, sign=sign, device=device)
        # Update number of input units
        self.total_num_inputs = self._get_total_num_units(self.inp_dict)

    def add_recurrent_connection(
        self,
        src_region: str,
        dst_region: str,
        sparsity: float = DEFAULT_CONNECTIONS["sparsity"],
    ):
        """Create a recurrent connection from one region to another.

        Registers the weight parameter and associated masks. If ``sparsity`` is
        provided, a binary connectivity mask is sampled accordingly.

        Currently not allowed to make a zero connection.

        Args:
            src_region (str): Source recurrent region name.
            dst_region (str): Destination recurrent region name.
            sparsity (float | None): Fraction of connections to keep (0-1). If
                None, dense mask is used.
        """
        # Ensure that no more connections can be added if network is finalized
        if self.rec_finalized:
            raise Exception(
                "Recurrent connectivity already finalized, \
                please include all regions and connections beforehand"
            )

        # Add connection to specified region object
        self.region_dict[src_region].add_connection(
            dst_region_name=dst_region,
            dst_region_units=self.region_dict[dst_region].num_units,
            sparsity=sparsity,
        )

        # Get the empty weights
        weight = self.region_dict[src_region][dst_region].parameter

        # initialize the empty weights based on constraints
        if self.rec_constrained:
            self._constrained_default_init_rec(weight)
        else:
            nn.init.xavier_normal_(weight)

    def add_input_connection(
        self,
        src_region: str,
        dst_region: str,
        sparsity: float | None = DEFAULT_CONNECTIONS["sparsity"],
    ):
        """Create an input connection from an input region to a recurrent region.
        Currently not allowed to make a zero connection.

        Args:
            src_region (str): Source input region name.
            dst_region (str): Destination recurrent region name.
            sparsity (float | None): Fraction of connections to keep (0-1). If
                None, dense mask is used.
        """
        if self.inp_finalized:
            raise Exception(
                "Input connectivity already finalized, \
                please include all regions and connections beforehand"
            )

        # Add connection to specified input region object
        self.inp_dict[src_region].add_connection(
            dst_region_name=dst_region,
            dst_region_units=self.region_dict[dst_region].num_units,
            sparsity=sparsity,
        )

        # Access empty input weight
        weight = self.inp_dict[src_region][dst_region].parameter

        # initialize the weight depending on constraints
        if self.inp_constrained:
            self._constrained_default_init_inp(weight)
        else:
            nn.init.xavier_normal_(weight)

    def set_spectral_radius(
        self, W: torch.Tensor, W_tmp: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Scale recurrent weights so their spectral radius matches ``self.spectral_radius``.

        Usage:
        1. Define regions and connections (via config or manual methods).
        2. If building manually, call :meth:`finalize_connectivity` first.
        3. Set ``self.spectral_radius`` and call this method.
        4. W_tmp will compute spectral radius of another network (i.e dales law network)
        """
        # Compute spectral radius
        if W_tmp is not None:
            cur_spectral_radius = self.compute_spectral_radius(W_tmp)
        else:
            cur_spectral_radius = self.compute_spectral_radius(W)
        W_scaled = (W / cur_spectral_radius) * self.spectral_radius
        return W_scaled

    def finalize_connectivity(self):
        """Finalize both input and recurrent connectivity
        This function is primarily implemented so users don't have to
        separately call rec and inp connectivity functions
        """
        if not self.rec_finalized:
            self.finalize_rec_connectivity()
        if not self.inp_finalized:
            self.finalize_inp_connectivity()

    def finalize_rec_connectivity(self):
        """Fill rest of recurrent connections with zeros
        Ensure finalized flag is set to true
        """
        for region in self.region_dict:
            self._get_full_connectivity(self.region_dict[region])
        # Apply Dale's Law if constrained
        W_rec, W_rec_mask, W_rec_sign_matrix = self.gen_w(self.region_dict)
        # Set spectral radius
        if self.spectral_radius is not None:
            if self.rec_constrained:
                W_rec_tmp = self.apply_dales_law(W_rec, W_rec_mask, W_rec_sign_matrix)
            else:
                W_rec_tmp = W_rec * W_rec_mask
            W_rec = self.set_spectral_radius(W_rec, W_tmp=W_rec_tmp)
        # Create parameters
        self.W_rec = nn.Parameter(W_rec)
        self.W_rec_mask = nn.Parameter(W_rec_mask, requires_grad=False)
        self.W_rec_sign_matrix = nn.Parameter(W_rec_sign_matrix, requires_grad=False)
        # Set finalized flag to true, no more connections can be added
        self.rec_finalized = True

    def finalize_inp_connectivity(self):
        """Fill rest of input connections with zeros
        Ensure finalized flag is set to true
        """
        for inp in self.inp_dict:
            self._get_full_connectivity(self.inp_dict[inp])
        # Apply to input weights as well
        W_inp, W_inp_mask, W_inp_sign_matrix = self.gen_w(self.inp_dict)
        # Create parameters
        self.W_inp = nn.Parameter(W_inp)
        self.W_inp_mask = nn.Parameter(W_inp_mask, requires_grad=False)
        self.W_inp_sign_matrix = nn.Parameter(W_inp_sign_matrix, requires_grad=False)
        # Set finalized flag to true, no more connections can be added
        self.inp_finalized = True

    def compute_spectral_radius(self, weight: torch.Tensor) -> float:
        """Compute the spectral radius (max |eigenvalue|) of a square matrix.

        Args:
            weight (torch.Tensor): Square weight matrix.

        Returns:
            torch.Tensor: Spectral radius as a scalar tensor.
        """
        # Largest absolute eigenvalue of W_rec
        eig_vals = torch.linalg.eigvals(weight)
        abs_eig_vals = eig_vals.abs()
        spectral_radius = abs_eig_vals.max()
        return spectral_radius

    def gen_w(self, dict_: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates the full recurrent connectivity matrix and associated masks.

        Returns:
            tuple: (W_rec, W_rec_mask, W_rec_sign_matrix)
                - W_rec: Learnable weight matrix
                - W_rec_mask: Binary mask for allowed connections
                - W_rec_sign_matrix: Sign constraints for Dale's Law
        """

        # Initialize empty lists to hold the concatenated tensors
        region_connection_columns = []
        region_weight_mask_columns = []
        region_sign_matrix_columns = []

        # Iterate over the regions in dict_
        for cur_region in dict_:
            # List comprehensions to collect connections, masks, and sign matrices
            # calling Region[connection] should invoke __getitem__
            connections_from_region = [
                dict_[cur_region][connection].parameter
                for connection in self.region_dict
            ]
            weight_mask_from_region = [
                dict_[cur_region][connection].weight_mask
                for connection in self.region_dict
            ]
            sign_matrix_from_region = [
                dict_[cur_region][connection].sign_matrix
                for connection in self.region_dict
            ]

            # Concatenate the region-specific matrices and append to the lists
            region_connection_columns.append(torch.cat(connections_from_region, dim=0))
            region_weight_mask_columns.append(torch.cat(weight_mask_from_region, dim=0))
            region_sign_matrix_columns.append(torch.cat(sign_matrix_from_region, dim=0))
        # Concatenate all region-specific matrices along the column dimension
        W_rec = torch.cat(region_connection_columns, dim=1)
        W_rec_mask = torch.cat(region_weight_mask_columns, dim=1)
        W_rec_sign = torch.cat(region_sign_matrix_columns, dim=1)

        return W_rec, W_rec_mask, W_rec_sign

    def apply_dales_law(
        self,
        W_rec: torch.Tensor,
        W_rec_mask: torch.Tensor,
        W_rec_sign_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies Dale's Law constraints to the recurrent weight matrix.
        Dale's Law states that a neuron can be either excitatory or inhibitory, but not both.

        Returns:
            torch.Tensor: Constrained weight matrix
        """
        return W_rec_mask * torch.abs(W_rec) * W_rec_sign_matrix

    @property
    def tonic_inp(self):
        """
        Collects baseline firing rates for all regions.

        Returns:
            torch.Tensor: Vector of baseline firing rates
        """
        return torch.cat(
            [region.base_firing for region in self.region_dict.values()]
        ).to(self.device)

    def named_rec_regions(self, prefix: str = ""):
        """Loop through rec region names and objects

        Args:
            prefix (str, optional): Defaults to ''.
        """
        for name, region in self.region_dict.items():
            yield prefix + name, region

    def named_inp_regions(self, prefix: str = ""):
        """Loop through inp region names and objects

        Args:
            prefix (str, optional): Defaults to ''.
        """
        for name, region in self.inp_dict.items():
            yield prefix + name, region

    def get_region_size(self, region: str) -> int:
        """Get the number of units in a region

        Args:
            region (str): region to get size of
        """
        return (
            self.region_dict[region].num_units
            if region in self.region_dict
            else self.inp_dict[region].num_units
        )

    def get_region_activity(self, act: torch.Tensor, *args) -> torch.Tensor:
        """
        Takes in hn and the specified region and returns the activity hn for the corresponding region

        Args:
            act (Torch.Tensor): tensor containing model hidden activity. Activations must be in last dimension (-1)
            args (str): name of regions to collect activity from

        Returns:
            region_hn: tensor containing hidden activity only for specified region
        """
        # Default to returning whole activity
        unique_regions = list(OrderedDict.fromkeys(args))
        if not args:
            return act
        # Check to ensure region is recurrent
        for region in args:
            if region in self.inp_dict:
                raise Exception("Can only get activity for recurrent regions")

        args = self._ensure_order(*args)

        # Go and check if any parent regions are entered
        for region in unique_regions.copy():
            if self._check_if_parent_region(region):
                unique_regions.remove(region)
                unique_regions.extend(self._get_child_regions(region))

        # collect all necessary indices now
        region_indices = {
            region: self.get_region_indices(region) for region in unique_regions
        }

        region_acts = torch.cat(
            [
                act[..., start_idx:end_idx]
                for region in unique_regions
                for (start_idx, end_idx) in [region_indices[region]]
            ],
            dim=-1,
        )
        return region_acts

    def get_weight_subset(self, *args, W: torch.Tensor | None = None) -> torch.Tensor:
        """Gather a subset of the weights from all regions in args to and from
        each other and themselves.

        This should return a square matrix of all connections between regions in
        args

        Args:
            args (str): all regions specified
            W (torch.Tensor): use this specified weight matrix instead

        Returns:
            torch.Tensor: subset of the total weight matrix
        """

        if W is None:
            # Gather original weight matrix and apply Dale's Law if constrained
            # Can only be recurrent if not using to and from
            if self.rec_constrained:
                mrnn_weight = self.apply_dales_law(
                    self.W_rec, self.W_rec_mask, self.W_rec_sign_matrix
                )
            else:
                mrnn_weight = self.W_rec * self.W_rec_mask

            # Default to standard weight matrix if no regions are provided
            if not args:
                return mrnn_weight
        else:
            mrnn_weight = W

        # Check if user specifies input region through args instead of to, from
        for region in args:
            if region in self.inp_dict:
                raise Exception("Can only gather input subsets using get_projection")

        args = self._ensure_order(*args)

        # This is used to store the final collected weight matrix
        global_weight_collection = []

        region_indices = {region: self.get_region_indices(region) for region in args}

        # List comprehension that gathers all information gathering weight subset
        global_weight_collection = [
            torch.cat(
                [
                    mrnn_weight[src_start_idx:src_end_idx, dst_start_idx:dst_end_idx]
                    for dst_region in args
                    for dst_start_idx, dst_end_idx in [region_indices[dst_region]]
                ],
                dim=1,
            )
            for _, (src_start_idx, src_end_idx) in region_indices.items()
        ]

        # Similar to before but now concatenating along rows
        global_weight_collection = torch.cat(global_weight_collection, dim=0)

        return global_weight_collection

    def get_projection(self, to: str, from_: str) -> torch.Tensor:
        """Gather a subset of the weights

        Args:
            to (str): Name of region that is recieving projection (row)
            from_ (str): Name of region projecting (column)

        Returns:
            torch.Tensor: weight matrix of from_->to projection
        """

        # Store regions if parent regions are given
        to_regions = []
        from_regions = []

        # If to region is a parent region, then get children regions
        if self._check_if_parent_region(to):
            to_regions.extend(self._get_child_regions(to))
        else:
            to_regions.append(to)
        # If from region is a parent region, then get children regions
        if self._check_if_parent_region(from_):
            from_regions.extend(self._get_child_regions(from_))
        else:
            from_regions.append(from_)

        # Check which weight matrix to use based on from region
        if from_ in self.region_dict:
            if self.rec_constrained:
                weight = self.apply_dales_law(
                    self.W_rec, self.W_rec_mask, self.W_rec_sign_matrix
                )
            else:
                weight = self.W_rec
        elif from_ in self.inp_dict:
            if self.inp_constrained:
                weight = self.apply_dales_law(
                    self.W_inp, self.W_inp_mask, self.W_inp_sign_matrix
                )
            else:
                weight = self.W_inp
        else:
            raise Exception("from_ region not in region or input dictionary")

        # Store all of the weights to region from another
        to_from_weight = []
        # Now go through each of the collected regions and get the weights
        for to_region in to_regions:
            from_weight = []
            # Get the indices for to region
            to_start_idx, to_end_idx = self.get_region_indices(to_region)
            for from_region in from_regions:
                # Gather indices
                from_start_idx, from_end_idx = self.get_region_indices(from_region)
                from_weight.append(
                    weight[to_start_idx:to_end_idx, from_start_idx:from_end_idx]
                )
            # Collect all of the weights for from region
            collected_from_weight = torch.cat(from_weight, dim=1)
            to_from_weight.append(collected_from_weight)
        # Collect final weight matrix
        collected_to_from_weight = torch.cat(to_from_weight, dim=0)
        return collected_to_from_weight

    def get_region_indices(self, region: str) -> tuple[int, int]:
        """
        Gets the start and end indices for a specific region in the hidden state vector.

        Args:
            region (str): Name of the region

        Returns:
            tuple: (start_idx, end_idx)
        """

        if self._check_if_parent_region(region):
            raise ValueError(
                "Can only get indices of a single region, not parent region"
            )

        # Get the region indices
        start_idx = 0
        end_idx = 0

        # Check whether or not specified region is input or rec
        # This is to handle indices for both rec and inp regions
        if region in self.region_dict:
            dict_ = self.region_dict
        elif region in self.inp_dict:
            dict_ = self.inp_dict
        else:
            raise Exception("Not an input or recurrent region")

        for cur_reg in dict_:
            region_units = dict_[cur_reg].num_units
            if cur_reg == region:
                end_idx = start_idx + region_units
                break
            start_idx += region_units

        return start_idx, end_idx

    @property
    def initial_condition(self) -> torch.Tensor:
        """Create an initial xn for the network

        Returns:
            Tensor: tensor like xn filled with region specified initial conds
        """
        return torch.cat([region.init for region in self.region_dict.values()]).to(
            self.device
        )

    def batched_initial_condition(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return initial condition expanded to meet batch_size

        Args:
            batch_size (int): batch size to expand to

        Returns:
            xn, hn (Tensor, Tensor): initial preactivation and hidden activation
        """
        xn = self.initial_condition.unsqueeze(0).repeat(batch_size, 1)
        hn = self.initial_condition.unsqueeze(0).repeat(batch_size, 1)
        return xn, hn

    def forward(
        self,
        inp: torch.Tensor,
        x0: torch.Tensor,
        h0: torch.Tensor,
        *args,
        noise: bool = False,
        tv_noise: bool = False,
        tv_noise_scale: float = 0.1,
        start_noise: torch.Tensor | None = None,
        W_rec: torch.Tensor | None = None,
        W_inp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent dynamics over a sequence.

        Discretized update: ``x_{t+1} = x_t + alpha * (-x_t + W_rec h_t + W_inp u_t + b + noise)``
        and ``h_{t+1} = activation(x_{t+1})``.

        Args:
            inp (torch.Tensor): Input sequence. Shape ``[B, T, I]`` if batch_first
                else ``[T, B, I]``.
            x0 (torch.Tensor): Initial pre-activation hidden state, shape ``[B, H]``.
            h0 (torch.Tensor): Initial activation, shape ``[B, H]``.
            *args (torch.Tensor): Optional additive inputs with same temporal layout
                as ``inp`` and feature size ``H``.
            noise (bool): If True, add Gaussian noise to hidden state and inputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(x_seq, h_seq)`` sequences matching
            the temporal layout of ``inp``.
        """
        assert len(self.region_dict) > 0
        assert len(self.inp_dict) > 0
        assert self.rec_finalized or self.inp_finalized, (
            "Recurrent or input weights are not finalized, \
            call finalize_connectivity() in your custom model definition"
        )

        # warnings and assertions for noise parameters
        if tv_noise and not noise:
            warnings.warn(
                "tv_noise set to True and noise is False, no noise \
                will be applied to the network"
            )
        if start_noise is not None and not noise:
            warnings.warn(
                "start_noise is not None and noise is False, no noise \
                will be applied to the network"
            )

        if inp.dim() != 3:
            raise Exception(
                "input must be 3 dimensional, \
                            [batch, time, units] for batch_first=True, \
                            and [time, batch, units] otherwise]."
            )
        if x0.dim() != 2:
            raise Exception("x0 must be 2 dimensional, [batch, units].")

        if W_rec is None:
            # Apply Dale's Law if constrained
            if self.rec_constrained:
                W_rec = self.apply_dales_law(
                    self.W_rec, self.W_rec_mask, self.W_rec_sign_matrix
                )
            else:
                W_rec = self.W_rec * self.W_rec_mask

        if W_inp is None:
            # Apply to input weights as well
            if self.inp_constrained:
                W_inp = self.apply_dales_law(
                    self.W_inp, self.W_inp_mask, self.W_inp_sign_matrix
                )
            else:
                W_inp = self.W_inp * self.W_inp_mask

        baseline_inp = self.tonic_inp

        xn_next = x0
        hn_next = h0

        if self.batch_first:
            # If batch first then batch is first dim
            batch_shape = inp.shape[0]
            seq_len = inp.shape[1]
            shape = (batch_shape, seq_len, self.total_num_units)
        else:
            # If not batch first then seq_len is first dim
            seq_len = inp.shape[0]
            batch_shape = inp.shape[1]
            shape = (seq_len, batch_shape, self.total_num_units)

        # Create lists for xs and hns
        new_hs = torch.empty(size=shape, device=self.device)
        new_xs = torch.empty(size=shape, device=self.device)

        # Process sequence
        for t in range(seq_len):
            # Gather input at current timestep
            if self.batch_first:
                inp_t = inp[:, t, :]
            else:
                inp_t = inp[t, :, :]

            # Sample from normal distribution and scale by constant term
            if noise:
                # Separate noise levels will be applied to each neuron/input
                hid_noise = self._hid_noise(self.hid_noise_const, batch_shape)
                inp_noise = self._inp_noise(self.inp_noise_const, batch_shape)
            else:
                hid_noise = inp_noise = 0

            # tv noise is only an option if noise is set to True
            if tv_noise and noise:
                hid_noise = hid_noise / (1 + (t * tv_noise_scale))
                inp_noise = inp_noise / (1 + (t * tv_noise_scale))

            # start_noise only an option if noise is True
            if start_noise is not None and noise:
                start_mask = (
                    torch.where(start_noise > t, 0, 1).unsqueeze(1).to(self.device)
                )
                hid_noise = hid_noise * start_mask
                inp_noise = inp_noise * start_mask

            """
            Update hidden state
            Discretized equation of the form: 
                x_(t+1) = x_t + alpha * (-x_t + Wh + W_ix + b)
            """

            xn_next = xn_next + self.alpha * (
                -xn_next + (W_rec @ hn_next.T).T + baseline_inp + hid_noise
            )

            # Add input
            xn_next = xn_next + self.alpha * (W_inp @ (inp_t + inp_noise).T).T

            # Add any additional arg inputs (stim inputs typically)
            if self.batch_first:
                for idx in range(len(args)):
                    xn_next = xn_next + self.alpha * args[idx][:, t, :]
            else:
                for idx in range(len(args)):
                    xn_next = xn_next + self.alpha * args[idx][t, :, :]

            """
            Compute activation
            Activation of the form: 
                h_t = sigma(x_t)
            """

            hn_next = self.activation(xn_next)

            if self.batch_first:
                new_xs[:, t, :] = xn_next
                new_hs[:, t, :] = hn_next
            else:
                new_xs[t, :, :] = xn_next
                new_hs[t, :, :] = hn_next

        return new_xs, new_hs

    @property
    def hid_noise_const(self):
        """noise constant used for hidden activity"""
        const_hid = (1 / self.alpha) * np.sqrt(2 * self.alpha * self.sigma_recur**2)
        return const_hid

    @property
    def inp_noise_const(self):
        """noise constant used for inputs"""
        const_inp = (1 / self.alpha) * np.sqrt(2 * self.alpha * self.sigma_input**2)
        return const_inp

    def _hid_noise(self, const: float, batch_shape: int):
        """
        Gather a random noise sample at a given timepoint

        Args:
            const (float): hidden noise constant
            batch_shape (int): batch_shape

        Returns:
            Tensor: total_num_units sized tensor containing Gaussian noise
        """
        perturb_hid = const * torch.randn(
            size=(batch_shape, self.total_num_units), device=self.device
        )
        return perturb_hid

    def _inp_noise(self, const, batch_shape):
        """
        Gather a random noise sample at a given timepoint

        Args:
            const (float): hidden noise constant
            batch_shape (int): batch_shape

        Returns:
            Tensor: total_num_inputs sized tensor containing Gaussian noise
        """
        perturb_inp = const * torch.randn(
            size=(batch_shape, self.total_num_inputs), device=self.device
        )
        return perturb_inp

    def _create_def_values(self, config: dict):
        """Generate default values for configuration

        Args:
            config (json): Network configuration file
        """

        # Set default values for recurrent region connections
        for i, region in enumerate(config["recurrent_regions"]):
            # Go through all possible default options in default dict
            for param in DEFAULT_REC_REGIONS:
                # If the parameter is not specified by the user in the configuration...
                if param not in region:
                    # If parameter is name, add the index to ensure unique naming
                    if param == "name":
                        region[param] = DEFAULT_REC_REGIONS[param] + str(i)
                    # Otherwise, default the parameter
                    else:
                        region[param] = DEFAULT_REC_REGIONS[param]

        # Set default values for recurrent region connections
        for connection in config["recurrent_connections"]:
            for param in DEFAULT_CONNECTIONS:
                if param not in connection:
                    connection[param] = DEFAULT_CONNECTIONS[param]

        # Set default values for input regions
        for i, region in enumerate(config["input_regions"]):
            for param in DEFAULT_REGION_BASE:
                if param not in region:
                    if param == "name":
                        region[param] = DEFAULT_REGION_BASE[param] + str(i)
                    else:
                        region[param] = DEFAULT_REGION_BASE[param]

        # Set default values for input region connections
        for connection in config["input_connections"]:
            for param in DEFAULT_CONNECTIONS:
                if param not in connection:
                    connection[param] = DEFAULT_CONNECTIONS[param]

    def _gen_region_mask(self, region: str) -> torch.Tensor:
        """
        Generates a mask for a specific region and optionally a cell type.

        Args:
            region (str): Region name

        Returns:
            torch.Tensor: Binary mask
        """
        mask = []

        for next_region in self.region_dict:
            if region == next_region:
                mask.append(self.region_dict[region].masks["ones"])
            else:
                mask.append(self.region_dict[next_region].masks["zeros"])

        return torch.cat(mask).to(self.device)

    def _get_full_connectivity(self, region: Region):
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
                    dst_region_units=self.region_dict[other_region].num_units,
                    sparsity=None,
                    zero_connection=True,
                )

    def _get_total_num_units(self, dict_: dict) -> int:
        """
        Calculates total number of units across all regions.

        Args:
            dict_ (dict): either region_dict or inp_dict

        Returns:
            int: Total number of units
        """
        return sum(region.num_units for region in dict_.values())

    def _check_if_parent_region(self, parent_region: str) -> bool:
        """
        Return True if any region has ``parent_region`` set to the given name.

        Args:
            parent_region (str): name of parent region

        Returns:
            bool: whether or not parent_region is a parent region
        """
        for region in self.region_dict.values():
            if region.parent_region == parent_region:
                return True
        return False

    def _get_child_regions(self, parent_region: str) -> tuple[str]:
        """
        Return a tuple of region names that list ``parent_region`` as their parent.

        Args:
            parent_region (str): name of parent region

        Returns:
            tuple: names of all child regions under parent region
        """
        child_region_list = []
        for region in self.region_dict:
            if self.region_dict[region].parent_region == parent_region:
                child_region_list.append(region)
        return tuple(child_region_list)

    def _constrained_default_init_rec(self, weight: torch.Tensor):
        """Default init for recurrent weights under Dale's Law constraints.

        Draws weights from a zero-mean normal with variance 1/(2H), then applies
        a sign mask to respect excitation/inhibition of source regions.
        """
        nn.init.normal_(weight, mean=0, std=np.sqrt(1 / (2 * self.total_num_units)))
        mask = torch.sign(weight)
        weight *= mask

    def _constrained_default_init_inp(self, weight: torch.Tensor):
        """Default init for input weights under Dale's Law constraints.

        Draws weights from a zero-mean normal with variance 1/(H + I), then
        applies a sign mask to respect region sign.
        """
        nn.init.normal_(
            weight,
            mean=0,
            std=np.sqrt(1 / (self.total_num_units + self.total_num_inputs)),
        )
        mask = torch.sign(weight)
        weight *= mask

    def _ensure_order(self, *args) -> tuple[str]:
        """Reorder args if given regions are out of order"""
        return tuple(r for r in self.region_dict if r in args)
