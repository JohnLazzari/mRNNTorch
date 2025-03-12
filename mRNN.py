import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from mRNNTorch.Region import RecurrentRegion, InputRegion

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
    "learnable_bias": False
}

DEFAULT_REC_CONNECTIONS = {
    # How sparse the connections will be (float from 0-1)
    "sparsity": None
}

DEFAULT_INP_REGIONS = {
    "name": "inp_",
    "sign": "pos"
}

DEFAULT_INP_CONNECTIONS = {
    "sparsity": None
}

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
        config=None,
        activation="relu",
        noise_level_act=0.01, 
        noise_level_inp=0.01, 
        constrained=True, 
        dt=10,
        tau=100,
        batch_first=True,
        lower_bound_rec=0,
        upper_bound_rec=10,
        lower_bound_inp=0,
        upper_bound_inp=10,
        spectral_radius=None,
        config_finalize=True,
        device="cuda",
    ):
        super(mRNN, self).__init__()

        # Initialize network parameters
        self.region_dict = {}
        self.inp_dict = {}
        self.region_mask_dict = {}
        self.constrained = constrained
        self.device = device
        self.dt = dt
        self.tau = tau
        self.alpha = self.dt / self.tau
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation
        self.lower_bound_rec = lower_bound_rec
        self.upper_bound_rec = upper_bound_rec
        self.lower_bound_inp = lower_bound_inp
        self.upper_bound_inp = upper_bound_inp
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
            self.activation == linear()
        else:
            raise Exception("Only relu, tanh, sigmoid, or linear activations are implemented")
        
        # Allow configuration file to be optional
        # If configuration is given, network will be automatically generated using it
        # Otherwise, user will manually build a network in their own class
        if config is not None:

            # Load and process configuration
            with open(config, 'r') as f:
                config = json.load(f)
            
            # Default everything to empty dict
            # Nothing inherently needs to be specified and can instead be created in custom network
            # This should default all connectivity to zeros (not learnable)
            config.setdefault('recurrent_regions', {})
            config.setdefault('input_regions', {})
            config.setdefault('recurrent_connections', {})
            config.setdefault('input_connections', {})
            
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
            self.__create_def_values(config)

            if len(config["recurrent_regions"]) >= 1:
                # Generate recurrent regions
                for region in config["recurrent_regions"]:
                    self.add_recurrent_region(
                        name=region["name"],
                        num_units=region["num_units"],
                        sign=region["sign"],
                        base_firing=region["base_firing"],
                        init=region["init"],
                        device=self.device,
                        parent_region=region["parent_region"],
                        learnable_bias=region["learnable_bias"]
                    )
            
            if len(config["input_regions"]) >= 1:
                # Generate input regions
                for region in config["input_regions"]:
                    self.add_input_region(
                        name=region["name"],
                        num_units=region["num_units"],
                        sign=region["sign"],
                        device=self.device
                    )

            # Now checking whether or not connections are specified in config
            if len(config["recurrent_connections"]) >= 1:
                # Generate recurrent connections
                for connection in config["recurrent_connections"]:
                    self.add_recurrent_connection(
                        src_region=connection["src_region"],
                        dst_region=connection["dst_region"],
                        sparsity=connection["sparsity"]
                    )
                """
                    Finalization for Configuration:

                        This completes the connections matrix between regions 
                        by padding with zeros where explicit connections are not specified.
                        Does so for both recurrent and input connections.
                """

                if self.config_finalize:
                    self.finalize_rec_connectivity()

            if len(config["input_connections"]) >= 1:
                # Generate input connections
                for connection in config["input_connections"]:
                    self.add_input_connection(
                        src_region=connection["src_region"],
                        dst_region=connection["dst_region"],
                        sparsity=connection["sparsity"]
                    )
                # Finalization input regions
                if self.config_finalize:
                    self.finalize_inp_connectivity()
    
    def add_recurrent_region(
        self, 
        name, 
        num_units, 
        sign=DEFAULT_REC_REGIONS["sign"], 
        base_firing=DEFAULT_REC_REGIONS["base_firing"], 
        init=DEFAULT_REC_REGIONS["init"], 
        device="cuda", 
        parent_region=DEFAULT_REC_REGIONS["parent_region"], 
        learnable_bias=DEFAULT_REC_REGIONS["learnable_bias"]
    ):
        """_summary_

        Args:
            name (_type_): _description_
            num_units (_type_): _description_
            base_firing (int, optional): _description_. Defaults to 0.
            init (int, optional): _description_. Defaults to 0.
            device (str, optional): _description_. Defaults to "cuda".
            cell_types (dict, optional): _description_. Defaults to {}.
        """
        if self.rec_finalized:
            raise Exception("Recurrent connectivity already finalized, please include all regions and connections beforehand")

        self.region_dict[name] = RecurrentRegion(
            num_units=num_units,
            base_firing=base_firing,
            init=init,
            sign=sign,
            device=device,
            parent_region=parent_region,
            learnable_bias=learnable_bias
        )
        # General network parameters
        self.total_num_units = self.__get_total_num_units(self.region_dict)
        # Registering bias parameters
        if isinstance(self.region_dict[name].base_firing, nn.Parameter):
            self.register_parameter(f"{name}_base_firing", self.region_dict[name].base_firing)
        # Get indices for specific regions
        for region in self.region_dict:
            # Get the mask for the whole region, regardless of cell type
            self.region_mask_dict[region] = {}
            self.region_mask_dict[region] = self.__gen_region_mask(region)

    def add_input_region(
        self, 
        name, 
        num_units, 
        sign=DEFAULT_INP_REGIONS["sign"], 
        device="cuda"
    ):
        """_summary_

        Args:
            name (_type_): _description_
            num_units (_type_): _description_
            base_firing (int, optional): _description_. Defaults to 0.
            init (int, optional): _description_. Defaults to 0.
            device (str, optional): _description_. Defaults to "cuda".
            cell_types (dict, optional): _description_. Defaults to {}.
        """
        if self.inp_finalized:
            raise Exception("Input connectivity already finalized, please include all regions and connections beforehand")

        self.inp_dict[name] = InputRegion(
            num_units=num_units,
            sign=sign,
            device=device
        )
        self.total_num_inputs = self.__get_total_num_units(self.inp_dict)
    
    def add_recurrent_connection(self, src_region, dst_region, sparsity=DEFAULT_REC_CONNECTIONS["sparsity"]):
        """_summary_

        Args:
            src_region (_type_): _description_
            dst_region (_type_): _description_
            sign (str, optional): _description_. Defaults to "exc".
            sparsity (_type_, optional): _description_. Defaults to None.
        """
        # Ensure that no more connections can be added if network is finalized
        if self.rec_finalized:
            raise Exception("Recurrent connectivity already finalized, please include all regions and connections beforehand")

        self.region_dict[src_region].add_connection(
            dst_region_name=dst_region,
            dst_region=self.region_dict[dst_region],
            sparsity=sparsity
        )
        # Register current connection as parameter
        # Check that we do not register same parameters more than once
        param_name = f"{src_region}_{dst_region}"
        for name, param in self.region_dict[src_region].named_parameters():
            if name == dst_region:
                # Default initialization for recurrent weights
                self.__constrained_default_init_rec(param) if self.constrained else nn.init.xavier_normal_(param)
                # Register parameter manually
                self.register_parameter(param_name, param)

    def add_input_connection(self, src_region, dst_region, sparsity=DEFAULT_INP_CONNECTIONS["sparsity"]):
        """_summary_

        Args:
            src_region (_type_): _description_
            dst_region (_type_): _description_
            dst_region_cell_type (_type_, optional): _description_. Defaults to None.
            sign (str, optional): _description_. Defaults to "exc".
            sparsity (_type_, optional): _description_. Defaults to None.
        """
        if self.inp_finalized:
            raise Exception("Input connectivity already finalized, please include all regions and connections beforehand")

        self.inp_dict[src_region].add_connection(
            dst_region_name=dst_region,
            dst_region=self.region_dict[dst_region],
            sparsity=sparsity
        )
        # Register all parameters for inputs
        param_name = f"{src_region}_{dst_region}"
        for name, param in self.inp_dict[src_region].named_parameters():
            if name == dst_region:
                # Default initialization for inputs
                self.__constrained_default_init_inp(param) if self.constrained else nn.init.xavier_normal_(param)
                # Manually register parameter
                self.register_parameter(param_name, param)

    def set_spectral_radius(self):
        """Set the spectral radius of the recurrent weight matrix
        
            Usage (in your own custom network class):
                1. Define connectivity in any way preferred (config or manual)
                2. If using manual, then make sure to call mrnn.finalize_connectivity()
                3. Lastly, call mrnn.set_spectral_radius(radius)

        Args:
            orig_radius (_type_): _description_
            desired_radius (_type_): _description_
        """
        # Get weight matrix
        W_rec, W_rec_mask, W_rec_sign_matrix = self.gen_w(self.region_dict)
        if self.constrained:
            W_rec = self.apply_dales_law(
                W_rec, 
                W_rec_mask, 
                W_rec_sign_matrix, 
                lower_bound=self.lower_bound_rec, 
                upper_bound=self.upper_bound_rec
            )
        # Compute spectral radius
        cur_spectral_radius = self.compute_spectral_radius(W_rec)
        # Go through each region and scale weights
        for region in self.region_dict:
            for connection in self.region_dict:
                region_data = self.region_dict[region].connections[connection]
                region_data["parameter"].data /= cur_spectral_radius
                region_data["parameter"].data *= self.spectral_radius
    
    def finalize_connectivity(self):
        """ Finalize both input and recurrent connectivity
            This function is primarily implemented so users don't have to 
            separately call rec and inp connectivity functions
        """
        if self.rec_finalized == False:
            self.finalize_rec_connectivity()
        if self.inp_finalized == False:
            self.finalize_inp_connectivity()

    def finalize_rec_connectivity(self):
        """ Fill rest of recurrent connections with zeros
            Ensure finalized flag is set to true
        """
        for region in self.region_dict:
            self.__get_full_connectivity(self.region_dict[region])
        # Set spectral radius using desired value
        if self.spectral_radius is not None:
            self.set_spectral_radius()
        self.rec_finalized = True

    def finalize_inp_connectivity(self):
        """ Fill rest of input connections with zeros
            Ensure finalized flag is set to true
        """
        for inp in self.inp_dict:
            self.__get_full_connectivity(self.inp_dict[inp])
        self.inp_finalized = True
    
    def compute_spectral_radius(self, weight):
        """_summary_

        Args:
            weight (torch.Tensor): recurrent weight matrix

        Returns:
            float: spectral radius of weight
        """
        # Largest absolute eigenvalue of W_rec
        eig_vals = torch.linalg.eigvals(weight)
        abs_eig_vals = eig_vals.abs()
        spectral_radius = abs_eig_vals.max()
        return spectral_radius

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
                # gather all connection properties
                region_data = dict_[cur_region].connections[connection]
                # collect parameter and associated masks
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

    def get_tonic_inp(self):
        """
        Collects baseline firing rates for all regions.

        Returns:
            torch.Tensor: Vector of baseline firing rates
        """
        return torch.cat([region.base_firing for region in self.region_dict.values()]).to(self.device) 
    
    def named_rec_regions(self, prefix=''):
        """ Loop through rec region names and objects

        Args:
            prefix (str, optional): Defaults to ''.
        """
        for name, region in self.region_dict.items():
            yield prefix + name, region

    def named_inp_regions(self, prefix=''):
        """ Loop through inp region names and objects

        Args:
            prefix (str, optional): Defaults to ''.
        """
        for name, region in self.inp_dict.items():
            yield prefix + name, region

    def forward(self, xn, inp, *args, noise=True):
        """
        Forward pass through the network.
        Implements discretized equation of the form: x_(t+1) = x_t + alpha * (-x_t + Wh + W_ix + b)
        Applies activation function h_(t+1) = sigma(x_(t+1))

        Args:
            xn (torch.Tensor): Pre-activation hidden state
            input: Weighted input into the network (must be same shape as specified in config)
            args: Additional unweighted network input
            noise (bool): Whether to apply noise. Defaults to True

        Returns:
            torch.Tensor: Network output sequence
        """
        assert len(self.region_dict) > 0
        assert len(self.inp_dict) > 0

        # Apply Dale's Law if constrained
        W_rec, W_rec_mask, W_rec_sign_matrix = self.gen_w(self.region_dict)
        if self.constrained:
            W_rec = self.apply_dales_law(W_rec, W_rec_mask, W_rec_sign_matrix, self.lower_bound_rec, self.upper_bound_rec)
        
        # Apply to input weights as well
        W_inp, W_inp_mask, W_inp_sign_matrix = self.gen_w(self.inp_dict)
        if self.constrained:
            W_inp = self.apply_dales_law(W_inp, W_inp_mask, W_inp_sign_matrix, self.lower_bound_inp, self.upper_bound_inp)
        
        baseline_inp = self.get_tonic_inp()
        
        xn_next = xn
        hn_next = xn.clone()

        # Create lists for xs and hns
        new_hs = []
        new_xs = []

        # Specify sequence length defined by input
        if self.batch_first:
            seq_len = inp.shape[1]
            batch_shape = inp.shape[0]
        else:
            seq_len = inp.shape[0]
            batch_shape = inp.shape[1]
        
        # Process sequence
        for t in range(seq_len):

            # Calculate noise terms
            if noise:
                # Calculate noise constants
                const_hid = (1 / self.alpha) * np.sqrt(2 * self.alpha * self.sigma_recur**2)
                const_inp = (1 / self.alpha) * np.sqrt(2 * self.alpha * self.sigma_input**2)
                # Sample from normal distribution and scale by constant term
                # Separate noise levels will be applied to each neuron/input
                perturb_hid = const_hid * torch.randn(size=(batch_shape, self.total_num_units), device=self.device)
                perturb_inp = const_inp * torch.randn(size=(batch_shape, self.total_num_inputs), device=self.device)
            else:
                perturb_hid = perturb_inp = 0

            # Update hidden state
            # Discretized equation of the form: x_(t+1) = x_t + alpha * (-x_t + Wh + W_ix + b)
            xn_next = (xn_next 
                        + self.alpha 
                        * (-xn_next
                            + (W_rec @ hn_next.T).T
                            + baseline_inp
                            + perturb_hid
                        )
            )

            # Add input to the network
            if self.batch_first:
                # Apply input noise
                inp_t = inp[:, t, :] + perturb_inp
                xn_next = xn_next + self.alpha * (W_inp @ inp_t.T).T
                # Add any remaining inputs without weights
                # Example of when this would be useful is for optogenetic manipulations
                for idx in range(len(args)):
                    xn_next = xn_next + self.alpha * args[idx][:, t, :]
            else:
                # Same as above but for different shaped input
                inp_t = inp[t, :, :] + perturb_inp
                xn_next = xn_next + self.alpha * (W_inp @ inp_t.T).T
                for idx in range(len(args)):
                    xn_next = xn_next + self.alpha * args[idx][t, :, :]

            # Compute activation
            # Gather activation and pre-activation into lists
            # Activation of the form: h_t = sigma(x_t)
            hn_next = self.activation(xn_next)
            new_xs.append(xn_next)
            new_hs.append(hn_next)
        
        # Correct the final output sizes based on input shape
        if self.batch_first:
            x_final = torch.stack(new_xs, dim=1)
            h_final = torch.stack(new_hs, dim=1)
        else:
            x_final = torch.stack(new_xs, dim=0)
            h_final = torch.stack(new_hs, dim=0)
        
        return x_final, h_final

    def __create_def_values(self, config):
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
            for param in DEFAULT_REC_CONNECTIONS:
                if param not in connection:
                    connection[param] = DEFAULT_REC_CONNECTIONS[param]

        # Set default values for input regions
        for i, region in enumerate(config["input_regions"]):
            for param in DEFAULT_INP_REGIONS:
                if param not in region:
                    if param == "name":
                        region[param] = DEFAULT_INP_REGIONS[param] + str(i)
                    else:
                        region[param] = DEFAULT_INP_REGIONS[param]

        # Set default values for input region connections
        for connection in config["input_connections"]:
            for param in DEFAULT_INP_CONNECTIONS:
                if param not in connection:
                    connection[param] = DEFAULT_INP_CONNECTIONS[param]

    def __gen_region_mask(self, region):
        """
        Generates a mask for a specific region and optionally a cell type.

        Args:
            region (str): Region name
            cell_type (str, optional): Cell type within region. Defaults to None

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
    
    def __constrained_default_init_rec(self, weight):
        """_summary_

        Args:
            weight (_type_): _description_

        Returns:
            _type_: _description_
        """
        weight.data = torch.normal(mean=0, std=np.sqrt(1 / (2*self.total_num_units)), size=weight.shape, device=self.device)
        mask = torch.sign(weight.data)
        weight.data *= mask
        return weight

    def __constrained_default_init_inp(self, weight):
        """_summary_

        Args:
            weight (_type_): _description_

        Returns:
            _type_: _description_
        """
        weight.data = torch.normal(mean=0, std=np.sqrt(1 / (self.total_num_units + self.total_num_inputs)), size=weight.shape, device=self.device)
        mask = torch.sign(weight.data)
        weight.data *= mask
        return weight