import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

def manipulation_stim(mrnn, start_silence, end_silence, seq_len, extra_steps, stim_strength, batch_size, *args):

    """
    Get inhibitory or excitatory stimulus for optogenetic replication
    Function will gather the mask for the specified region and cell type then make a stimulus targeting these regions

    Returns:
        rnn:                        mRNN to silence
        regions_cell_types:         List of tuples specifying the region and corresponding cell type to get a mask for
        start_silence:              inteer index of when to start perturbations in the sequence
        end_silence:                integer index of when to stop perturbations in the sequence
        max_seq_len:                max sequence length
        extra_steps:                Number of extra steps to add to the sequence if necessary
        stim_strength:              Floating point value that specifies how strong the perturbation is (- or +)
        batch_size:                 Number of conditions to be included in the sequence
    """

    mask = torch.zeros(size=(mrnn.total_num_units,), device="cuda")
    for region in args:
        cur_mask = stim_strength * (mrnn.region_mask_dict[region])
        mask = mask + cur_mask
    
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # It applies the inhibitory stimulus to all of the conditions specified in data (or max_seq_len) equally
    inhib_stim_pre = torch.zeros(size=(batch_size, start_silence, mrnn.total_num_units), device="cuda")
    inhib_stim_silence = torch.ones(size=(batch_size, end_silence - start_silence, mrnn.total_num_units), device="cuda") * mask
    inhib_stim_post = torch.zeros(size=(batch_size, (seq_len - end_silence) + extra_steps, mrnn.total_num_units), device="cuda")
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)
    
    return inhib_stim

def get_region_activity(mrnn, act, *args):
    """
    Takes in hn and the specified region and returns the activity hn for the corresponding region

    Args:
        region (str): Name of the region
        hn (Torch.Tensor): tensor containing model hidden activity. Activations must be in last dimension (-1)

    Returns:
        region_hn: tensor containing hidden activity only for specified region
    """
    # Default to returning whole activity
    if not args:
        return act
    # Gather all of the specified region activities in here
    region_acts = []
    # If end region is specified, make new end_idx
    for region in args:
        start_idx, end_idx = get_region_indices(mrnn, region)
        region_acts.append(act[..., start_idx:end_idx])
    # Now concatenate all of the region activities
    region_acts = torch.cat(region_acts, dim=-1)
    return region_acts

def get_weight_subset(mrnn, *args, to=None, from_=None):
    """ Gather a subset of the weights

    Args:
        mrnn (_type_): _description_
        start_region (_type_, optional): _description_. Defaults to None.
        end_region (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Gather original weight matrix and apply Dale's Law if constrained
    mrnn_weight, W_rec_mask, W_rec_sign = mrnn.gen_w(mrnn.region_dict)
    if mrnn.constrained == True:
        mrnn_weight = mrnn.apply_dales_law(mrnn_weight, W_rec_mask, W_rec_sign)
    
    if to and from_:
        to_start_idx, to_end_idx = get_region_indices(mrnn, to)
        from_start_idx, from_end_idx = get_region_indices(mrnn, from_)
        return mrnn_weight[to_start_idx:to_end_idx, from_start_idx:from_end_idx].detach().cpu().numpy()
    
    # Default to standard weight matrix if no regions are provided
    if not args:
        return mrnn_weight

    # This is used to store the final collected weight matrix
    global_weight_collection = [] 

    # Loop through each region
    for src_region in args:
        # This will be used to collect all of the weights going TO the source region
        # This should gather weights along columns
        src_weight_collection = []
        src_start_idx, src_end_idx = get_region_indices(mrnn, src_region)
        # Now loop through all connections going TO source region
        for dst_region in args:
            # Get the region indices and capture the weight subset
            dst_start_idx, dst_end_idx = get_region_indices(mrnn, dst_region)
            weight_subset = mrnn_weight[src_start_idx:src_end_idx, dst_start_idx:dst_end_idx]
            # Append to src collection
            src_weight_collection.append(weight_subset)
        # Concatenate all of the weights in order to create the specified subset as a tensor
        src_weight_collection = torch.cat(src_weight_collection, dim=1)
        global_weight_collection.append(src_weight_collection)
    # Similar to before but now concatenating along rows
    global_weight_collection = torch.cat(global_weight_collection, dim=0).detach().cpu().numpy()
    
    return global_weight_collection

def linearize_trajectory(mrnn, x, *args):
    """ Find jacobian of network Taylor series expansion

    Args:
        mrnn (_type_): _description_
        x (_type_): _description_
        start_region (_type_): _description_
        end_region (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get the subset of the weights required for jacobian 
    weight_subset = get_weight_subset(mrnn, *args)
    # linearize the dynamics about state
    x_sub = get_region_activity(mrnn, x, *args) 
    # Manually computing jacobians
    # Shouldn't be that hard
    if mrnn.activation_name == "relu": 
        x_act = F.relu(x_sub)
        d_x_act = torch.where(x_act > 0, 1., 0.)
        d_x_act_diag = torch.diag(d_x_act)
        jacobian = d_x_act_diag @ weight_subset.T
    elif mrnn.activation_name == "linear":
        jacobian = weight_subset.T
    
    return jacobian

def get_region_indices(mrnn, region):
    """
    Gets the start and end indices for a specific region in the hidden state vector.

    Args:
        region (str): Name of the region

    Returns:
        tuple: (start_idx, end_idx)
    """
    
    # Get the region indices
    start_idx = 0
    end_idx = 0
    for cur_reg in mrnn.region_dict:
        region_units = mrnn.region_dict[cur_reg].num_units
        if cur_reg == region:
            end_idx = start_idx + region_units
            break
        start_idx += region_units
    
    return start_idx, end_idx

def get_initial_condition(mrnn, xn):
    """ Create an initial xn for the network

    Args:
        mrnn (_type_): _description_
        xn (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Initialize x and h
    for region in mrnn.region_dict:
        start_idx, end_idx = get_region_indices(mrnn, region)
        xn[..., start_idx:end_idx] = mrnn.region_dict[region].init
    return xn