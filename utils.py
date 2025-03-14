import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def manipulation_stim(mrnn, start_silence, end_silence, seq_len, extra_steps, stim_strength, batch_size, n_steps_rampup, n_steps_rampdown, *args):

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

    mask = torch.zeros(size=(mrnn.total_num_units,), device=mrnn.device)
    for region in args:
        mask = mask + mrnn.region_mask_dict[region]
    
    total_stim_time = (end_silence - start_silence) - n_steps_rampup - n_steps_rampdown
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # It applies the inhibitory stimulus to all of the conditions specified in data (or max_seq_len) equally
    stim_pre = torch.zeros(size=(batch_size, start_silence, mrnn.total_num_units), device=mrnn.device)
    if n_steps_rampup > 0:
        stim_ramp_up = torch.linspace(0, stim_strength, n_steps_rampup).unsqueeze(0).unsqueeze(2).to(mrnn.device)
        stim_ramp_up = stim_ramp_up.repeat(batch_size, 1, mrnn.total_num_units) * mask
    stim_const = torch.ones(size=(batch_size, total_stim_time, mrnn.total_num_units), device=mrnn.device) * mask * stim_strength
    if n_steps_rampdown > 0:
        stim_ramp_down = torch.linspace(stim_strength, 0, n_steps_rampdown).unsqueeze(0).unsqueeze(2).to(mrnn.device)
        stim_ramp_down = stim_ramp_down.repeat(batch_size, 1, mrnn.total_num_units) * mask
    stim_post = torch.zeros(size=(batch_size, (seq_len - end_silence) + extra_steps, mrnn.total_num_units), device=mrnn.device)
    stim = torch.cat([stim_pre, stim_ramp_up, stim_const, stim_ramp_down, stim_post], dim=1)

    return stim

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
    unique_regions = set(args)
    if not args:
        return act
    # Check to ensure region is recurrent
    for region in args:
        if region in mrnn.inp_dict:
            raise Exception("Can only get activity for recurrent regions")
    # Gather all of the specified region activities in here
    region_acts = []
    # Go and check if any parent regions are entered
    for region in unique_regions.copy():
        if __check_if_parent_region(mrnn, region):
            unique_regions.remove(region)
            unique_regions.update(__get_child_regions(mrnn, region))
    # If end region is specified, make new end_idx
    for region in unique_regions:
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

    # Store regions if parent regions are given    
    to_regions = []
    from_regions = []

    if to and from_:

        # If to region is a parent region, then get children regions
        if __check_if_parent_region(mrnn, to):
            to_regions.extend(__get_child_regions(mrnn, to))
        else:
            to_regions.append(to)
        # If from region is a parent region, then get children regions
        if __check_if_parent_region(mrnn, from_):
            from_regions.extend(__get_child_regions(mrnn, from_))
        else:
            from_regions.append(from_)
        
        # Check which weight matrix to use based on from region
        if from_ in mrnn.region_dict:
            # Gather recurrent weight matrix
            weight, mask, sign = mrnn.gen_w(mrnn.region_dict)
            if mrnn.constrained == True:
                weight = mrnn.apply_dales_law(weight, mask, sign)
        else:
            # Gather input weight matrix
            weight, mask, sign = mrnn.gen_w(mrnn.inp_dict)
            if mrnn.constrained == True:
                weight = mrnn.apply_dales_law(weight, mask, sign)
        
        # Store all of the weights to region from another
        to_from_weight = []
        # Now go through each of the collected regions and get the weights
        for to_region in to_regions:
            from_weight = []
            # Get the indices for to region
            to_start_idx, to_end_idx = get_region_indices(mrnn, to_region)
            for from_region in from_regions:
                # Gather indices
                from_start_idx, from_end_idx = get_region_indices(mrnn, from_region)
                from_weight.append(weight[to_start_idx:to_end_idx, from_start_idx:from_end_idx])
            # Collect all of the weights for from region
            collected_from_weight = torch.cat(from_weight, dim=1)
            to_from_weight.append(collected_from_weight)
        # Collect final weight matrix
        collected_to_from_weight = torch.cat(to_from_weight, dim=0)
        return collected_to_from_weight

    # Gather original weight matrix and apply Dale's Law if constrained
    # Can only be recurrent if not using to and from
    mrnn_weight, W_rec_mask, W_rec_sign = mrnn.gen_w(mrnn.region_dict)
    if mrnn.constrained == True:
        mrnn_weight = mrnn.apply_dales_law(mrnn_weight, W_rec_mask, W_rec_sign)
    
    # Default to standard weight matrix if no regions are provided
    if not args:
        return mrnn_weight
    
    # Check if user specifies input region through args instead of to, from
    for region in args:
        if region in mrnn.inp_dict:
            raise Exception("Can only gather input subsets using to and from arguments")

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
    global_weight_collection = torch.cat(global_weight_collection, dim=0)
    
    return global_weight_collection

def linearize_trajectory(mrnn, x, *args, W_inp=None, alpha=1):
    """ Find jacobian of network Taylor series expansion

    Args:
        mrnn (_type_): _description_
        x (_type_): _description_
        start_region (_type_): _description_
        end_region (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert x.dim() == 1
    # Get the subset of the weights required for jacobian 
    weight_subset = get_weight_subset(mrnn, *args)
    # linearize the dynamics about state
    x_sub = get_region_activity(mrnn, x, *args) 

    # Manually computing jacobians for now
    """
        Taking jacobian of x with respect to F
        In this case, the form should be:
            J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
    """

    if mrnn.activation_name == "relu": 

        # Identity representing -x in state equation
        I = torch.eye(n=x_sub.shape[0], device=mrnn.device)

        # Implementing h'(x), diagonalize to multiply by W
        x_act = F.relu(x_sub)
        d_x_act = torch.where(x_act > 0., 1., 0.)
        d_x_act_diag = torch.diag(d_x_act)

        d_x_act_diag = d_x_act_diag.to(torch.float64)
        weight_subset = weight_subset.to(torch.float64)

        # Get final jacobian using form above
        jacobian = (I - alpha * I) + alpha * (d_x_act_diag @ weight_subset)

        # If an input weight is specified
        if W_inp != None:
            # No need for identity in this case
            jacobian_inp = alpha * W_inp
            return jacobian, jacobian_inp

    elif mrnn.activation_name == "linear":
        # Linear network jacobian is the same
        jacobian = (I - alpha * I) + alpha * weight_subset
    
    return jacobian

def get_region_indices(mrnn, region):
    """
    Gets the start and end indices for a specific region in the hidden state vector.

    Args:
        region (str): Name of the region

    Returns:
        tuple: (start_idx, end_idx)
    """
    
    if __check_if_parent_region(mrnn, region):
        raise ValueError("Can only get indices of a single region, not parent region")
    # Get the region indices
    start_idx = 0
    end_idx = 0

    # Check whether or not specified region is input or rec
    # This is to handle indices for both rec and inp regions
    if region in mrnn.region_dict:
        dict_ = mrnn.region_dict
    elif region in mrnn.inp_dict:
        dict_ = mrnn.inp_dict
    else:
        raise Exception("Not an input or recurrent region")

    for cur_reg in dict_:
        region_units = dict_[cur_reg].num_units
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

def __check_if_parent_region(mrnn, parent_region):
    """ Check if the given region is a parent region or not

    Args:
        mrnn (_type_): _description_
        region (_type_): _description_

    Returns:
        _type_: _description_
    """
    for region in mrnn.region_dict.values():
        if region.parent_region == parent_region:
            return True
    return False

def __get_child_regions(mrnn, parent_region):
    """ Check if the given region is a parent region or not

    Args:
        mrnn (_type_): _description_
        region (_type_): _description_

    Returns:
        _type_: _description_
    """
    child_region_list = []
    for region in mrnn.region_dict:
        if mrnn.region_dict[region].parent_region == parent_region:
            child_region_list.append(region)
    return tuple(child_region_list)