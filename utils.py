import numpy as np
from sklearn.decomposition import PCA
import torch

def manipulation_stim(mrnn, start_silence, end_silence, seq_len, extra_steps, stim_strength, batch_size, region_list=None, region_cell_type_list=None):

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

    # Select mask based on region being silenced
    mask = torch.zeros(size=(mrnn.total_num_units,), device="cuda")
    for region, cell_type in regions_cell_types:
        if cell_type is not None:
            cur_mask = stim_strength * (mrnn.region_mask_dict[region][cell_type])
        else:
            cur_mask = stim_strength * (mrnn.region_mask_dict[region]["full"])
        mask = mask + cur_mask
    
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # It applies the inhibitory stimulus to all of the conditions specified in data (or max_seq_len) equally
    inhib_stim_pre = torch.zeros(size=(batch_size, start_silence, mrnn.total_num_units), device="cuda")
    inhib_stim_silence = torch.ones(size=(batch_size, end_silence - start_silence, mrnn.total_num_units), device="cuda") * mask
    inhib_stim_post = torch.zeros(size=(batch_size, (seq_len - end_silence) + extra_steps, mrnn.total_num_units), device="cuda")
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)
    
    return inhib_stim

def get_region_indices(mrnn, region, cell_type=None):
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
    
    # If cell type is specified, get the cell type indices
    if cell_type is not None:
        for cell in mrnn.region_dict[region].cell_type_info:
            # Gather necessary information to gather number of units for cell type
            region_units = mrnn.region_dict[region].num_units
            cell_percentage = mrnn.region_dict[region].cell_type_info[cell]
            cell_units = int(round(region_units * cell_percentage))
            if cell == cell_type:
                end_idx = start_idx + cell_units
                break
            start_idx += cell_units
        
    return start_idx, end_idx

def get_region_activity(mrnn, start_region, act, end_region=None, start_cell_type=None, end_cell_type=None):
    """
    Takes in hn and the specified region and returns the activity hn for the corresponding region

    Args:
        region (str): Name of the region
        hn (Torch.Tensor): tensor containing model hidden activity. Activations must be in last dimension (-1)

    Returns:
        region_hn: tensor containing hidden activity only for specified region
    """
    # Get start and end positions of region
    start_idx, end_idx = mrnn.get_region_indices(start_region, cell_type=start_cell_type)
    # If end region is specified, make new end_idx
    if end_region is not None:
        _, end_idx = mrnn.get_region_indices(end_region, cell_type=end_cell_type)
    # Gather specified regional activity
    region_act = act[..., start_idx:end_idx]
    return region_act

def get_weight_subset(mrnn, start_region=None, end_region=None, start_region_cell_type=None, end_region_cell_type=None):

    start_idx = 0 
    end_idx = mrnn.total_num_units
    
    W_rec, W_rec_mask, W_rec_sign = mrnn.gen_w(mrnn.region_dict)
    mrnn_weight = mrnn.apply_dales_law(W_rec, W_rec_mask, W_rec_sign)

    if start_region is not None:
        start_idx, _ = mrnn.get_region_indices(start_region, cell_type=start_region_cell_type)
    if end_region is not None:
        _, end_idx = mrnn.get_region_indices(end_region, cell_type=end_region_cell_type)

    weight_subset = mrnn_weight[start_idx:end_idx, 
                                start_idx:end_idx]
    weight_subset = weight_subset.detach().cpu().numpy()
    
    return weight_subset