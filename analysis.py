import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from mRNNTorch.utils import get_region_activity, linearize_trajectory, get_initial_condition
from tqdm import tqdm

def linearized_eigendecomposition(mrnn, x, *args):
    """Linearize the network and compute eigenvalues at each timestep

    Args:
        mrnn (mRNN): mRNN instance
        x (Torch.Tensor): 
        start_region (str, optional): Beginning of subset of weights to compute jacobian of. Defaults to None.
        end_region (str, optional): End of subset of weights to compute jacobian of. Defaults to None.
        start_cell_type (str, optional): Specify cell type for start region. Defaults to None.
        end_cell_type (str, optional): Specify cell type for end region. Defaults to None.

    Returns:
        list: Real eigenvalues
        list: Imaginary eigenvalues
        np.array: eigenvectors
    """
    jacobian = linearize_trajectory(mrnn, x, *args)
    eigenvalues, eigenvectors = torch.linalg.eig(jacobian)
    
    # Split real and imaginary parts
    reals = []
    for eigenvalue in eigenvalues:
        reals.append(eigenvalue.real.item())

    ims = []
    for eigenvalue in eigenvalues:
        ims.append(eigenvalue.imag.item())

    return reals, ims, eigenvectors

def psth(mrnn, act, *args, average=True):
    """Gather the PSTH for each region in the network

    Args:
        rnn (mRNN): mRNN instance
        act (Torch.Tensor): Tensor of mRNN activity during a trial

    Returns:
        Dictionary: Contains average activity for each individual region
    """
        
    activity_dict = []
    for region in args:
        if average == True:
            mean_act = torch.mean(get_region_activity(mrnn, act, region), axis=-1)
        else:
            mean_act = get_region_activity(mrnn, act, region)
        activity_dict.append(mean_act)
    
    return activity_dict

def flow_field(
    mrnn, 
    trajectory,
    inp,
    time_skips=1, 
    num_points=50,
    lower_bound_x=-10,
    upper_bound_x=10,
    lower_bound_y=-10,
    upper_bound_y=10,
    region_list=None,
    stim_input=None,
    cancel_other_regions=False,
    linearize=False
):
    """ Generate flow fields and energy landscapes of mRNN activity
        Allows for specifying certain subregions to obtain velocities for
        Regions that are not being tested will assume their control activity for given input while gathering specified regional velocities

    Args:
        mrnn (mRNN): _description_
        trajectory (Torch.Tensor): _description_
        input (Torch.Tensor): _description_
        time_skips (int): _description_
        num_points (int): _description_
        lower_bound_x (int): _description_
        upper_bound_x (int): _description_
        lower_bound_y (int): _description_
        upper_bound_y (int): _description_
        region_cell_type_list (list): 

    Returns:
        _type_: _description_
    """
    
    # Assuming batch_first=True for now
    batch_size = trajectory.shape[0]
    seq_len = trajectory.shape[1]

    # Gather data
    x_pca = PCA(n_components=2)

    # Num points is along each axis, not in total
    x = np.linspace(lower_bound_x, upper_bound_x, num_points)
    y = np.linspace(lower_bound_y, upper_bound_y, num_points)

    # Gather 2D grid for flow fields
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)

    # Convert the grid to a tensor and flatten for PCA
    data_coords = np.concatenate((xv, yv), axis=-1)
    data_coords = torch.tensor(data_coords, dtype=torch.float32)
    data_coords = torch.flatten(data_coords, start_dim=0, end_dim=1)

    # Initialize hidden states
    xn_temp = torch.zeros(size=(batch_size, mrnn.total_num_units), device="cuda")
    xn_temp = get_initial_condition(mrnn, xn_temp)

    # Gather activity for specified region and cell type
    temp_region_acts = []
    for region in region_list:
        temp_act_cur = get_region_activity(mrnn, trajectory, region)
        temp_region_acts.append(temp_act_cur)
    temp_act = torch.cat(temp_region_acts, dim=-1)

    # Reshape activity before performing PCA
    temp_act = torch.reshape(temp_act, shape=(temp_act.shape[0] * temp_act.shape[1], temp_act.shape[2])) 
    temp_act = temp_act.detach().cpu().numpy()

    # Do PCA on the specified region(s)
    x_pca.fit(temp_act)

    # Inverse PCA to input grid into network
    grid = x_pca.inverse_transform(data_coords)
    grid = grid.clone().detach().cuda()

    # Initialize activity dict
    next_acts = {}
    # Repeat over the sequence length dimension to match activity
    grid = grid.unsqueeze(1).repeat(1, trajectory.shape[1], 1)
    # Repeat along the batch dimension to match the grid
    full_act_batch = trajectory.repeat(grid.shape[0], 1, 1).cuda()

    # Gather batches of grids with trial activity at each timestep
    grid_region_idx = 0
    x_0_flow = []

    # In order to get next step activity for grid, each region not being represented
    # By the grid will assume their values from the original trajectory passed in at 
    # Each timestep. Here we will gather activity for the grid and properly append the 
    # non-specified regions to the activity tensor to get the full activation for the network
    for region in mrnn.region_dict:
        if region in region_list:
            x_0_flow.append(grid[..., grid_region_idx:grid_region_idx + mrnn.region_dict[region].num_units])
            grid_region_idx += mrnn.region_dict[region].num_units
        else:
            if cancel_other_regions == True:
                full_batch_region_temp = get_region_activity(mrnn, full_act_batch, region)
                x_0_flow.append(torch.zeros_like(full_batch_region_temp))
            else:
                x_0_flow.append(get_region_activity(mrnn, full_act_batch, region))
    x_0_flow = torch.cat(x_0_flow, dim=-1)
            
    # Now going through 
    for t in tqdm(range(0, x_0_flow.shape[1], time_skips)):
        with torch.no_grad():
            # Current timestep input
            inp_t = inp[:, t:t+1, :]
            # Get activity for current timestep
            # Since we are changing the initial condition at each timestep, we need to iterate through each timestep here
            if stim_input is not None:
                stim_input_t = stim_input[:, t:t+1, :]
                _, h = mrnn(x_0_flow[:, t, :], inp_t, stim_input_t, noise=False)
            else:
                _, h = mrnn(x_0_flow[:, t, :], inp_t, noise=False)
        # Get activity for regions of interest
        temp_region_acts = []
        for region in region_list:
            temp_act_cur = get_region_activity(mrnn, h, region)
            temp_region_acts.append(temp_act_cur)
        cur_region_h = torch.cat(temp_region_acts, dim=-1)
        next_acts[t] = cur_region_h.squeeze().detach().cpu().numpy()
        
    # Reshape data back to grid
    data_coords = data_coords.numpy()
    data_coords = np.reshape(data_coords, (num_points, num_points, data_coords.shape[-1]))

    for t in range(0, seq_len, time_skips):
        next_acts[t] = x_pca.transform(next_acts[t])
        next_acts[t] = np.reshape(next_acts[t], (num_points, num_points, next_acts[t].shape[-1]))

    # Lists for x and y velocities
    x_vels = []
    y_vels = []

    # Compute velocities between gathered trajectory of grid and original grid values
    for i in range(0, seq_len, time_skips):
        x_vels.append(next_acts[i][:, :, 0] - data_coords[:, :, 0])
        y_vels.append(next_acts[i][:, :, 1] - data_coords[:, :, 1])
    
    # Speed dictionary for energy landscape
    speeds = []
    
    # Take the magnitude of the previously calculated velocities
    for i in range(len(x_vels)):
        speed = np.sqrt(x_vels[i]**2 + y_vels[i]**2)
        # Normalization
        c = speed / speed.max()
        speeds.append(c)
    
    return data_coords, x_vels, y_vels, speeds

def communication_subspace(src_region, trg_region):
    pass
