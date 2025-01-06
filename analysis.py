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
from mRNNTorch.utils import get_region_activity, linearize_trajectory
from tqdm import tqdm

def linearized_eigendecomposition(mrnn, x, start_region=None, end_region=None, start_cell_type=None, end_cell_type=None):
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

    jacobian = linearize_trajectory(mrnn, x, start_region, end_region)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    
    # Split real and imaginary parts
    reals = []
    for eigenvalue in eigenvalues:
        reals.append(eigenvalue.real)

    ims = []
    for eigenvalue in eigenvalues:
        ims.append(eigenvalue.imag)

    return reals, ims, eigenvectors

def psth(mrnn, act, average=True):
    """Gather the PSTH for each region in the network

    Args:
        rnn (mRNN): mRNN instance
        act (Torch.Tensor): Tensor of mRNN activity during a trial

    Returns:
        Dictionary: Contains average activity for each individual region
    """
        
    activity_dict = []
    for region in mrnn.region_dict:
        if len(mrnn.region_dict[region].cell_type_info) > 0:
            for cell_type in mrnn.region_dict[region].cell_type_info:
                if average == True:
                    mean_act = np.mean(get_region_activity(mrnn, region, act, start_cell_type=cell_type), axis=-1)
                else:
                    mean_act = get_region_activity(mrnn, region, act, start_cell_type=cell_type)
                activity_dict.append(mean_act)
        else:
            if average == True:
                mean_act = np.mean(get_region_activity(mrnn, region, act), axis=-1)
            else:
                mean_act = get_region_activity(mrnn, region, act)
            activity_dict.append(mean_act)
    
    return activity_dict

def flow_field(
    mrnn, 
    trajectory,
    input,
    time_skips=1, 
    num_points=50,
    lower_bound_x=-10,
    upper_bound_x=10,
    lower_bound_y=-10,
    upper_bound_y=10,
    start_region=None,
    end_region=None,
    start_region_cell_type=None,
    end_region_cell_type=None,
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
        start_region (str): _description_
        end_region (str): _description_
        start_region_cell_type (str): _description_
        end_region_cell_type (str): _description_

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
    xn_temp = mrnn.get_initial_condition(xn_temp)

    # Gather activity for specified region and cell type
    temp_act = get_region_activity(
        mrnn, 
        start_region, 
        trajectory, 
        end_region=end_region, 
        start_cell_type=start_region_cell_type, 
        end_cell_type=end_region_cell_type,
    )

    # Reshape activity before performing PCA
    temp_act = torch.reshape(temp_act, shape=(temp_act.shape[0] * temp_act.shape[1], temp_act.shape[2])) 
    temp_act = temp_act.detach().cpu().numpy()

    # Do PCA on the specified region
    x_pca.fit(temp_act)

    # Inverse PCA to input grid into network
    grid = x_pca.inverse_transform(data_coords)
    grid = grid.clone().detach().cuda()

    # Initialize activity dict
    next_acts = {}
    # Repeat over the sequence length dimension to match activity
    grid = grid.unsqueeze(1).repeat(1, trajectory.shape[1], 1)
    # Repeat along the batch dimension to match the grid
    full_act_batch = trajectory.repeat(grid.shape[0], 1, 1)

    # Gather batches of grids with trial activity at each timestep
    start_reached = False
    end_reached = False
    x_0_flow = []

    # In order to get next step activity for grid, each region not being represented
    # By the grid will assume their values from the original trajectory passed in at 
    # Each timestep. Here we will gather activity for the grid and properly append the 
    # non-specified regions to the activity tensor to get the full activation for the network
    
    for region in mrnn.region_dict:
        if region == start_region:
            x_0_flow.append(grid)
            start_reached = True
        if start_reached == False:
            x_0_flow.append(get_region_activity(mrnn, region, full_act_batch))
        if start_reached and end_reached:
            x_0_flow.append(get_region_activity(mrnn, region, full_act_batch))
        if region == end_region:
            end_reached = True
    x_0_flow = torch.cat(x_0_flow, dim=-1)
            
    for t in tqdm(range(0, x_0_flow.shape[1], time_skips)):
        with torch.no_grad():
            # Current timestep input
            cur_iti_inp = input[:, t:t+1, :]
            # Get activity for current timestep
            # Since we are changing the initial condition at each timestep, we need to iterate through each timestep here
            _, h = mrnn(x_0_flow[:, t, :], cur_iti_inp, noise=False)
            # Get activity for regions of interest
            cur_region_h = get_region_activity(mrnn, start_region, h, end_region=end_region)
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
