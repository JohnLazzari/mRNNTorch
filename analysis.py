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
from mRNNTorch.utils import get_region_activity, linearize_trajectory, get_initial_condition, get_weight_subset
from tqdm import tqdm

def linearized_eigendecomposition(mrnn, x, *args, alpha=1):
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
    jacobian = linearize_trajectory(mrnn, x, *args, alpha=alpha)
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
    x_offset=1,
    y_offset=1,
    region_list=None,
    stim_input=None,
    cancel_other_regions=False
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
    # there must be a batch dimension even if it can only be 1 for now
    batch_size = trajectory.shape[0]
    seq_len = trajectory.shape[1]

    # Lists for x and y velocities
    x_vels = []
    y_vels = []
    data_coords_list = []

    # Speed dictionary for energy landscape
    speeds = []

    # Gather data
    x_pca = PCA(n_components=2)

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

    # Now going through trajectory
    for t in tqdm(range(1, seq_len, time_skips)):

        # We want to find the best grid
        # To do so, find where the last timestep is in pc space and center a grid around that
        proj = x_pca.transform(temp_act[:t, :])
        latest_t = proj[-1, :]

        lower_bound_x = np.round(latest_t[0] - x_offset, decimals=1)
        upper_bound_x = np.round(latest_t[0] + x_offset, decimals=1)
        lower_bound_y = np.round(latest_t[1] - y_offset, decimals=1)
        upper_bound_y = np.round(latest_t[1] + y_offset, decimals=1)

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
        data_coords_flatten = torch.flatten(data_coords, start_dim=0, end_dim=1)

        # Inverse PCA to input grid into network
        grid = x_pca.inverse_transform(data_coords_flatten)
        grid = grid.clone().detach().cuda()

        # Repeat over the sequence length dimension to match activity
        grid = grid.unsqueeze(1).repeat(1, seq_len, 1)
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

        with torch.no_grad():
            # Current timestep input
            inp_t = inp[:, t:t+1, :]
            # Get activity for current timestep
            # Since we are changing the initial condition at each timestep, we need to iterate through each timestep here
            if stim_input is not None:
                stim_input_t = stim_input[:, t:t+1, :]
            else:
                stim_input_t = torch.zeros_like(inp_t)

            _, h = mrnn(x_0_flow[:, t, :], inp_t, stim_input_t, noise=False)

        # Get activity for regions of interest
        temp_region_acts = []
        for region in region_list:
            temp_act_cur = get_region_activity(mrnn, h, region)
            temp_region_acts.append(temp_act_cur)
        cur_region_h = torch.cat(temp_region_acts, dim=-1).squeeze().detach().cpu().numpy()
        
        # Reshape data back to grid
        data_coords = data_coords.numpy()
        data_coords_list.append(data_coords)

        cur_region_h = x_pca.transform(cur_region_h)
        cur_region_h = np.reshape(cur_region_h, (num_points, num_points, cur_region_h.shape[-1]))

        # Compute velocities between gathered trajectory of grid and original grid values
        x_vel = cur_region_h[:, :, 0] - data_coords[:, :, 0]
        y_vel = cur_region_h[:, :, 1] - data_coords[:, :, 1]
        x_vels.append(x_vel)
        y_vels.append(y_vel)
        
        # Take the magnitude of the previously calculated velocities
        speed = np.sqrt(x_vel**2 + y_vel**2)
        # Normalization
        c = speed / speed.max()
        speeds.append(c)
    
    return data_coords_list, x_vels, y_vels, speeds


def linear_flow_field(
    mrnn, 
    trajectory,
    inp,
    time_skips=1, 
    num_points=50,
    x_offset=1,
    y_offset=1,
    region_list=None,
    stim_input=None,
    cancel_other_regions=False,
    alpha=1
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

    # Set a random seed for reproducibility
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    jac_list = []

    # Assuming batch_first=True for now
    # there must be a batch dimension even if it can only be 1 for now
    batch_size = trajectory.shape[0]
    seq_len = trajectory.shape[1]

    # Lists for x and y velocities
    x_vels = []
    y_vels = []
    data_coords_list = []

    # Speed dictionary for energy landscape
    speeds = []

    # Gather data
    x_pca = PCA(n_components=2, svd_solver="full", random_state=42, tol=1e-4)

    # Initialize hidden states
    xn_temp = torch.zeros(size=(batch_size, mrnn.total_num_units))
    xn_temp = get_initial_condition(mrnn, xn_temp)

    # Gather activity for specified region and cell type
    temp_region_acts = []
    for region in region_list:
        temp_act_cur = get_region_activity(mrnn, trajectory, region)
        temp_region_acts.append(temp_act_cur)
    temp_act = torch.cat(temp_region_acts, dim=-1)

    # Reshape activity before performing PCA
    temp_act = torch.reshape(temp_act, shape=(temp_act.shape[0] * temp_act.shape[1], temp_act.shape[2])) 
    temp_act = temp_act.detach().cpu().numpy().astype(np.float64)

    # Do PCA on the specified region(s)
    x_pca.fit(temp_act)

    # Now going through trajectory
    for t in tqdm(range(1, seq_len, time_skips)):

        # We want to find the best grid
        # To do so, find where the last timestep is in pc space and center a grid around that
        proj = x_pca.transform(temp_act[:t, :])
        latest_t = proj[-1, :]

        lower_bound_x = np.round(latest_t[0] - x_offset, decimals=1)
        upper_bound_x = np.round(latest_t[0] + x_offset, decimals=1)
        lower_bound_y = np.round(latest_t[1] - y_offset, decimals=1)
        upper_bound_y = np.round(latest_t[1] + y_offset, decimals=1)

        # Num points is along each axis, not in total
        x = np.linspace(lower_bound_x, upper_bound_x, num_points)
        y = np.linspace(lower_bound_y, upper_bound_y, num_points)

        # Gather 2D grid for flow fields
        xv, yv = np.meshgrid(x, y)
        xv = np.expand_dims(xv, axis=-1)
        yv = np.expand_dims(yv, axis=-1)

        # Convert the grid to a tensor and flatten for PCA
        data_coords = np.concatenate((xv, yv), axis=-1)
        data_coords = torch.tensor(data_coords, dtype=torch.float64)
        data_coords_flatten = torch.flatten(data_coords, start_dim=0, end_dim=1)

        # Inverse PCA to input grid into network
        grid = x_pca.inverse_transform(data_coords_flatten)
        grid = grid.clone().detach()
        x_0_flow = grid - torch.tensor(temp_act[t, :])

        # In order to get next step activity for grid, each region not being represented
        # By the grid will assume their values from the original trajectory passed in at 
        # Each timestep. Here we will gather activity for the grid and properly append the 
        # non-specified regions to the activity tensor to get the full activation for the network

        with torch.no_grad():
            # Current timestep input
            inp_t = inp[:, t:t+1, :]
            # Get activity for current timestep
            # Since we are changing the initial condition at each timestep, we need to iterate through each timestep here
            if stim_input is not None:
                stim_input_t = stim_input[:, t:t+1, :]
            else:
                stim_input_t = torch.zeros_like(inp_t)

            jac_rec = linearize_trajectory(mrnn, trajectory.squeeze()[t, :], *region_list, alpha=alpha)
            jac_rec = jac_rec.to(torch.float64)
            jac_list.append(jac_rec)

            x_0_flow = x_0_flow.to(torch.float64)
            h = torch.tensor(temp_act[t, :]) + (jac_rec @ x_0_flow.T).T + stim_input_t.squeeze()

        # Do this to mitigate floating point erros
        # Subtract to get perturbation in state space
        h = h.detach().cpu().numpy()
        
        # Reshape data back to grid
        data_coords = data_coords.numpy().astype(np.float64)
        data_coords_list.append(data_coords)

        cur_region_h = x_pca.transform(h)
        cur_region_h = np.reshape(cur_region_h, (num_points, num_points, cur_region_h.shape[-1])).astype(np.float64)

        # Compute velocities between gathered trajectory of grid and original grid values
        x_vel = cur_region_h[:, :, 0] - data_coords[:, :, 0]
        y_vel = cur_region_h[:, :, 1] - data_coords[:, :, 1]
        x_vels.append(x_vel)
        y_vels.append(y_vel)
        
        # Take the magnitude of the previously calculated velocities
        speed = np.sqrt(x_vel**2 + y_vel**2)
        # Normalization
        c = speed / speed.max()
        speeds.append(c)

    jac_list = torch.stack(jac_list)
    torch.save(jac_list, "jac_2.pt")
    
    return data_coords_list, x_vels, y_vels, speeds

def communication_subspace(src_region, trg_region):
    pass
