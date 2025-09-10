"""Analysis utilities for mRNN dynamics and structure.

Includes linearization, eigen analyses, PSTH-like summaries, and flow fields
to visualize local dynamics in low-dimensional subspaces.
"""

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
from tqdm import tqdm

def linearize_trajectory(mrnn, x, *args, W_inp=None, alpha=1):
    """Linearize the dynamics around a state and return the Jacobian.

    Computes the Jacobian of the mRNN update with respect to the hidden state
    evaluated at the provided state ``x`` and (optionally) a subset of regions
    defined by ``*args``. If ``W_inp`` is provided, also returns the Jacobian
    with respect to the input.

    Args:
        mrnn (mRNN): An initialized mRNN instance.
        x (torch.Tensor): 1D tensor representing the hidden state at which to
            linearize (shape ``[H]``).
        *args (str): Optional region names specifying a subset for the Jacobian.
        W_inp (torch.Tensor | None): Optional input weight matrix to include in
            the input Jacobian.
        alpha (float): Discretization factor used in the update.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Jacobian w.r.t. hidden
        state, and optionally (Jacobian w.r.t. input) if ``W_inp`` is provided.
    """
    assert x.dim() == 1
    # Get the subset of the weights required for jacobian 
    weight_subset = mrnn.get_weight_subset(*args)
    # linearize the dynamics about state
    x_sub = mrnn.get_region_activity(x, *args) 

    # Manually computing jacobians for now
    """
        Taking jacobian of x with respect to F
        In this case, the form should be:
            J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
    """
    
    def linear(x):
        return x

    # Identity representing -x in state equation
    I = torch.eye(n=x_sub.shape[0], device=mrnn.device)

    # Implementing h'(x), diagonalize to multiply by W
    if mrnn.activation_name == "relu":
        d_x_act_diag = torch.autograd.functional.jacobian(F.relu, x_sub)
    elif mrnn.activation_name == "linear":
        d_x_act_diag = torch.autograd.functional.jacobian(linear, x_sub)
    elif mrnn.activation_name == "tanh":
        d_x_act_diag = torch.autograd.functional.jacobian(F.tanh, x_sub)
    elif mrnn.activation_name == "sigmoid":
        d_x_act_diag = torch.autograd.functional.jacobian(F.sigmoid, x_sub)
    elif mrnn.activation_name == "softplus":
        d_x_act_diag = torch.autograd.functional.jacobian(F.softplus, x_sub)
    else:
        raise ValueError("not a valid activation function")

    # Get final jacobian using form above
    jacobian = (I - alpha * I) + alpha * (d_x_act_diag @ weight_subset)

    # If an input weight is specified
    if W_inp != None:
        # No need for identity in this case
        jacobian_inp = alpha * W_inp
        return jacobian, jacobian_inp
    
    return jacobian

def linearized_eigendecomposition(mrnn, x, *args, alpha=1):
    """Linearize the network and compute eigen decomposition.

    Args:
        mrnn (mRNN): mRNN instance.
        x (torch.Tensor): 1D hidden state where the system is linearized.
        *args (str): Optional subset of regions to consider.
        alpha (float): Discretization factor.

    Returns:
        list[float]: Real parts of eigenvalues.
        list[float]: Imag parts of eigenvalues.
        torch.Tensor: Eigenvectors stacked column-wise.
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
    """Compute per-region activity traces (PSTH-style).

    Args:
        mrnn (mRNN): mRNN instance.
        act (torch.Tensor): Activity tensor of shape ``[B, T, H]`` or ``[T, B, H]``
            depending on ``mrnn.batch_first``.
        *args (str): Region names to extract.
        average (bool): If True, average across time axis; otherwise return the
            full traces per region.

    Returns:
        list[torch.Tensor]: One tensor per region in ``args``.
    """
        
    activity_list = []
    for region in args:
        if average == True:
            mean_act = torch.mean(mrnn.get_region_activity(act, region), axis=-1)
        else:
            mean_act = mrnn.get_region_activity(act, region)
        activity_list.append(mean_act)
    
    return activity_list

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
    cancel_other_regions=False,
    follow_traj=True
):
    """Compute 2D flow fields in a region subspace along a trajectory.

    Projects selected region activity onto a 2D PCA subspace, constructs a grid
    around the current point, and advances the system by one step to estimate
    the local flow (velocity vectors). Can zero out non-selected regions or
    keep their control values.

    Args:
        mrnn (mRNN): mRNN instance.
        trajectory (torch.Tensor): Hidden activations over time.
        inp (torch.Tensor): External input sequence.
        time_skips (int): Step size when sampling along the trajectory.
        num_points (int): Number of points per axis in the grid.
        x_offset (float): Half-width of the grid in PC1.
        y_offset (float): Half-width of the grid in PC2.
        region_list (list[str] | None): Regions to include in the 2D subspace.
        stim_input (torch.Tensor | None): Optional additive stimulus input.
        cancel_other_regions (bool): If True, zero non-selected regions.
        follow_traj (bool): If True, center each grid on the current trajectory point.

    Returns:
        tuple: (coords_per_t, x_vels, y_vels, speeds) lists per sampled time.
    """

    # Get the correct batch and seq len
    if mrnn.batch_first:
        batch_size = trajectory.shape[0]
        seq_len = trajectory.shape[1]
    else:
        batch_size = trajectory.shape[1]
        seq_len = trajectory.shape[0]
    
    if region_list is None:
        region_list = [region for region in mrnn.region_dict]

    # Lists for x and y velocities
    x_vels = []
    y_vels = []
    data_coords_list = []

    # Speed dictionary for energy landscape
    speeds = []

    # Gather data
    x_pca = PCA(n_components=2)

    # Initialize hidden states
    xn_temp = torch.zeros(size=(batch_size, mrnn.total_num_units))
    xn_temp = mrnn.get_initial_condition(xn_temp)

    # Gather activity for specified region and cell type
    temp_act_cur = [mrnn.get_region_activity(trajectory, region) for region in region_list]
    temp_act = torch.cat(temp_act_cur, dim=-1)

    # Reshape activity before performing PCA
    temp_act = torch.reshape(temp_act, shape=(temp_act.shape[0] * temp_act.shape[1], temp_act.shape[2])) 
    temp_act = temp_act.numpy()

    # Do PCA on the specified region(s)
    x_pca.fit(temp_act)
    reduced_traj = x_pca.fit_transform(temp_act)

    lower_bound_x = -x_offset
    upper_bound_x = x_offset
    lower_bound_y = -y_offset
    upper_bound_y = y_offset

    # Now going through trajectory
    for t in tqdm(range(1, seq_len, time_skips)):

        # We want to find the best grid
        # To do so, find where the last timestep is in pc space and center a grid around that
        latest_t = reduced_traj[t, :]

        if follow_traj:
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

        # Repeat over the sequence length dimension to match activity
        grid = grid.unsqueeze(1).repeat(1, seq_len, 1)
        # Repeat along the batch dimension to match the grid
        full_act_batch = trajectory.repeat(grid.shape[0], 1, 1)

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
                # Get activity for non-specified regions (either from cache or compute)
                if cancel_other_regions:
                    region_activity = torch.zeros_like(mrnn.get_region_activity(full_act_batch, region))
                else:
                    region_activity = mrnn.get_region_activity(full_act_batch, region)
                x_0_flow.append(region_activity)
        # Finalize the concatenation
        x_0_flow = torch.cat(x_0_flow, dim=-1)

        with torch.no_grad():
            # Current timestep input
            inp_t = inp[:, t:t+1, :]
            # Get activity for current timestep
            # Since we are changing the initial condition at each timestep, we need to iterate through each timestep here
            if stim_input is not None:
                stim_input_t = stim_input[:, t:t+1, :]
            else:
                stim_input_t = torch.zeros_like(x_0_flow)
            
            # Make sure that h0 is the same as x_0 and that an activation is not applied, this causes issues
            _, h = mrnn(inp_t, x_0_flow[:, t, :], x_0_flow[:, t, :], stim_input_t[:, t:t+1, :], noise=False)

        # Get activity for regions of interest
        temp_region_acts = [mrnn.get_region_activity(h, region) for region in region_list]
        cur_region_h = torch.cat(temp_region_acts, dim=-1).squeeze().numpy()
        
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
    time_skips=1, 
    num_points=50,
    x_offset=1,
    y_offset=1,
    region_list=None,
    alpha=1
):
    """Compute linearized flow fields in a 2D subspace.

    Similar to :func:`flow_field`, but uses a local linear approximation (Jacobian)
    of the dynamics around points on the trajectory instead of a full forward
    step. Assumes no external input to the selected regions.

    Args:
        mrnn (mRNN): mRNN instance.
        trajectory (torch.Tensor): Hidden activations over time for selected regions.
        time_skips (int): Step size when sampling along the trajectory.
        num_points (int): Number of points per axis in the grid.
        x_offset (float): Half-width of the grid in PC1.
        y_offset (float): Half-width of the grid in PC2.
        region_list (list[str]): Regions to include in the 2D subspace.
        alpha (float): Discretization factor used in linearization.

    Returns:
        tuple: (coords_per_t, x_vels, y_vels, speeds) lists per sampled time.
    """

    # Assuming batch_first=True for now
    # there must be a batch dimension even if it can only be 1 for now
    seq_len = trajectory.shape[0]

    # Lists for x and y velocities
    x_vels = []
    y_vels = []
    data_coords_list = []

    # Speed dictionary for energy landscape
    speeds = []

    # Gather data
    x_pca = PCA(n_components=2, svd_solver="full")

    # Gather activity for specified region and cell type
    temp_region_acts = [mrnn.get_region_activity(trajectory, region) for region in region_list]
    temp_act = torch.cat(temp_region_acts, dim=-1)

    # Reshape activity before performing PCA
    temp_act = temp_act.detach().cpu()

    # Do PCA on the specified region(s)
    x_pca.fit(temp_act)
    reduced_traj = x_pca.fit_transform(temp_act)

    # Now going through trajectory
    for t in tqdm(range(1, seq_len, time_skips)):

        # We want to find the best grid
        # To do so, find where the last timestep is in pc space and center a grid around that
        latest_t = reduced_traj[t, :]

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
        x_0_flow = grid - temp_act[t, :]

        # In order to get next step activity for grid, each region not being represented
        # By the grid will assume their values from the original trajectory passed in at 
        # Each timestep. Here we will gather activity for the grid and properly append the 
        # non-specified regions to the activity tensor to get the full activation for the network

        with torch.no_grad():

            jac_rec = linearize_trajectory(mrnn, trajectory.squeeze()[t, :], *region_list, alpha=alpha, t=t)
            h = temp_act[t, :] + (jac_rec @ x_0_flow.T).T

        # Reshape data back to grid
        data_coords = data_coords
        data_coords_list.append(data_coords)

        cur_region_h = x_pca.transform(h)
        cur_region_h = torch.reshape(cur_region_h, (num_points, num_points, cur_region_h.shape[-1]))

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
